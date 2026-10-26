"""Bounded, model-bound CPU prefix snapshots with accounted borrowing.

The byte budget covers compact resident tensors, publication reservations and
in-scope leased copies, not model weights, inference workspace or allocator RSS.
Lease values must not be retained after the context exits. Callers never receive
the resident tensors, so mutation of an input or borrowed copy cannot poison it.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import re
import threading

import torch

from .ddp_cache import KVCache
from .gqa_config import cache_head_count
from .training_models import (
    base_model_config,
    model_config_mapping,
    training_model_fingerprint,
)

MAX_PREFIX_BYTES = 64 * 1024 * 1024
MAX_PREFIX_ENTRIES = 128
CONTEXT_POLICY = "left-truncate-reset-positions-v1"


@dataclass(frozen=True)
class PrefixIdentity:
    namespace: str
    model_sha256: str
    architecture_sha256: str
    dtype: torch.dtype
    layers: int
    kv_heads: int
    head_dim: int
    context_length: int
    vocab_size: int
    context_policy: str = CONTEXT_POLICY

    def __post_init__(self):
        if (
            type(self.namespace) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.namespace) is None
        ):
            raise ValueError("prefix namespace must be a bounded trusted identifier")
        for value in (self.model_sha256, self.architecture_sha256):
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("prefix identity requires SHA-256 model contracts")
        if (
            self.dtype not in (torch.float32, torch.float64)
            or self.context_policy != CONTEXT_POLICY
        ):
            raise ValueError(
                "prefix cache requires the supported CPU dtype/context policy"
            )
        for value, upper in (
            (self.layers, 64),
            (self.kv_heads, 4096),
            (self.head_dim, 4096),
            (self.context_length, 4096),
            (self.vocab_size, 1_000_000),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("prefix dimensions exceed their bounded contract")

    def bytes_for(self, length, *, logits=True):
        if type(length) is not int or not 1 <= length <= self.context_length:
            raise ValueError("prefix length must fit the model context")
        element_size = 8 if self.dtype == torch.float64 else 4
        return element_size * (
            2 * self.layers * self.kv_heads * length * self.head_dim
            + (self.vocab_size if logits else 0)
        )


def prefix_identity(model, *, namespace="default"):
    """Bind both weights and full architecture, including baseline norm epsilon."""
    base = base_model_config(model)
    if model.wte.weight.device.type != "cpu":
        raise ValueError("prefix cache supports CPU snapshots only")
    architecture = json.dumps(
        model_config_mapping(model.config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return PrefixIdentity(
        namespace,
        training_model_fingerprint(model),
        sha256(architecture.encode()).hexdigest(),
        model.wte.weight.dtype,
        len(model.h),
        cache_head_count(model),
        base.head_dim,
        base.context_length,
        base.vocab_size,
    )


@dataclass(frozen=True)
class PrefixMatch:
    cache: KVCache
    last_logits: torch.Tensor | None
    prefix_length: int


def _tokens(identity, tokens):
    if not isinstance(identity, PrefixIdentity):
        raise TypeError("prefix cache requires a PrefixIdentity")
    if (
        type(tokens) is not tuple
        or not 1 <= len(tokens) <= identity.context_length
        or any(
            type(token) is not int or not 0 <= token < identity.vocab_size
            for token in tokens
        )
    ):
        raise ValueError("prefix tokens must be a bounded exact CPU token tuple")


def _copy_value(cache, logits, length):
    result = KVCache(
        cache.model_sha256,
        tuple(
            value[:, :, :length, :]
            .detach()
            .clone(memory_format=torch.contiguous_format)
            for value in cache.keys
        ),
        tuple(
            value[:, :, :length, :]
            .detach()
            .clone(memory_format=torch.contiguous_format)
            for value in cache.values
        ),
    )
    return PrefixMatch(
        result,
        None
        if logits is None
        else logits.detach().clone(memory_format=torch.contiguous_format),
        length,
    )


def _storage_bytes(value):
    tensors = (*value.cache.keys, *value.cache.values)
    if value.last_logits is not None:
        tensors += (value.last_logits,)
    return sum(
        {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in tensors
        }.values()
    )


class PrefixKVStore:
    """Thread-safe LRU of compact prefix snapshots, with nonblocking admission."""

    def __init__(self, *, max_bytes, max_entries=64):
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_PREFIX_BYTES:
            raise ValueError("prefix byte budget must be in [1, 64 MiB]")
        if type(max_entries) is not int or not 1 <= max_entries <= MAX_PREFIX_ENTRIES:
            raise ValueError("prefix entry budget must be in [1, 128]")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._lock = threading.RLock()
        self._entries = OrderedDict()
        self._resident = self._leased = self._reserved = self._active = 0
        self._counts = dict.fromkeys(
            ("hits", "misses", "bypasses", "publications", "evictions"), 0
        )

    def _reserve(self, size, *, protected=None, entry=False):
        if size > self.max_bytes:
            return False
        while (
            self._resident + self._leased + self._reserved + size > self.max_bytes
            or (entry and len(self._entries) >= self.max_entries)
        ):
            victim = next((key for key in self._entries if key != protected), None)
            if victim is None:
                return False
            _, amount = self._entries.pop(victim)
            self._resident -= amount
            self._counts["evictions"] += 1
        self._reserved += size
        return True

    def publish(self, identity, tokens, cache, last_logits):
        """Copy a complete verified prefill; lack of space safely bypasses caching."""
        _tokens(identity, tokens)
        size = identity.bytes_for(len(tokens))
        if size > self.max_bytes:
            with self._lock:
                self._counts["bypasses"] += 1
            return False
        shape = (1, identity.kv_heads, len(tokens), identity.head_dim)
        if (
            not isinstance(cache, KVCache)
            or cache.model_sha256 != identity.model_sha256
            or len(cache.keys) != identity.layers
            or any(
                value.shape != shape
                or value.dtype != identity.dtype
                or value.device.type != "cpu"
                or not bool(torch.isfinite(value).all())
                for value in (*cache.keys, *cache.values)
            )
        ):
            raise ValueError(
                "published prefix KV state differs from its model contract"
            )
        if (
            not isinstance(last_logits, torch.Tensor)
            or last_logits.shape != (1, identity.vocab_size)
            or last_logits.dtype != identity.dtype
            or last_logits.device.type != "cpu"
            or not bool(torch.isfinite(last_logits).all())
        ):
            raise ValueError("published prefix requires finite raw last-token logits")
        key = (identity, tokens)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return False
            if not self._reserve(size, entry=True):
                self._counts["bypasses"] += 1
                return False
            try:
                value = _copy_value(cache, last_logits, len(tokens))
                if _storage_bytes(value) != size:
                    raise ValueError(
                        "prefix snapshot storage differs from its reservation"
                    )
                self._entries[key] = (value, size)
                self._resident += size
                self._counts["publications"] += 1
                return True
            finally:
                self._reserved -= size

    @contextmanager
    def acquire(self, identity, tokens):
        """Borrow an isolated compatible longest-prefix copy for this context only."""
        _tokens(identity, tokens)
        size = 0
        value = None
        with self._lock:
            selected, length, exact = None, 0, False
            for key in self._entries:
                owner, prefix = key
                if owner != identity:
                    continue
                common = 0
                for old, new in zip(prefix, tokens):
                    if old != new:
                        break
                    common += 1
                same = prefix == tokens
                if common == len(tokens) and not same:
                    common -= 1  # No saved logits for an interior prefix position.
                if common > length or (common == length and same and not exact):
                    selected, length, exact = key, common, same
            if selected is None or not length:
                self._counts["misses"] += 1
            else:
                size = identity.bytes_for(length, logits=exact)
                if self._reserve(size, protected=selected):
                    try:
                        resident, _ = self._entries[selected]
                        value = _copy_value(
                            resident.cache,
                            resident.last_logits if exact else None,
                            length,
                        )
                        if _storage_bytes(value) != size:
                            raise ValueError(
                                "prefix lease storage differs from reservation"
                            )
                        self._leased += size
                        self._active += 1
                        self._entries.move_to_end(selected)
                        self._counts["hits"] += 1
                    finally:
                        self._reserved -= size
                else:
                    size = 0
                    self._counts["bypasses"] += 1
        try:
            yield value
        finally:
            if value is not None:
                with self._lock:
                    self._leased -= size
                    self._active -= 1

    def clear(self):
        """Evict residents; existing isolated leases stay charged until released."""
        with self._lock:
            self._counts["evictions"] += len(self._entries)
            self._entries.clear()
            self._resident = 0

    def snapshot(self):
        with self._lock:
            return {
                **self._counts,
                "entries": len(self._entries),
                "resident_bytes": self._resident,
                "leased_bytes": self._leased,
                "reserved_bytes": self._reserved,
                "active_leases": self._active,
                "max_bytes": self.max_bytes,
                "max_entries": self.max_entries,
            }
