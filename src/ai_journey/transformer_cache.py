"""Validated, restartable inference caches for the small decoder lab."""

from __future__ import annotations

import re
import base64
import binascii
import json
import math
from hashlib import sha256

import torch
from torch import Tensor
from torch.nn import functional as F

from .transformer_lab import (
    CausalSelfAttention,
    DecoderLanguageModel,
    TransformerLabError,
)


class LayerKV:
    """Own immutable copies of batched head-major keys and values."""

    __slots__ = ("_keys", "_values")

    def __init__(self, keys: Tensor, values: Tensor) -> None:
        if not isinstance(keys, Tensor) or not isinstance(values, Tensor):
            raise TypeError("keys and values must be tensors")
        if keys.ndim != 4 or any(size <= 0 for size in keys.shape):
            raise TransformerLabError(
                "keys must have nonempty [batch, heads, time, dim] shape"
            )
        if (
            keys.shape != values.shape
            or keys.dtype != values.dtype
            or keys.device != values.device
        ):
            raise TransformerLabError(
                "keys and values must match shape, dtype, and device"
            )
        if (
            not keys.is_floating_point()
            or not torch.isfinite(keys).all()
            or not torch.isfinite(values).all()
        ):
            raise TransformerLabError(
                "cache tensors must be finite floating-point values"
            )
        self._keys = keys.detach().clone()
        self._values = values.detach().clone()

    @property
    def keys(self) -> Tensor:
        return self._keys.clone()

    @property
    def values(self) -> Tensor:
        return self._values.clone()


class DecoderCache:
    """Own token history and per-layer state bound to one model version."""

    __slots__ = ("_tokens", "_layers", "_config_digest", "_model_digest")

    def __init__(
        self,
        tokens: Tensor,
        layers: tuple[LayerKV, ...],
        *,
        config_digest: str,
        model_digest: str,
    ) -> None:
        if (
            not isinstance(tokens, Tensor)
            or tokens.ndim != 2
            or tokens.dtype != torch.long
        ):
            raise TypeError("tokens must be a two-dimensional torch.long tensor")
        if any(size <= 0 for size in tokens.shape) or torch.any(tokens < 0):
            raise TransformerLabError("cache tokens must be nonempty and nonnegative")
        if (
            not isinstance(layers, tuple)
            or not layers
            or not all(isinstance(layer, LayerKV) for layer in layers)
        ):
            raise TypeError("layers must be a nonempty tuple of LayerKV states")
        shape = layers[0]._keys.shape
        for layer in layers:
            if (
                layer._keys.shape != shape
                or layer._keys.dtype != layers[0]._keys.dtype
                or layer._keys.device != tokens.device
            ):
                raise TransformerLabError(
                    "cache layers must agree on shape, dtype, and token device"
                )
        if (shape[0], shape[2]) != tuple(tokens.shape):
            raise TransformerLabError("cache layers must match token batch and time")
        for digest in (config_digest, model_digest):
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise TransformerLabError("cache identity must be a SHA-256 digest")
        self._tokens = tokens.detach().clone()
        self._layers = tuple(LayerKV(layer._keys, layer._values) for layer in layers)
        self._config_digest = config_digest
        self._model_digest = model_digest

    @property
    def tokens(self) -> Tensor:
        return self._tokens.clone()

    @property
    def layers(self) -> tuple[LayerKV, ...]:
        return tuple(LayerKV(layer._keys, layer._values) for layer in self._layers)

    @property
    def config_digest(self) -> str:
        return self._config_digest

    @property
    def model_digest(self) -> str:
        return self._model_digest


@torch.no_grad()
def cached_attention(
    attention: CausalSelfAttention,
    inputs: Tensor,
    cache: LayerKV | None = None,
) -> tuple[Tensor, LayerKV]:
    """Evaluate a causal attention chunk against a head-major prefix cache."""

    if not isinstance(attention, CausalSelfAttention):
        raise TypeError("attention must be CausalSelfAttention")
    if any(module.training for module in attention.modules()):
        raise TransformerLabError("cached attention requires evaluation mode")
    weight = attention.query_key_value.weight
    if (
        not isinstance(inputs, Tensor)
        or inputs.ndim != 3
        or any(size <= 0 for size in inputs.shape)
        or inputs.shape[-1] != weight.shape[1]
        or inputs.dtype != weight.dtype
        or inputs.device != weight.device
        or not torch.isfinite(inputs).all()
    ):
        raise TransformerLabError(
            "attention inputs must be finite and match model shape, dtype, and device"
        )
    batch, time, channels = inputs.shape
    query, key, value = attention.query_key_value(inputs).chunk(3, dim=-1)
    shape = (batch, time, attention.head_count, attention.head_dim)
    query = query.view(shape).transpose(1, 2)
    key = key.view(shape).transpose(1, 2)
    value = value.view(shape).transpose(1, 2)
    prefix = 0
    if cache is not None:
        if not isinstance(cache, LayerKV):
            raise TypeError("cache must be LayerKV")
        old = cache._keys
        if (
            (old.shape[0], old.shape[1], old.shape[3])
            != (batch, attention.head_count, attention.head_dim)
            or old.dtype != inputs.dtype
            or old.device != inputs.device
        ):
            raise TransformerLabError(
                "attention cache shape, dtype, or device mismatch"
            )
        prefix = old.shape[2]
        key = torch.cat((old, key), dim=2)
        value = torch.cat((cache._values, value), dim=2)
    if prefix + time > attention.causal_mask.shape[0]:
        raise TransformerLabError("cached attention exceeds configured block_size")
    query_positions = torch.arange(prefix, prefix + time, device=inputs.device)
    key_positions = torch.arange(prefix + time, device=inputs.device)
    allowed = key_positions[None, :] <= query_positions[:, None]
    scores = query @ key.transpose(-2, -1) * attention.head_dim**-0.5
    weights = F.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
    output = (weights @ value).transpose(1, 2).contiguous().view(batch, time, channels)
    return attention.projection(output), LayerKV(key, value)


def _model_digest(model: DecoderLanguageModel) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(data.dtype).encode())
        digest.update(str(tuple(data.shape)).encode())
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_model_tokens(model: DecoderLanguageModel, tokens: Tensor) -> None:
    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be DecoderLanguageModel")
    if any(module.training for module in model.modules()):
        raise TransformerLabError("decoder cache requires evaluation mode")
    if model.config.normalization_mode != "layer_norm":
        raise TransformerLabError("decoder cache currently requires layer_norm")
    if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
        raise TypeError("tokens must be a two-dimensional torch.long tensor")
    if (
        any(size <= 0 for size in tokens.shape)
        or tokens.device != model.token_embedding.weight.device
        or torch.any(tokens < 0)
        or torch.any(tokens >= model.config.vocab_size)
    ):
        raise TransformerLabError(
            "tokens must be nonempty, in vocabulary, and on the model device"
        )


def _validate_binding(model: DecoderLanguageModel, cache: DecoderCache) -> None:
    if not isinstance(cache, DecoderCache):
        raise TypeError("cache must be DecoderCache")
    if (
        cache.config_digest != model.config.fingerprint()
        or cache.model_digest != _model_digest(model)
    ):
        raise TransformerLabError("cache belongs to a different or modified model")
    shape = cache._layers[0]._keys.shape
    if (
        len(cache._layers) != model.config.layer_count
        or shape[1] != model.config.head_count
        or shape[3] != model.config.head_dim
        or shape[2] > model.config.block_size
        or cache._layers[0]._keys.dtype != model.token_embedding.weight.dtype
        or cache._tokens.device != model.token_embedding.weight.device
    ):
        raise TransformerLabError("cache architecture, dtype, or device mismatch")


@torch.no_grad()
def _forward_chunk(
    model: DecoderLanguageModel, tokens: Tensor, cache: DecoderCache | None
) -> tuple[Tensor, DecoderCache]:
    prefix = 0 if cache is None else cache._tokens.shape[1]
    time = tokens.shape[1]
    if prefix + time > model.config.block_size:
        raise TransformerLabError("decoder cache exceeds configured block_size")
    positions = torch.arange(prefix, prefix + time, device=tokens.device)
    hidden = model.token_embedding(tokens) + model.position_embedding(positions)
    layers = []
    placement = getattr(model.config, "normalization_placement", "pre")
    for index, block in enumerate(model.blocks):
        old = None if cache is None else cache._layers[index]
        if placement == "pre":
            attended, layer = cached_attention(
                block.attention, block.attention_norm(hidden), old
            )
            hidden = hidden + attended
            hidden = hidden + block.feed_forward(block.feed_forward_norm(hidden))
        elif placement == "post":
            attended, layer = cached_attention(block.attention, hidden, old)
            hidden = block.attention_norm(hidden + attended)
            hidden = block.feed_forward_norm(hidden + block.feed_forward(hidden))
        else:
            raise TransformerLabError("unsupported normalization placement")
        layers.append(layer)
    logits = model.lm_head(model.final_norm(hidden))
    if not torch.isfinite(logits).all():
        raise TransformerLabError("decoder produced nonfinite logits")
    history = tokens if cache is None else torch.cat((cache._tokens, tokens), dim=1)
    state = DecoderCache(
        history,
        tuple(layers),
        config_digest=model.config.fingerprint(),
        model_digest=_model_digest(model),
    )
    return logits, state


def prefill(model: DecoderLanguageModel, tokens: Tensor) -> tuple[Tensor, DecoderCache]:
    """Evaluate a complete prompt and retain each layer's causal prefix state."""
    _validate_model_tokens(model, tokens)
    return _forward_chunk(model, tokens, None)


def decode(
    model: DecoderLanguageModel, tokens: Tensor, cache: DecoderCache
) -> tuple[Tensor, DecoderCache]:
    """Decode a chunk, rebuilding cropped contexts when learned positions reset.

    Dropping only the oldest keys is incorrect: every retained token changes
    position and its layer activations may still depend on the evicted prefix.
    """
    _validate_model_tokens(model, tokens)
    _validate_binding(model, cache)
    if tokens.shape[0] != cache._tokens.shape[0]:
        raise TransformerLabError("decode token batch must match cache")
    if cache._tokens.shape[1] + tokens.shape[1] > model.config.block_size:
        outputs = []
        state = cache
        for token in tokens.split(1, dim=1):
            if state._tokens.shape[1] < model.config.block_size:
                output, state = _forward_chunk(model, token, state)
            else:
                cropped = torch.cat((state._tokens, token), dim=1)[
                    :, -model.config.block_size :
                ]
                rebuilt, state = _forward_chunk(model, cropped, None)
                output = rebuilt[:, -1:]
            outputs.append(output)
        return torch.cat(outputs, dim=1), state
    return _forward_chunk(model, tokens, cache)


def migrate_cache(
    cache: DecoderCache,
    source_model: DecoderLanguageModel,
    target_model: DecoderLanguageModel,
) -> DecoderCache:
    """Rebuild inference state for an explicitly converted copy of the model.

    Casting cached activations alone is unsafe because rounded parameters and
    activations change the result. Verify conversion provenance, then prefill.
    """

    _validate_model_tokens(source_model, cache.tokens)
    _validate_binding(source_model, cache)
    tokens = cache.tokens.to(target_model.token_embedding.weight.device)
    _validate_model_tokens(target_model, tokens)
    if source_model.config != target_model.config:
        raise TransformerLabError(
            "cache migration requires identical model configuration"
        )
    source = source_model.state_dict()
    target = target_model.state_dict()
    if source.keys() != target.keys():
        raise TransformerLabError("cache migration requires matching state names")
    for name, value in source.items():
        expected = value.to(dtype=target[name].dtype, device=target[name].device)
        if not torch.equal(expected, target[name]):
            raise TransformerLabError(
                "target model is not a dtype/device conversion of source"
            )
    _, migrated = prefill(target_model, tokens)
    return migrated


def reorder_cache(cache: DecoderCache, indexes: Tensor) -> DecoderCache:
    """Select, reorder, or duplicate request rows without sharing mutable storage."""

    if not isinstance(cache, DecoderCache):
        raise TypeError("cache must be DecoderCache")
    if (
        not isinstance(indexes, Tensor)
        or indexes.ndim != 1
        or indexes.dtype != torch.long
    ):
        raise TypeError("indexes must be a one-dimensional torch.long tensor")
    if (
        indexes.numel() == 0
        or indexes.device != cache._tokens.device
        or torch.any(indexes < 0)
        or torch.any(indexes >= cache._tokens.shape[0])
    ):
        raise TransformerLabError(
            "cache indexes must be nonempty, in range, and on cache device"
        )
    layers = tuple(
        LayerKV(
            layer._keys.index_select(0, indexes), layer._values.index_select(0, indexes)
        )
        for layer in cache._layers
    )
    return DecoderCache(
        cache._tokens.index_select(0, indexes),
        layers,
        config_digest=cache.config_digest,
        model_digest=cache.model_digest,
    )


_DTYPES = {
    "torch.int64": torch.int64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}
_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _encode_tensor(tensor: Tensor) -> dict[str, object]:
    data = tensor.detach().cpu().contiguous()
    raw = data.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(data.dtype),
        "shape": list(data.shape),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def cache_to_bytes(cache: DecoderCache) -> bytes:
    """Serialize CPU-portable tensor bytes and checksums without pickle.

    SHA-256 detects corruption, not authenticity. Restore against a trusted
    model before accepting a snapshot for inference.
    """
    if not isinstance(cache, DecoderCache):
        raise TypeError("cache must be DecoderCache")
    payload = {
        "tokens": _encode_tensor(cache._tokens),
        "layers": [
            {
                "keys": _encode_tensor(layer._keys),
                "values": _encode_tensor(layer._values),
            }
            for layer in cache._layers
        ],
        "config_digest": cache.config_digest,
        "model_digest": cache.model_digest,
    }
    result = _json_bytes(
        {
            "version": 1,
            "sha256": sha256(_json_bytes(payload)).hexdigest(),
            "payload": payload,
        }
    )
    if len(result) > _MAX_SNAPSHOT_BYTES:
        raise TransformerLabError("cache snapshot exceeds 64 MiB limit")
    return result


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise TransformerLabError("duplicate cache snapshot field")
        result[key] = value
    return result


def _fields(value: object, expected: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise TransformerLabError("cache snapshot fields do not match schema")
    return value


def _decode_tensor(value: object, *, dimensions: int, floating: bool) -> Tensor:
    entry = _fields(value, {"dtype", "shape", "data"})
    dtype = entry["dtype"]
    shape = entry["shape"]
    data = entry["data"]
    if (
        not isinstance(dtype, str)
        or dtype not in _DTYPES
        or (dtype == "torch.int64") == floating
        or not isinstance(shape, list)
        or len(shape) != dimensions
        or any(type(size) is not int or size <= 0 for size in shape)
        or not isinstance(data, str)
    ):
        raise TransformerLabError("invalid snapshot tensor dtype, shape, or bytes")
    expected = math.prod(shape) * torch.empty((), dtype=_DTYPES[dtype]).element_size()
    if expected > _MAX_SNAPSHOT_BYTES:
        raise TransformerLabError("cache snapshot tensor exceeds size limit")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TransformerLabError("invalid snapshot tensor encoding") from exc
    if len(raw) != expected:
        raise TransformerLabError("snapshot tensor byte count does not match shape")
    return torch.frombuffer(bytearray(raw), dtype=_DTYPES[dtype]).reshape(shape).clone()


def cache_from_bytes(data: bytes) -> DecoderCache:
    """Decode a bounded, strictly validated, CPU-owned snapshot."""
    if not isinstance(data, bytes):
        raise TypeError("snapshot must be bytes")
    if len(data) > _MAX_SNAPSHOT_BYTES:
        raise TransformerLabError("cache snapshot exceeds 64 MiB limit")
    try:
        envelope = json.loads(data.decode("utf-8"), object_pairs_hook=_strict_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TransformerLabError("invalid cache snapshot JSON") from exc
    envelope = _fields(envelope, {"version", "sha256", "payload"})
    if type(envelope["version"]) is not int or envelope["version"] != 1:
        raise TransformerLabError("unsupported cache snapshot version")
    payload = _fields(
        envelope["payload"], {"tokens", "layers", "config_digest", "model_digest"}
    )
    try:
        digest = sha256(_json_bytes(payload)).hexdigest()
    except (ValueError, RecursionError) as exc:
        raise TransformerLabError("invalid cache snapshot payload") from exc
    if envelope["sha256"] != digest:
        raise TransformerLabError("cache snapshot checksum mismatch")
    entries = payload["layers"]
    if not isinstance(entries, list) or not entries:
        raise TransformerLabError("cache snapshot requires layer entries")
    layers = []
    for entry in entries:
        entry = _fields(entry, {"keys", "values"})
        layers.append(
            LayerKV(
                _decode_tensor(entry["keys"], dimensions=4, floating=True),
                _decode_tensor(entry["values"], dimensions=4, floating=True),
            )
        )
    return DecoderCache(
        _decode_tensor(payload["tokens"], dimensions=2, floating=False),
        tuple(layers),
        config_digest=payload["config_digest"],
        model_digest=payload["model_digest"],
    )


def restore_cache(model: DecoderLanguageModel, data: bytes) -> DecoderCache:
    """Validate a snapshot against trusted weights before publishing live state.

    Recomputing the retained context is intentional: a checksum is not proof
    that saved keys and values were produced by the stated model. All work is
    local until validation succeeds, and neither model nor caller state changes.
    """
    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be DecoderLanguageModel")
    parsed = cache_from_bytes(data)
    device = model.token_embedding.weight.device
    tokens = parsed.tokens.to(device)
    _validate_model_tokens(model, tokens)
    layers = tuple(
        LayerKV(layer._keys.to(device), layer._values.to(device))
        for layer in parsed._layers
    )
    candidate = DecoderCache(
        tokens,
        layers,
        config_digest=parsed.config_digest,
        model_digest=parsed.model_digest,
    )
    _validate_binding(model, candidate)
    _, verified = prefill(model, tokens)
    for saved, expected in zip(candidate._layers, verified._layers, strict=True):
        tolerance = torch.finfo(expected._keys.dtype).eps * 32
        for actual_tensor, expected_tensor in (
            (saved._keys, expected._keys),
            (saved._values, expected._values),
        ):
            if not torch.allclose(
                actual_tensor, expected_tensor, rtol=tolerance, atol=tolerance
            ):
                raise TransformerLabError(
                    "snapshot layer state does not match trusted model context"
                )
    return verified
