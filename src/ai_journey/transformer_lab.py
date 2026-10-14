"""Deterministic, checkpointable decoder-only transformer training primitives."""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import random
import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .batch_normalization import ScratchBatchNorm
from .tokenizer_codec import BPECodec


class TransformerLabError(ValueError):
    """Raised when transformer data, configuration, or state is invalid."""


_LEGACY_TRANSFORMER_CONTROLS = {
    "normalization_placement": "pre",
    "feed_forward_expansion": 4,
    "feed_forward_activation": "gelu",
    "activation_checkpointing": False,
    "attention_backend": "manual",
}


def _require_sha256_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise TransformerLabError(f"{name} must be a SHA-256 hex digest")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and the CPU generator for deterministic CPU runs."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**32:
        raise TransformerLabError("seed must be in NumPy's range [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class TransformerConfig:
    """Validated architecture and training-independent model settings."""

    vocab_size: int
    block_size: int = 16
    embedding_dim: int = 32
    head_count: int = 4
    layer_count: int = 2
    dropout: float = 0.0
    initialization_std: float = 0.02
    initialization_mode: str = "fixed_normal"
    initialization_gain: float = math.sqrt(2.0)
    normalization_mode: str = "layer_norm"
    normalization_placement: str = "pre"
    feed_forward_expansion: int = 4
    feed_forward_activation: str = "gelu"
    activation_checkpointing: bool = False
    attention_backend: str = "manual"
    batch_norm_eps: float = 1e-5
    batch_norm_momentum: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "block_size",
            "embedding_dim",
            "head_count",
            "layer_count",
            "feed_forward_expansion",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise TransformerLabError(f"{name} must be positive")
        if self.embedding_dim % self.head_count:
            raise TransformerLabError("embedding_dim must be divisible by head_count")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0 <= self.dropout < 1
        ):
            raise TransformerLabError("dropout must be in [0, 1)")
        if (
            isinstance(self.initialization_std, bool)
            or not isinstance(self.initialization_std, (int, float))
            or not math.isfinite(self.initialization_std)
            or self.initialization_std <= 0
        ):
            raise TransformerLabError("initialization_std must be positive and finite")
        if self.initialization_mode not in {"fixed_normal", "kaiming_normal"}:
            raise TransformerLabError(
                "initialization_mode must be 'fixed_normal' or 'kaiming_normal'"
            )
        if (
            isinstance(self.initialization_gain, bool)
            or not isinstance(self.initialization_gain, (int, float))
            or not math.isfinite(self.initialization_gain)
            or self.initialization_gain <= 0
        ):
            raise TransformerLabError("initialization_gain must be positive and finite")
        if self.normalization_mode not in {"layer_norm", "scratch_batch_norm"}:
            raise TransformerLabError(
                "normalization_mode must be 'layer_norm' or 'scratch_batch_norm'"
            )
        if self.normalization_placement not in {"pre", "post"}:
            raise TransformerLabError("normalization_placement must be 'pre' or 'post'")
        if self.feed_forward_activation not in {"gelu", "relu"}:
            raise TransformerLabError(
                "feed_forward_activation must be 'gelu' or 'relu'"
            )
        if not isinstance(self.activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be a boolean")
        if self.activation_checkpointing and self.normalization_mode != "layer_norm":
            raise TransformerLabError(
                "activation checkpointing requires stateless layer_norm"
            )
        if self.attention_backend not in {"manual", "sdpa"}:
            raise TransformerLabError("attention_backend must be 'manual' or 'sdpa'")
        if (
            isinstance(self.batch_norm_eps, bool)
            or not isinstance(self.batch_norm_eps, (int, float))
            or not math.isfinite(self.batch_norm_eps)
            or self.batch_norm_eps <= 0
        ):
            raise TransformerLabError("batch_norm_eps must be positive and finite")
        if (
            isinstance(self.batch_norm_momentum, bool)
            or not isinstance(self.batch_norm_momentum, (int, float))
            or not math.isfinite(self.batch_norm_momentum)
            or not 0 < self.batch_norm_momentum <= 1
        ):
            raise TransformerLabError("batch_norm_momentum must be in (0, 1]")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    def fingerprint(self) -> str:
        settings = asdict(self)
        # Legacy defaults preserve the same equations and checkpoint identity.
        for name, default in _LEGACY_TRANSFORMER_CONTROLS.items():
            if settings.get(name) == default:
                settings.pop(name)
        payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class TrainingConfig:
    """Validated controls for deterministic CPU training."""

    steps: int = 100
    batch_size: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 24

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise TransformerLabError(f"{name} must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed < 2**32:
            raise TransformerLabError("seed must be in NumPy's range [0, 2**32)")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise TransformerLabError(f"{name} must be positive and finite")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise TransformerLabError("weight_decay must be non-negative and finite")


@dataclass(frozen=True)
class CharacterCodec:
    """Lossless deterministic mapping between characters and token ids."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) < 2:
            raise TransformerLabError("codec requires at least two distinct tokens")
        if any(not isinstance(token, str) or len(token) != 1 for token in self.tokens):
            raise TransformerLabError("codec tokens must be single characters")
        if tuple(sorted(set(self.tokens))) != self.tokens:
            raise TransformerLabError("codec tokens must be unique and sorted")

    @classmethod
    def from_text(cls, text: str) -> CharacterCodec:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return cls(tuple(sorted(set(text))))

    def encode(self, text: str) -> tuple[int, ...]:
        indexes = {token: index for index, token in enumerate(self.tokens)}
        try:
            return tuple(indexes[token] for token in text)
        except KeyError as exc:
            raise TransformerLabError(f"unknown character: {exc.args[0]!r}") from exc

    def decode(self, token_ids: tuple[int, ...] | list[int]) -> str:
        decoded: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token ids must be integers")
            if not 0 <= token_id < len(self.tokens):
                raise TransformerLabError(f"token id out of range: {token_id}")
            decoded.append(self.tokens[token_id])
        return "".join(decoded)

    def fingerprint(self) -> str:
        return sha256("".join(self.tokens).encode()).hexdigest()

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class TokenCorpus:
    """Tokenized public text with immutable provenance and held-out split."""

    codec: CharacterCodec | BPECodec
    train_tokens: Tensor
    validation_tokens: Tensor
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.codec, (CharacterCodec, BPECodec)):
            raise TypeError("codec must be CharacterCodec or BPECodec")
        _require_sha256_digest(self.source_sha256, "source_sha256")
        for name in ("train_tokens", "validation_tokens"):
            tokens = getattr(self, name)
            if (
                not isinstance(tokens, Tensor)
                or tokens.layout != torch.strided
                or tokens.ndim != 1
                or tokens.dtype != torch.long
                or tokens.device.type != "cpu"
            ):
                raise TypeError(
                    f"{name} must be a dense one-dimensional CPU long tensor"
                )
            if tokens.numel() < 2 or bool(
                ((tokens < 0) | (tokens >= self.vocab_size)).any()
            ):
                raise TransformerLabError(f"{name} must contain valid vocabulary ids")
            object.__setattr__(self, name, tokens.detach().clone())

    @classmethod
    def from_path(
        cls, path: Path, *, validation_fraction: float = 0.2, block_size: int = 16
    ) -> TokenCorpus:
        if not isinstance(path, Path):
            raise TypeError("path must be pathlib.Path")
        if (
            isinstance(validation_fraction, bool)
            or not isinstance(validation_fraction, (int, float))
            or not 0 < validation_fraction < 1
        ):
            raise TransformerLabError(
                "validation_fraction must be between zero and one"
            )
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise TypeError("block_size must be an integer")
        if block_size <= 0:
            raise TransformerLabError("block_size must be positive")
        source = path.read_bytes()
        text = source.decode("utf-8")
        codec = CharacterCodec.from_text(text)
        tokens = torch.tensor(codec.encode(text), dtype=torch.long)
        validation_size = max(block_size + 1, round(len(tokens) * validation_fraction))
        split = len(tokens) - validation_size
        if split <= block_size:
            raise TransformerLabError("corpus is too small for the requested split")
        return cls(
            codec=codec,
            train_tokens=tokens[:split].clone(),
            validation_tokens=tokens[split:].clone(),
            source_sha256=sha256(source).hexdigest(),
        )

    @property
    def vocab_size(self) -> int:
        return self.codec.vocab_size

    def fingerprint(self) -> str:
        digest = sha256()
        digest.update(self.source_sha256.encode())
        digest.update(self.codec.fingerprint().encode())
        digest.update(b"partition-lengths-v2:")
        digest.update(
            f"{len(self.train_tokens)}:{len(self.validation_tokens)}:".encode()
        )
        digest.update(self.train_tokens.numpy().tobytes())
        digest.update(self.validation_tokens.numpy().tobytes())
        return digest.hexdigest()


class BatchCursor:
    """Deterministic shuffled language-model batches with resumable position."""

    def __init__(
        self, tokens: Tensor, *, block_size: int, batch_size: int, seed: int = 0
    ) -> None:
        if tokens.ndim != 1 or tokens.dtype != torch.long:
            raise TypeError("tokens must be a one-dimensional torch.long tensor")
        for name, value in (("block_size", block_size), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise TransformerLabError(f"{name} must be positive")
        if len(tokens) <= block_size:
            raise TransformerLabError("token stream must be longer than block_size")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= seed < 2**63:
            raise TransformerLabError("cursor seed must be in [0, 2**63)")
        self.tokens = tokens.clone()
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.offset = 0

    @property
    def sample_count(self) -> int:
        return len(self.tokens) - self.block_size

    def _order(self) -> Tensor:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.sample_count, generator=generator)

    def next(self) -> tuple[Tensor, Tensor]:
        if self.offset >= self.sample_count:
            self.epoch += 1
            self.offset = 0
        order = self._order()
        starts = order[self.offset : self.offset + self.batch_size]
        self.offset += len(starts)
        x = torch.stack(
            [self.tokens[start : start + self.block_size] for start in starts]
        )
        y = torch.stack(
            [self.tokens[start + 1 : start + self.block_size + 1] for start in starts]
        )
        return x, y

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}

    def fingerprint(self) -> str:
        """Bind restart position to the actual stream and sampler controls."""
        digest = sha256(
            json.dumps(
                {
                    "block_size": self.block_size,
                    "batch_size": self.batch_size,
                    "seed": self.seed,
                },
                sort_keys=True,
            ).encode()
        )
        digest.update(self.tokens.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def load_state_dict(self, state: dict[str, int]) -> None:
        if set(state) != {"epoch", "offset"}:
            raise TransformerLabError("batch cursor state has invalid fields")
        epoch, offset = state["epoch"], state["offset"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in state.values()
        ):
            raise TypeError("batch cursor state values must be integers")
        if epoch < 0 or not 0 <= offset <= self.sample_count:
            raise TransformerLabError("batch cursor state is out of range")
        if epoch + self.seed >= 2**63:
            raise TransformerLabError("batch cursor epoch exceeds its RNG seed range")
        if offset != self.sample_count and offset % self.batch_size:
            raise TransformerLabError("batch cursor offset is not a batch boundary")
        self.epoch = epoch
        self.offset = offset


def valid_sequence_positions(
    lengths: Tensor, *, batch: int, time: int, device: torch.device
) -> Tensor:
    """Validate positive right-padded lengths and return the valid-token mask."""

    if (
        not isinstance(lengths, Tensor)
        or lengths.dtype != torch.long
        or lengths.shape != (batch,)
    ):
        raise TypeError("lengths must be a torch.long tensor with shape (batch,)")
    if lengths.device != device:
        raise TransformerLabError("lengths must match the input device")
    if bool(((lengths <= 0) | (lengths > time)).any()):
        raise TransformerLabError(
            "lengths must be positive and not exceed sequence time"
        )
    return torch.arange(time, device=device).unsqueeze(0) < lengths.unsqueeze(1)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit causal mask."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.head_count = config.head_count
        self.head_dim = config.head_dim
        self.attention_backend = config.attention_backend
        self.query_key_value = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, inputs: Tensor, *, lengths: Tensor | None = None) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("attention inputs must be a torch.Tensor")
        if inputs.ndim != 3:
            raise TransformerLabError(
                "attention inputs must have shape (batch, time, channels)"
            )
        batch, time, channels = inputs.shape
        if batch == 0 or time == 0:
            raise TransformerLabError(
                "attention batch and time dimensions must be non-empty"
            )
        if channels != self.query_key_value.in_features:
            raise TransformerLabError("attention input width must match embedding_dim")
        if not inputs.is_floating_point():
            raise TypeError("attention inputs must have a floating-point dtype")
        weight = self.query_key_value.weight
        if inputs.dtype != weight.dtype or inputs.device != weight.device:
            raise TransformerLabError(
                "attention inputs must match parameter dtype and device"
            )
        if not bool(torch.isfinite(inputs).all()):
            raise TransformerLabError("attention inputs must be finite")
        if time > self.causal_mask.shape[0]:
            raise TransformerLabError("sequence exceeds configured block_size")
        valid = (
            None
            if lengths is None
            else valid_sequence_positions(
                lengths, batch=batch, time=time, device=inputs.device
            )
        )
        qkv = self.query_key_value(inputs)
        if not bool(torch.isfinite(qkv).all()):
            raise TransformerLabError("attention projections must be finite")
        query, key, value = qkv.chunk(3, dim=-1)
        shape = (batch, time, self.head_count, self.head_dim)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        mask = self.causal_mask[:time, :time]
        if valid is not None:
            mask = mask[None, None, :, :] & valid[:, None, None, :]
        if self.attention_backend == "sdpa":
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None if valid is None else mask,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=valid is None,
            )
        else:
            scores = query @ key.transpose(-2, -1) * self.head_dim**-0.5
            if not bool(torch.isfinite(scores).all()):
                raise TransformerLabError("attention scores must be finite")
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = self.attention_dropout(F.softmax(scores, dim=-1))
            attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        output = self.residual_dropout(self.projection(attended))
        if valid is not None:
            output = output.masked_fill(~valid[:, :, None], 0.0)
        if not bool(torch.isfinite(output).all()):
            raise TransformerLabError("attention output must be finite")
        return output


class FeedForward(nn.Module):
    """Transformer position-wise MLP with a validated hidden expansion."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden_dim = config.feed_forward_expansion * config.embedding_dim
        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.GELU() if config.feed_forward_activation == "gelu" else nn.ReLU(),
            nn.Linear(hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


def build_normalization(config: TransformerConfig) -> nn.Module:
    """Build the configured last-dimension normalization layer."""

    if not isinstance(config, TransformerConfig):
        raise TypeError("config must be TransformerConfig")
    if config.normalization_mode == "scratch_batch_norm":
        return ScratchBatchNorm(
            config.embedding_dim,
            eps=config.batch_norm_eps,
            momentum=config.batch_norm_momentum,
        )
    return nn.LayerNorm(config.embedding_dim)


class TransformerBlock(nn.Module):
    """Configurable pre- or post-normalized attention and MLP residual block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.normalization_mode = config.normalization_mode
        self.normalization_placement = config.normalization_placement
        self.attention_norm = build_normalization(config)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = build_normalization(config)
        self.feed_forward = FeedForward(config)

    def forward(self, inputs: Tensor, *, lengths: Tensor | None = None) -> Tensor:
        valid = None
        if lengths is not None:
            if self.normalization_mode != "layer_norm":
                raise TransformerLabError("padded blocks require layer_norm")
            valid = valid_sequence_positions(
                lengths,
                batch=inputs.shape[0],
                time=inputs.shape[1],
                device=inputs.device,
            )
        if self.normalization_placement == "pre":
            inputs = inputs + self.attention(
                self.attention_norm(inputs), lengths=lengths
            )
            output = inputs + self.feed_forward(self.feed_forward_norm(inputs))
        else:
            inputs = self.attention_norm(
                inputs + self.attention(inputs, lengths=lengths)
            )
            output = self.feed_forward_norm(inputs + self.feed_forward(inputs))
        return output if valid is None else output.masked_fill(~valid[:, :, None], 0.0)


def expected_initialization_std(module: nn.Module, config: TransformerConfig) -> float:
    """Return the configured standard deviation for an initialized weight."""

    if not isinstance(module, (nn.Linear, nn.Embedding)):
        raise TypeError("module must be torch.nn.Linear or torch.nn.Embedding")
    if config.initialization_mode == "kaiming_normal" and isinstance(module, nn.Linear):
        fan_in = module.weight.shape[1]
        return config.initialization_gain / math.sqrt(fan_in)
    return config.initialization_std


def expected_parameter_count(config: TransformerConfig) -> int:
    """Count decoder parameters without allocating weights or causal masks."""
    if not isinstance(config, TransformerConfig):
        raise TypeError("config must be TransformerConfig")
    width = config.embedding_dim
    expansion = config.feed_forward_expansion
    embeddings_and_head = width * (2 * config.vocab_size + config.block_size)
    per_block = (4 + 2 * expansion) * width**2 + (9 + expansion) * width
    return embeddings_and_head + config.layer_count * per_block + 2 * width


class DecoderLanguageModel(nn.Module):
    """Small decoder-only character language model."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.layer_count)]
        )
        self.final_norm = build_normalization(config)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=expected_initialization_std(module, self.config),
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _validate_token_ids(
        self, token_ids: Tensor, *, limit_time: bool = True
    ) -> tuple[int, int]:
        if (
            not isinstance(token_ids, Tensor)
            or token_ids.ndim != 2
            or token_ids.dtype != torch.long
        ):
            raise TypeError("token_ids must be a two-dimensional torch.long tensor")
        batch, time = token_ids.shape
        if batch == 0 or time == 0:
            raise TransformerLabError(
                "token batch and time dimensions must be non-empty"
            )
        if token_ids.device != self.token_embedding.weight.device:
            raise TransformerLabError("token_ids must match the model device")
        if limit_time and time > self.config.block_size:
            raise TransformerLabError("sequence exceeds configured block_size")
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= self.config.vocab_size:
            raise TransformerLabError("token id is outside the vocabulary")
        return batch, time

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
        *,
        lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        batch, time = self._validate_token_ids(token_ids)
        valid = None
        if lengths is not None:
            if self.config.normalization_mode != "layer_norm":
                raise TransformerLabError("padded decoder batches require layer_norm")
            valid = valid_sequence_positions(
                lengths, batch=batch, time=time, device=token_ids.device
            )
        if targets is not None:
            if (
                not isinstance(targets, Tensor)
                or targets.shape != token_ids.shape
                or targets.dtype != torch.long
            ):
                raise TypeError("targets must match token_ids shape and dtype")
            if targets.device != token_ids.device:
                raise TransformerLabError("targets must match token_ids device")
            loss_targets = targets.reshape(-1) if valid is None else targets[valid]
            if bool(
                ((loss_targets < 0) | (loss_targets >= self.config.vocab_size)).any()
            ):
                raise TransformerLabError("valid target id is outside the vocabulary")
        positions = torch.arange(time, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            if (
                self.config.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = activation_checkpoint(
                    block,
                    hidden,
                    lengths=lengths,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden = block(hidden, lengths=lengths)
        logits = self.lm_head(self.final_norm(hidden))
        if valid is not None:
            logits = logits.masked_fill(~valid[:, :, None], 0.0)
        loss = None
        if targets is not None:
            loss_logits = (
                logits.reshape(-1, self.config.vocab_size)
                if valid is None
                else logits[valid]
            )
            loss = F.cross_entropy(loss_logits, loss_targets)
        return logits, loss

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        *,
        new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if isinstance(new_tokens, bool) or not isinstance(new_tokens, int):
            raise TypeError("new_tokens must be an integer")
        if new_tokens < 0:
            raise TransformerLabError("new_tokens must be non-negative")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise TransformerLabError("temperature must be positive and finite")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
        ):
            raise TransformerLabError("top_k must be a positive integer")
        self._validate_token_ids(token_ids, limit_time=False)
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            generated = token_ids.clone()
            for _ in range(new_tokens):
                context = generated[:, -self.config.block_size :]
                logits, _ = self(context)
                next_logits = logits[:, -1] / temperature
                if not bool(torch.isfinite(next_logits).all()):
                    raise TransformerLabError("sampling logits must be finite")
                if top_k is not None:
                    indexes = (
                        next_logits.argmax(dim=-1, keepdim=True)
                        if top_k == 1
                        else torch.topk(
                            next_logits, min(top_k, next_logits.shape[-1])
                        ).indices
                    )
                    allowed = torch.zeros_like(next_logits, dtype=torch.bool).scatter_(
                        1, indexes, True
                    )
                    next_logits = next_logits.masked_fill(~allowed, float("-inf"))
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, num_samples=1, generator=generator
                )
                generated = torch.cat((generated, next_token), dim=1)
            return generated
        finally:
            for module, training in modes:
                module.training = training


def build_optimizer(
    model: DecoderLanguageModel, config: TrainingConfig
) -> torch.optim.AdamW:
    """Build AdamW with decay limited to matrix-shaped parameters."""

    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    if sum(map(len, (decay, no_decay))) != len(list(model.parameters())):
        raise RuntimeError("optimizer parameter grouping is incomplete")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


@torch.no_grad()
def evaluate_nll(
    model: DecoderLanguageModel, tokens: Tensor, *, batch_size: int = 64
) -> float:
    """Evaluate every contiguous window in a token stream without shuffling."""

    if len(tokens) <= model.config.block_size:
        raise TransformerLabError("evaluation stream is too short")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise TransformerLabError("batch_size must be a positive integer")
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        total_loss = 0.0
        token_count = 0
        for start in range(0, len(tokens) - model.config.block_size, batch_size):
            indexes = range(
                start,
                min(start + batch_size, len(tokens) - model.config.block_size),
            )
            x = torch.stack([tokens[i : i + model.config.block_size] for i in indexes])
            y = torch.stack(
                [tokens[i + 1 : i + model.config.block_size + 1] for i in indexes]
            )
            logits, _ = model(x)
            per_token = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                y.reshape(-1),
                reduction="none",
            )
            if not bool(torch.isfinite(per_token).all()):
                raise TransformerLabError("evaluation losses must remain finite")
            total_loss += float(per_token.double().sum())
            token_count += per_token.numel()
        return total_loss / token_count
    finally:
        for module, training in modes:
            module.training = training


@dataclass(frozen=True)
class StepMetric:
    step: int
    loss: float
    gradient_norm: float


def _validate_optimizer_binding(
    model: DecoderLanguageModel, optimizer: torch.optim.Optimizer
) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer")
    parameters = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    expected = {id(parameter) for parameter in model.parameters()}
    if len(parameters) != len(set(parameters)) or set(parameters) != expected:
        raise TransformerLabError("optimizer parameters must match the complete model")


def train_steps(
    model: DecoderLanguageModel,
    cursor: BatchCursor,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    *,
    start_step: int = 0,
    step_count: int | None = None,
) -> tuple[StepMetric, ...]:
    """Train a bounded number of steps and return inspectable metrics."""

    _validate_optimizer_binding(model, optimizer)
    if not isinstance(cursor, BatchCursor):
        raise TypeError("cursor must be BatchCursor")
    if cursor.block_size != model.config.block_size:
        raise TransformerLabError("training cursor block_size must match the model")
    model._validate_token_ids(cursor.tokens.unsqueeze(0), limit_time=False)
    count = config.steps if step_count is None else step_count
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise TransformerLabError("step_count must be a positive integer")
    if (
        isinstance(start_step, bool)
        or not isinstance(start_step, int)
        or start_step < 0
    ):
        raise TransformerLabError("start_step must be a non-negative integer")
    caller_modes = [(module, module.training) for module in model.modules()]
    model.train()
    metrics: list[StepMetric] = []
    for step in range(start_step, start_step + count):
        # This bounded lab prioritizes retry correctness over snapshot overhead.
        saved_model = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_cursor = cursor.state_dict()
        saved_rng = torch.get_rng_state().clone()
        saved_python_rng = random.getstate()
        saved_numpy_rng = np.random.get_state()
        device = model.token_embedding.weight.device
        saved_device_rng = (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        )
        saved_gradients = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        ]
        try:
            x, y = cursor.next()
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("training loss was not computed")
            if not bool(torch.isfinite(loss)):
                raise TransformerLabError("training loss must be finite")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            _require_finite_checkpoint_state(model.state_dict(), "updated model")
            _require_finite_checkpoint_state(
                optimizer.state_dict(), "updated optimizer"
            )
        except Exception:
            model.load_state_dict(saved_model)
            optimizer.load_state_dict(saved_optimizer)
            cursor.load_state_dict(saved_cursor)
            torch.set_rng_state(saved_rng)
            random.setstate(saved_python_rng)
            np.random.set_state(saved_numpy_rng)
            if saved_device_rng is not None:
                torch.cuda.set_rng_state(saved_device_rng, device)
            for parameter, gradient in zip(
                model.parameters(), saved_gradients, strict=True
            ):
                parameter.grad = gradient
            for module, mode in caller_modes:
                module.training = mode
            raise
        metrics.append(
            StepMetric(
                step=step + 1,
                loss=float(loss.detach()),
                gradient_norm=float(gradient_norm),
            )
        )
    return tuple(metrics)


CHECKPOINT_SCHEMA_VERSION = 2
_CHECKPOINT_FIELDS = {
    "schema_version",
    "step",
    "model_config",
    "training_config",
    "corpus_fingerprint",
    "model_fingerprint",
    "model_state",
    "optimizer_state",
    "cursor_state",
    "torch_rng_state",
}


def _require_finite_checkpoint_state(value: Any, name: str) -> None:
    """Reject unusable numeric state before publication or optimizer resume."""

    if isinstance(value, Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise TransformerLabError(f"{name} must be finite")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise TransformerLabError(f"{name} must be finite")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_checkpoint_state(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_checkpoint_state(item, f"{name}[{index}]")


def _validate_adamw_checkpoint_state(optimizer: torch.optim.Optimizer) -> None:
    """Catch moments that PyTorch accepts on load but cannot safely update."""

    if not isinstance(optimizer, torch.optim.AdamW):
        return
    for group in optimizer.param_groups:
        for name in ("lr", "eps", "weight_decay"):
            value = group.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name == "eps" and value == 0)
            ):
                raise TransformerLabError(f"AdamW checkpoint {name} is invalid")
        betas = group.get("betas")
        if (
            not isinstance(betas, (list, tuple))
            or len(betas) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value < 1
                for value in betas
            )
        ):
            raise TransformerLabError("AdamW checkpoint betas are invalid")
        for name in ("amsgrad", "maximize", "capturable", "differentiable"):
            if not isinstance(group.get(name), bool):
                raise TransformerLabError(f"AdamW checkpoint {name} must be boolean")
        moments = ["exp_avg", "exp_avg_sq"]
        if group.get("amsgrad", False):
            moments.append("max_exp_avg_sq")
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if not state:
                continue  # AdamW allocates state lazily on the first update.
            if not {"step", *moments}.issubset(state):
                raise TransformerLabError("AdamW checkpoint is missing optimizer state")
            step = state["step"]
            if (
                not isinstance(step, Tensor)
                or step.ndim != 0
                or step.dtype == torch.bool
                or step.is_complex()
                or not math.isfinite(float(step))
                or float(step) < 0
                or not float(step).is_integer()
            ):
                raise TransformerLabError(
                    "AdamW checkpoint step must be a non-negative integer"
                )
            for name in moments:
                moment = state[name]
                if (
                    not isinstance(moment, Tensor)
                    or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype
                    or moment.device != parameter.device
                ):
                    raise TransformerLabError(
                        f"AdamW checkpoint {name} must match its parameter"
                    )
                if name != "exp_avg" and bool((moment < 0).any()):
                    raise TransformerLabError(
                        f"AdamW checkpoint {name} must be non-negative"
                    )


def model_fingerprint(model: DecoderLanguageModel) -> str:
    """Hash model state names, dtypes, shapes, and values."""

    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_training_checkpoint(
    path: Path,
    *,
    model: DecoderLanguageModel,
    optimizer: torch.optim.Optimizer,
    cursor: BatchCursor,
    training_config: TrainingConfig,
    corpus_fingerprint: str,
    step: int,
) -> None:
    """Atomically save and durably publish exact training restart state.

    A parent-directory fsync failure is reported after replacement because the
    new checkpoint is valid but its directory entry may not survive a crash.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise TransformerLabError("step must be a non-negative integer")
    _require_sha256_digest(corpus_fingerprint, "corpus_fingerprint")
    _validate_optimizer_binding(model, optimizer)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "corpus_fingerprint": corpus_fingerprint,
        "model_fingerprint": model_fingerprint(model),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cursor_state": cursor.state_dict(),
        "cursor_fingerprint": cursor.fingerprint(),
        "torch_rng_state": torch.get_rng_state(),
    }
    for name in ("model_state", "optimizer_state", "model_config", "training_config"):
        _require_finite_checkpoint_state(payload[name], name)
    _validate_adamw_checkpoint_state(optimizer)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_training_checkpoint(
    path: Path,
    *,
    model: DecoderLanguageModel,
    optimizer: torch.optim.Optimizer,
    cursor: BatchCursor,
    training_config: TrainingConfig,
    corpus_fingerprint: str,
    max_checkpoint_bytes: int = 64 * 1024 * 1024,
) -> int:
    """Validate and restore an exact transformer training checkpoint."""

    _validate_optimizer_binding(model, optimizer)
    _require_sha256_digest(corpus_fingerprint, "corpus_fingerprint")
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if type(max_checkpoint_bytes) is not int or max_checkpoint_bytes <= 0:
        raise TransformerLabError("checkpoint byte budget must be positive")
    try:
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size > max_checkpoint_bytes:
                raise TransformerLabError("checkpoint archive exceeds its byte budget")
            with zipfile.ZipFile(stream) as archive:
                members = archive.infolist()
                if sum(member.file_size for member in members) > max_checkpoint_bytes:
                    raise TransformerLabError(
                        "expanded checkpoint archive exceeds its byte budget"
                    )
                if any(
                    member.compress_type != zipfile.ZIP_STORED for member in members
                ):
                    raise TransformerLabError(
                        "checkpoint requires uncompressed torch.save ZIP records"
                    )
            stream.seek(0)
            payload = torch.load(stream, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
        IndexError,
        KeyError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        raise TransformerLabError(f"unable to read checkpoint: {path}") from exc
    if not isinstance(payload, Mapping) or set(payload) not in (
        _CHECKPOINT_FIELDS,
        _CHECKPOINT_FIELDS | {"cursor_fingerprint"},
    ):
        raise TransformerLabError("checkpoint fields are invalid")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in (1, CHECKPOINT_SCHEMA_VERSION)
    ):
        raise TransformerLabError("unsupported checkpoint schema")
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        if set(payload) != _CHECKPOINT_FIELDS | {"cursor_fingerprint"}:
            raise TransformerLabError("checkpoint fields are invalid")
        if payload["cursor_fingerprint"] != cursor.fingerprint():
            raise TransformerLabError("checkpoint cursor identity mismatch")
    elif set(payload) != _CHECKPOINT_FIELDS:
        raise TransformerLabError("legacy checkpoint fields are invalid")
    saved_config = payload.get("model_config")
    if not isinstance(saved_config, Mapping):
        raise TransformerLabError("checkpoint model configuration mismatch")
    normalized_config = dict(saved_config)
    for name, default in _LEGACY_TRANSFORMER_CONTROLS.items():
        normalized_config.setdefault(name, default)
    if normalized_config.keys() != asdict(model.config).keys():
        raise TransformerLabError("checkpoint model configuration mismatch")
    try:
        restored_config = TransformerConfig(**normalized_config)
    except (TypeError, ValueError) as exc:
        raise TransformerLabError("checkpoint model configuration is invalid") from exc
    if restored_config != model.config:
        raise TransformerLabError("checkpoint model configuration mismatch")
    saved_training_config = payload.get("training_config")
    if not isinstance(saved_training_config, Mapping):
        raise TransformerLabError("checkpoint training configuration mismatch")
    if saved_training_config.keys() != asdict(training_config).keys():
        raise TransformerLabError("checkpoint training configuration mismatch")
    try:
        restored_training_config = TrainingConfig(**dict(saved_training_config))
    except (TypeError, ValueError) as exc:
        raise TransformerLabError(
            "checkpoint training configuration is invalid"
        ) from exc
    if restored_training_config != training_config:
        raise TransformerLabError("checkpoint training configuration mismatch")
    saved_corpus_fingerprint = payload.get("corpus_fingerprint")
    _require_sha256_digest(saved_corpus_fingerprint, "checkpoint corpus_fingerprint")
    if saved_corpus_fingerprint != corpus_fingerprint:
        raise TransformerLabError("checkpoint corpus fingerprint mismatch")
    _require_sha256_digest(
        payload.get("model_fingerprint"), "checkpoint model_fingerprint"
    )
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise TransformerLabError("checkpoint step is invalid")
    for name in ("model_state", "optimizer_state"):
        _require_finite_checkpoint_state(payload[name], name)
    original_model = copy.deepcopy(model.state_dict())
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    original_cursor = cursor.state_dict()
    original_rng = torch.get_rng_state().clone()
    try:
        model.load_state_dict(payload["model_state"])
        if model_fingerprint(model) != payload.get("model_fingerprint"):
            raise TransformerLabError("checkpoint model fingerprint mismatch")
        optimizer.load_state_dict(payload["optimizer_state"])
        _require_finite_checkpoint_state(optimizer.state_dict(), "optimizer_state")
        _validate_adamw_checkpoint_state(optimizer)
        cursor.load_state_dict(payload["cursor_state"])
        torch.set_rng_state(payload["torch_rng_state"])
    except Exception:
        model.load_state_dict(original_model)
        optimizer.load_state_dict(original_optimizer)
        cursor.load_state_dict(original_cursor)
        torch.set_rng_state(original_rng)
        raise
    return step


@dataclass(frozen=True)
class ExperimentResult:
    model_config: TransformerConfig
    training_config: TrainingConfig
    corpus_fingerprint: str
    codec_fingerprint: str
    model_fingerprint: str
    parameter_count: int
    completed_steps: int
    initial_train_nll: float
    final_train_nll: float
    initial_validation_nll: float
    final_validation_nll: float
    trace: tuple[StepMetric, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trace"] = [asdict(metric) for metric in self.trace]
        return payload


def run_transformer_experiment(
    corpus: TokenCorpus,
    *,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    checkpoint_path: Path,
    resume: bool = False,
) -> ExperimentResult:
    """Train, evaluate, and checkpoint a deterministic transformer experiment."""

    if model_config.vocab_size != corpus.vocab_size:
        raise TransformerLabError("model vocabulary does not match corpus codec")
    seed_everything(training_config.seed)
    model = DecoderLanguageModel(model_config)
    optimizer = build_optimizer(model, training_config)
    cursor = BatchCursor(
        corpus.train_tokens,
        block_size=model_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    initial_train_nll = evaluate_nll(model, corpus.train_tokens)
    initial_validation_nll = evaluate_nll(model, corpus.validation_tokens)
    start_step = 0
    if resume:
        start_step = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            cursor=cursor,
            training_config=training_config,
            corpus_fingerprint=corpus.fingerprint(),
        )
    if start_step > training_config.steps:
        raise TransformerLabError("checkpoint is beyond configured training steps")
    remaining = training_config.steps - start_step
    trace = (
        train_steps(
            model,
            cursor,
            optimizer,
            training_config,
            start_step=start_step,
            step_count=remaining,
        )
        if remaining
        else ()
    )
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        cursor=cursor,
        training_config=training_config,
        corpus_fingerprint=corpus.fingerprint(),
        step=training_config.steps,
    )
    return ExperimentResult(
        model_config=model_config,
        training_config=training_config,
        corpus_fingerprint=corpus.fingerprint(),
        codec_fingerprint=corpus.codec.fingerprint(),
        model_fingerprint=model_fingerprint(model),
        parameter_count=model.parameter_count,
        completed_steps=training_config.steps,
        initial_train_nll=initial_train_nll,
        final_train_nll=evaluate_nll(model, corpus.train_tokens),
        initial_validation_nll=initial_validation_nll,
        final_validation_nll=evaluate_nll(model, corpus.validation_tokens),
        trace=trace,
    )


def write_experiment_report(path: Path, result: ExperimentResult) -> None:
    """Atomically write stable JSON evidence for an experiment."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(result, ExperimentResult):
        raise TypeError("result must be ExperimentResult")
    rendered = (
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class OverfitProbeResult:
    example_count: int
    steps: int
    initial_nll: float
    final_nll: float
    model_fingerprint: str


def run_overfit_probe(
    corpus: TokenCorpus,
    *,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    example_count: int = 100,
) -> OverfitProbeResult:
    """Deliberately fit an exact-size window subset as a capacity regression."""

    if isinstance(example_count, bool) or not isinstance(example_count, int):
        raise TypeError("example_count must be an integer")
    if example_count <= 0:
        raise TransformerLabError("example_count must be positive")
    combined = torch.cat((corpus.train_tokens, corpus.validation_tokens))
    required = example_count + model_config.block_size
    if len(combined) < required:
        raise TransformerLabError("corpus has too few windows for the overfit probe")
    subset = combined[:required]
    seed_everything(training_config.seed)
    model = DecoderLanguageModel(model_config)
    optimizer = build_optimizer(model, training_config)
    cursor = BatchCursor(
        subset,
        block_size=model_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    initial = evaluate_nll(model, subset)
    train_steps(model, cursor, optimizer, training_config)
    return OverfitProbeResult(
        example_count=example_count,
        steps=training_config.steps,
        initial_nll=initial,
        final_nll=evaluate_nll(model, subset),
        model_fingerprint=model_fingerprint(model),
    )
