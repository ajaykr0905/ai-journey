"""Deterministic, checkpointable decoder-only transformer training primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TransformerLabError(ValueError):
    """Raised when transformer data, configuration, or state is invalid."""


@dataclass(frozen=True)
class TransformerConfig:
    """Validated architecture and training-independent model settings."""

    vocab_size: int
    block_size: int = 16
    embedding_dim: int = 32
    head_count: int = 4
    layer_count: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "block_size",
            "embedding_dim",
            "head_count",
            "layer_count",
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

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


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


@dataclass(frozen=True)
class TokenCorpus:
    """Tokenized public text with immutable provenance and held-out split."""

    codec: CharacterCodec
    train_tokens: Tensor
    validation_tokens: Tensor
    source_sha256: str

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
            raise TransformerLabError("validation_fraction must be between zero and one")
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
        return len(self.codec.tokens)

    def fingerprint(self) -> str:
        digest = sha256()
        digest.update(self.source_sha256.encode())
        digest.update(self.codec.fingerprint().encode())
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
        x = torch.stack([self.tokens[start : start + self.block_size] for start in starts])
        y = torch.stack(
            [self.tokens[start + 1 : start + self.block_size + 1] for start in starts]
        )
        return x, y

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if set(state) != {"epoch", "offset"}:
            raise TransformerLabError("batch cursor state has invalid fields")
        epoch, offset = state["epoch"], state["offset"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in state.values()):
            raise TypeError("batch cursor state values must be integers")
        if epoch < 0 or not 0 <= offset <= self.sample_count:
            raise TransformerLabError("batch cursor state is out of range")
        self.epoch = epoch
        self.offset = offset


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit causal mask."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.head_count = config.head_count
        self.head_dim = config.head_dim
        self.query_key_value = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, time, channels = inputs.shape
        if time > self.causal_mask.shape[0]:
            raise TransformerLabError("sequence exceeds configured block_size")
        qkv = self.query_key_value(inputs)
        query, key, value = qkv.chunk(3, dim=-1)
        shape = (batch, time, self.head_count, self.head_dim)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) * self.head_dim**-0.5
        mask = self.causal_mask[:time, :time]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.residual_dropout(self.projection(attended))


class FeedForward(nn.Module):
    """Transformer position-wise MLP with a four-times expansion."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.embedding_dim
        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)
