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


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise TransformerLabError(f"{name} must be positive")
        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise TransformerLabError("weight_decay must be non-negative")


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


class TransformerBlock(nn.Module):
    """Pre-normalized attention and feed-forward residual block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.feed_forward(self.feed_forward_norm(inputs))


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
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, token_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise TypeError("token_ids must be a two-dimensional torch.long tensor")
        _, time = token_ids.shape
        if time > self.config.block_size:
            raise TransformerLabError("sequence exceeds configured block_size")
        if token_ids.numel() and (
            int(token_ids.min()) < 0 or int(token_ids.max()) >= self.config.vocab_size
        ):
            raise TransformerLabError("token id is outside the vocabulary")
        positions = torch.arange(time, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape or targets.dtype != torch.long:
                raise TypeError("targets must match token_ids shape and dtype")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
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
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise TransformerLabError("temperature must be positive")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
        ):
            raise TransformerLabError("top_k must be a positive integer")
        was_training = self.training
        self.eval()
        generated = token_ids.clone()
        for _ in range(new_tokens):
            context = generated[:, -self.config.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1] / temperature
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                next_logits[next_logits < values[:, [-1]]] = float("-inf")
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            )
            generated = torch.cat((generated, next_token), dim=1)
        self.train(was_training)
        return generated


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
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise TransformerLabError("batch_size must be a positive integer")
    was_training = model.training
    model.eval()
    losses: list[Tensor] = []
    for start in range(0, len(tokens) - model.config.block_size, batch_size):
        indexes = range(
            start,
            min(start + batch_size, len(tokens) - model.config.block_size),
        )
        x = torch.stack([tokens[i : i + model.config.block_size] for i in indexes])
        y = torch.stack([tokens[i + 1 : i + model.config.block_size + 1] for i in indexes])
        logits, _ = model(x)
        per_token = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            y.reshape(-1),
            reduction="none",
        )
        losses.append(per_token)
    model.train(was_training)
    return float(torch.cat(losses).mean())


@dataclass(frozen=True)
class StepMetric:
    step: int
    loss: float
    gradient_norm: float


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

    count = config.steps if step_count is None else step_count
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise TransformerLabError("step_count must be a positive integer")
    if isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0:
        raise TransformerLabError("start_step must be a non-negative integer")
    model.train()
    metrics: list[StepMetric] = []
    for step in range(start_step, start_step + count):
        x, y = cursor.next()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("training loss was not computed")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.gradient_clip
        )
        optimizer.step()
        metrics.append(
            StepMetric(
                step=step + 1,
                loss=float(loss.detach()),
                gradient_norm=float(gradient_norm),
            )
        )
    return tuple(metrics)
