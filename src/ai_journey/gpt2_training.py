"""Deterministic GPT-2 training controls with exact restart evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .gpt2_model import GPT2LanguageModel


class GPT2TrainingError(ValueError):
    """Raised when a GPT-2 training contract or run is invalid."""


@dataclass(frozen=True)
class GPT2OptimizerGroups:
    """Inspectable AdamW grouping without duplicating tied parameters."""

    decay_names: tuple[str, ...]
    no_decay_names: tuple[str, ...]
    decay_parameters: int
    no_decay_parameters: int

    @property
    def total_parameters(self) -> int:
        return self.decay_parameters + self.no_decay_parameters


@dataclass(frozen=True)
class GPT2StepMetric:
    """Inspectable outcome of one accumulated optimizer update."""

    step: int
    mean_loss: float
    learning_rate: float
    gradient_norm_before_clip: float
    gradient_norm_after_clip: float
    clipped: bool
    micro_batches: int
    target_tokens: int


@dataclass(frozen=True)
class GPT2ByteCorpus:
    """Immutable byte-token corpus with exact source provenance."""

    tokens: Tensor
    source_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tokens, Tensor)
            or self.tokens.layout != torch.strided
            or self.tokens.device.type != "cpu"
            or self.tokens.dtype != torch.long
            or self.tokens.ndim != 1
        ):
            raise TypeError("tokens must be a dense one-dimensional CPU long tensor")
        if self.tokens.numel() < 2:
            raise GPT2TrainingError("byte corpus requires at least two tokens")
        if bool(((self.tokens < 0) | (self.tokens > 255)).any()):
            raise GPT2TrainingError("byte corpus token ids must be in [0, 255]")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.source_sha256
            )
        ):
            raise GPT2TrainingError("source_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "tokens", self.tokens.detach().clone())

    @classmethod
    def from_bytes(
        cls, source: bytes, *, max_bytes: int = 16 * 1024 * 1024
    ) -> "GPT2ByteCorpus":
        """Build a byte-token corpus under an explicit input-size budget."""

        if not isinstance(source, bytes):
            raise TypeError("source must be bytes")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("max_bytes must be an integer")
        if max_bytes <= 0:
            raise GPT2TrainingError("max_bytes must be positive")
        if len(source) < 2:
            raise GPT2TrainingError("source must contain at least two bytes")
        if len(source) > max_bytes:
            raise GPT2TrainingError("source exceeds max_bytes")
        return cls(
            tokens=torch.tensor(tuple(source), dtype=torch.long),
            source_sha256=sha256(source).hexdigest(),
        )

    def fingerprint(self) -> str:
        """Bind the declared source digest to the materialized token sequence."""

        digest = sha256()
        digest.update(self.source_sha256.encode())
        digest.update(self.tokens.numpy().tobytes())
        return digest.hexdigest()


class GPT2BatchCursor:
    """Deterministic full micro-batches with resumable shuffled epochs."""

    def __init__(self, corpus: GPT2ByteCorpus, config: "GPT2TrainingConfig") -> None:
        if not isinstance(corpus, GPT2ByteCorpus):
            raise TypeError("corpus must be GPT2ByteCorpus")
        if not isinstance(config, GPT2TrainingConfig):
            raise TypeError("config must be GPT2TrainingConfig")
        sample_count = len(corpus.tokens) - config.sequence_length
        if sample_count < config.micro_batch_size:
            raise GPT2TrainingError(
                "corpus must provide at least one complete micro-batch"
            )
        self.tokens = corpus.tokens.clone()
        self.sequence_length = config.sequence_length
        self.batch_size = config.micro_batch_size
        self.seed = config.seed
        self.sample_count = sample_count
        self.usable_samples = sample_count - sample_count % self.batch_size
        self.epoch = 0
        self.offset = 0

    @property
    def discarded_samples_per_epoch(self) -> int:
        """Return tail windows omitted to preserve exact micro-batch weight."""

        return self.sample_count - self.usable_samples

    def _order(self) -> Tensor:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        return torch.randperm(self.sample_count, generator=generator)

    def next(self) -> tuple[Tensor, Tensor]:
        """Return the next complete input/target micro-batch."""

        if self.offset >= self.usable_samples:
            self.epoch += 1
            self.offset = 0
        starts = self._order()[self.offset : self.offset + self.batch_size]
        self.offset += self.batch_size
        inputs = torch.stack(
            [
                self.tokens[start : start + self.sequence_length]
                for start in starts.tolist()
            ]
        )
        targets = torch.stack(
            [
                self.tokens[start + 1 : start + self.sequence_length + 1]
                for start in starts.tolist()
            ]
        )
        return inputs, targets

    def state_dict(self) -> dict[str, int]:
        """Return the minimum exact restart position."""

        return {"epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore a validated micro-batch boundary."""

        if not isinstance(state, dict) or set(state) != {"epoch", "offset"}:
            raise GPT2TrainingError("batch cursor state has invalid fields")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in state.values()
        ):
            raise TypeError("batch cursor state values must be integers")
        epoch, offset = state["epoch"], state["offset"]
        if epoch < 0 or not 0 <= offset <= self.usable_samples:
            raise GPT2TrainingError("batch cursor state is out of range")
        if offset % self.batch_size:
            raise GPT2TrainingError("batch cursor offset is not a batch boundary")
        if self.seed + epoch >= 2**63:
            raise GPT2TrainingError("batch cursor epoch exceeds its RNG range")
        self.epoch = epoch
        self.offset = offset

    def fingerprint(self) -> str:
        """Bind restart state compatibility to data and sampler controls."""

        digest = sha256()
        digest.update(self.tokens.numpy().tobytes())
        digest.update(
            json.dumps(
                {
                    "sequence_length": self.sequence_length,
                    "batch_size": self.batch_size,
                    "seed": self.seed,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        return digest.hexdigest()


@dataclass(frozen=True)
class GPT2TrainingConfig:
    """Validated optimizer, accumulation, schedule, and reproducibility controls."""

    optimizer_steps: int
    micro_batch_size: int
    accumulation_steps: int
    sequence_length: int
    max_learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    seed: int

    def __post_init__(self) -> None:
        for name in (
            "optimizer_steps",
            "micro_batch_size",
            "accumulation_steps",
            "sequence_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise GPT2TrainingError(f"{name} must be positive")
        if isinstance(self.warmup_steps, bool) or not isinstance(
            self.warmup_steps, int
        ):
            raise TypeError("warmup_steps must be an integer")
        if not 0 <= self.warmup_steps <= self.optimizer_steps:
            raise GPT2TrainingError(
                "warmup_steps must be between zero and optimizer_steps"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed < 2**32:
            raise GPT2TrainingError("seed must be in [0, 2**32)")
        for name in (
            "max_learning_rate",
            "min_learning_rate",
            "weight_decay",
            "gradient_clip",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise GPT2TrainingError(f"{name} must be finite")
        if self.max_learning_rate <= 0:
            raise GPT2TrainingError("max_learning_rate must be positive")
        if not 0 <= self.min_learning_rate <= self.max_learning_rate:
            raise GPT2TrainingError(
                "min_learning_rate must be between zero and max_learning_rate"
            )
        if self.weight_decay < 0:
            raise GPT2TrainingError("weight_decay must be non-negative")
        if self.gradient_clip <= 0:
            raise GPT2TrainingError("gradient_clip must be positive")

    @property
    def effective_batch_size(self) -> int:
        """Return examples contributing to each optimizer update."""

        return self.micro_batch_size * self.accumulation_steps

    @property
    def tokens_per_step(self) -> int:
        """Return target-token exposure for one optimizer update."""

        return self.effective_batch_size * self.sequence_length

    @property
    def total_target_tokens(self) -> int:
        """Return target-token exposure for the complete schedule."""

        return self.tokens_per_step * self.optimizer_steps

    def fingerprint(self) -> str:
        """Return a canonical digest of every training control."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


def learning_rate_at(step: int, config: GPT2TrainingConfig) -> float:
    """Return linear warmup followed by cosine decay to the declared floor."""

    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer")
    if not isinstance(config, GPT2TrainingConfig):
        raise TypeError("config must be GPT2TrainingConfig")
    if not 0 <= step < config.optimizer_steps:
        raise GPT2TrainingError("step must be in the configured optimizer schedule")
    if step < config.warmup_steps:
        return config.max_learning_rate * (step + 1) / config.warmup_steps
    decay_steps = config.optimizer_steps - config.warmup_steps
    if decay_steps == 1:
        return config.min_learning_rate
    progress = (step - config.warmup_steps) / (decay_steps - 1)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + coefficient * (
        config.max_learning_rate - config.min_learning_rate
    )


def optimizer_group_summary(model: GPT2LanguageModel) -> GPT2OptimizerGroups:
    """Classify matrix parameters for decay and vectors for no decay."""

    if not isinstance(model, GPT2LanguageModel):
        raise TypeError("model must be GPT2LanguageModel")
    named = tuple(model.named_parameters())
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise RuntimeError("named parameters contain duplicate tensor identities")
    decay = tuple((name, parameter) for name, parameter in named if parameter.ndim >= 2)
    no_decay = tuple(
        (name, parameter) for name, parameter in named if parameter.ndim < 2
    )
    if len(decay) + len(no_decay) != len(named):
        raise RuntimeError("optimizer grouping is incomplete")
    return GPT2OptimizerGroups(
        decay_names=tuple(name for name, _ in decay),
        no_decay_names=tuple(name for name, _ in no_decay),
        decay_parameters=sum(parameter.numel() for _, parameter in decay),
        no_decay_parameters=sum(parameter.numel() for _, parameter in no_decay),
    )


def build_gpt2_optimizer(
    model: GPT2LanguageModel, config: GPT2TrainingConfig
) -> torch.optim.AdamW:
    """Build AdamW with weight decay only on matrix-shaped parameters."""

    if not isinstance(config, GPT2TrainingConfig):
        raise TypeError("config must be GPT2TrainingConfig")
    summary = optimizer_group_summary(model)
    parameters = dict(model.named_parameters())
    decay = [parameters[name] for name in summary.decay_names]
    no_decay = [parameters[name] for name in summary.no_decay_names]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.max_learning_rate,
    )


def language_model_loss(
    model: GPT2LanguageModel, inputs: Tensor, targets: Tensor
) -> Tensor:
    """Calculate finite mean next-token loss for one validated micro-batch."""

    if not isinstance(model, GPT2LanguageModel):
        raise TypeError("model must be GPT2LanguageModel")
    for name, tensor in (("inputs", inputs), ("targets", targets)):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.long or tensor.ndim != 2:
            raise TypeError(f"{name} must be a rank-two torch.long tensor")
        if tensor.device != model.wte.weight.device:
            raise GPT2TrainingError(f"{name} must match the model device")
    if inputs.shape != targets.shape or inputs.numel() == 0:
        raise GPT2TrainingError("inputs and targets must have the same non-empty shape")
    if inputs.shape[1] > model.config.context_length:
        raise GPT2TrainingError("training sequence exceeds model context length")
    for name, tensor in (("inputs", inputs), ("targets", targets)):
        if bool(((tensor < 0) | (tensor >= model.config.vocab_size)).any()):
            raise GPT2TrainingError(f"{name} contain an out-of-range token id")
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    if not bool(torch.isfinite(loss)):
        raise GPT2TrainingError("training loss must be finite")
    return loss


def _validate_optimizer_binding(
    model: GPT2LanguageModel, optimizer: torch.optim.Optimizer
) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer")
    actual = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    expected = {id(parameter) for parameter in model.parameters()}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise GPT2TrainingError("optimizer parameters must match the complete model")


def _gradient_norm(parameters: tuple[Tensor, ...]) -> float:
    total = math.fsum(
        float(parameter.grad.detach().double().pow(2).sum()) for parameter in parameters
    )
    return math.sqrt(total)


def train_optimizer_step(
    model: GPT2LanguageModel,
    cursor: GPT2BatchCursor,
    optimizer: torch.optim.Optimizer,
    config: GPT2TrainingConfig,
    *,
    step: int,
) -> GPT2StepMetric:
    """Accumulate exact mean gradients and apply one scheduled clipped update."""

    if not isinstance(model, GPT2LanguageModel):
        raise TypeError("model must be GPT2LanguageModel")
    if not isinstance(cursor, GPT2BatchCursor):
        raise TypeError("cursor must be GPT2BatchCursor")
    if not isinstance(config, GPT2TrainingConfig):
        raise TypeError("config must be GPT2TrainingConfig")
    if cursor.sequence_length > model.config.context_length:
        raise GPT2TrainingError("cursor sequence exceeds model context length")
    _validate_optimizer_binding(model, optimizer)
    learning_rate = learning_rate_at(step, config)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for _ in range(config.accumulation_steps):
        inputs, targets = cursor.next()
        loss = language_model_loss(model, inputs, targets)
        losses.append(float(loss.detach()))
        (loss / config.accumulation_steps).backward()
    parameters = tuple(model.parameters())
    if any(
        parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    ):
        raise GPT2TrainingError("all GPT-2 gradients must be present and finite")
    norm_before = float(
        torch.nn.utils.clip_grad_norm_(
            parameters, config.gradient_clip, error_if_nonfinite=True
        )
    )
    norm_after = _gradient_norm(parameters)
    optimizer.step()
    return GPT2StepMetric(
        step=step + 1,
        mean_loss=math.fsum(losses) / len(losses),
        learning_rate=learning_rate,
        gradient_norm_before_clip=norm_before,
        gradient_norm_after_clip=norm_after,
        clipped=norm_before > config.gradient_clip,
        micro_batches=config.accumulation_steps,
        target_tokens=config.tokens_per_step,
    )
