"""Deterministic GPT-2 training controls with exact restart evidence."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any
import zipfile

import torch
from torch import Tensor
from torch.nn import functional as F

from .gpt2_config import GPT2Config
from .gpt2_model import GPT2LanguageModel
from .decoder_variant_config import DecoderVariantConfig
from .moe_config import MoEConfig
from .moe_model import MoEDecoderModel
from .decoder_variant_model import DecoderVariantModel
from .training_models import (
    TrainingModel,
    TrainingModelConfig,
    base_model_config,
    build_training_model,
    expected_training_parameters,
    model_config_from_mapping,
    model_config_mapping,
    training_model_fingerprint,
)
from .training_optimizer_state import validate_adamw_state
from .checkpoint_archive import load_checkpoint_archive


class GPT2TrainingError(ValueError):
    """Raised when a GPT-2 training contract or run is invalid."""


GPT2_TRAINING_CHECKPOINT_SCHEMA_VERSION = 1
GPT2_TRAINING_SPEC_SCHEMA_VERSION = 1
GPT2_TRAINING_TORCH_THREADS = 1
_GPT2_TRAINING_CHECKPOINT_FIELDS = {
    "schema_version",
    "step",
    "model_config",
    "training_config",
    "training_fingerprint",
    "corpus_fingerprint",
    "cursor_fingerprint",
    "model_fingerprint",
    "model_state",
    "optimizer_state",
    "cursor_state",
    "torch_rng_state",
}


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
class GPT2TrainingSource:
    """Public source identity required for a reproducible training run."""

    name: str
    url: str
    revision: str
    license: str
    sha256: str

    def __post_init__(self) -> None:
        for name in ("name", "url", "revision", "license"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise GPT2TrainingError(f"source {name} must be a non-empty string")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise GPT2TrainingError("source sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class GPT2TrainingSpec:
    """Complete model, optimizer, and public-source contract for one run."""

    model: TrainingModelConfig
    training: GPT2TrainingConfig
    source: GPT2TrainingSource

    def __post_init__(self) -> None:
        if not isinstance(self.model, (GPT2Config, DecoderVariantConfig)):
            raise TypeError("model must be GPT2Config or DecoderVariantConfig")
        if isinstance(self.model, MoEConfig):
            raise GPT2TrainingError(
                "MoE training requires its explicit auxiliary objective"
            )
        if not isinstance(self.training, GPT2TrainingConfig):
            raise TypeError("training must be GPT2TrainingConfig")
        if not isinstance(self.source, GPT2TrainingSource):
            raise TypeError("source must be GPT2TrainingSource")
        base = base_model_config(self.model)
        if base.vocab_size != 256:
            raise GPT2TrainingError("byte-token training requires vocab_size 256")
        if self.training.sequence_length > base.context_length:
            raise GPT2TrainingError("sequence_length exceeds model context_length")
        if any(
            value != 0
            for value in (
                base.embedding_dropout,
                base.attention_dropout,
                base.residual_dropout,
            )
        ):
            raise GPT2TrainingError(
                "exact replay requires all model dropout to be zero"
            )

    def fingerprint(self) -> str:
        return sha256(
            json.dumps(
                self.to_mapping(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def to_mapping(self) -> dict:
        """Serialize legacy baseline specs unchanged and tag modern specs as v2."""
        return {
            "schema_version": 2
            if isinstance(self.model, DecoderVariantConfig)
            else GPT2_TRAINING_SPEC_SCHEMA_VERSION,
            "model": model_config_mapping(self.model),
            "training": asdict(self.training),
            "source": asdict(self.source),
        }


@dataclass(frozen=True)
class GPT2TrainingRun:
    """Measured state transition for one fresh or resumed training segment."""

    spec: GPT2TrainingSpec
    corpus_fingerprint: str
    corpus_bytes: int
    discarded_samples_per_epoch: int
    optimizer_groups: GPT2OptimizerGroups
    parameter_count: int
    torch_threads: int
    started_step: int
    completed_step: int
    initial_model_fingerprint: str
    final_model_fingerprint: str
    metrics: tuple[GPT2StepMetric, ...]


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


def optimizer_group_summary(model: TrainingModel) -> GPT2OptimizerGroups:
    """Classify matrix parameters for decay and vectors for no decay."""

    if not isinstance(model, (GPT2LanguageModel, DecoderVariantModel)):
        raise TypeError("model must be GPT2LanguageModel or DecoderVariantModel")
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
    model: TrainingModel, config: GPT2TrainingConfig
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
        capturable=False,
        differentiable=False,
        foreach=None,
        fused=None,
    )


def language_model_loss(
    model: TrainingModel, inputs: Tensor, targets: Tensor
) -> Tensor:
    """Calculate finite mean next-token loss for one validated micro-batch."""

    if isinstance(model, MoEDecoderModel):
        raise GPT2TrainingError(
            "MoE training requires its explicit auxiliary objective"
        )
    if not isinstance(model, (GPT2LanguageModel, DecoderVariantModel)):
        raise TypeError("model must be GPT2LanguageModel or DecoderVariantModel")
    base = base_model_config(model)
    for name, tensor in (("inputs", inputs), ("targets", targets)):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.long or tensor.ndim != 2:
            raise TypeError(f"{name} must be a rank-two torch.long tensor")
        if tensor.device != model.wte.weight.device:
            raise GPT2TrainingError(f"{name} must match the model device")
    if inputs.shape != targets.shape or inputs.numel() == 0:
        raise GPT2TrainingError("inputs and targets must have the same non-empty shape")
    if inputs.shape[1] > base.context_length:
        raise GPT2TrainingError("training sequence exceeds model context length")
    for name, tensor in (("inputs", inputs), ("targets", targets)):
        if bool(((tensor < 0) | (tensor >= base.vocab_size)).any()):
            raise GPT2TrainingError(f"{name} contain an out-of-range token id")
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    if not bool(torch.isfinite(loss)):
        raise GPT2TrainingError("training loss must be finite")
    return loss


def _validate_optimizer_binding(
    model: TrainingModel, optimizer: torch.optim.Optimizer
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


def _validate_optimizer_contract(
    model: TrainingModel, optimizer, config: GPT2TrainingConfig, *, step: int
) -> None:
    """Bind restored AdamW controls and group order to the declared training run."""
    validate_adamw_state(model, optimizer, step=step)
    summary = optimizer_group_summary(model)
    named = dict(model.named_parameters())
    groups = optimizer.param_groups
    expected_names = (summary.decay_names, summary.no_decay_names)
    if len(groups) != 2:
        raise GPT2TrainingError("AdamW requires the declared decay and no-decay groups")
    expected_lr = (
        learning_rate_at(step - 1, config) if step else config.max_learning_rate
    )
    for group, names, decay in zip(
        groups, expected_names, (config.weight_decay, 0.0), strict=True
    ):
        if [id(parameter) for parameter in group["params"]] != [
            id(named[name]) for name in names
        ]:
            raise GPT2TrainingError(
                "AdamW parameter group order does not match the model"
            )
        if (
            group["lr"] != expected_lr
            or group["weight_decay"] != decay
            or tuple(group["betas"]) != (0.9, 0.999)
            or group["eps"] != 1e-8
            or group["amsgrad"] is not False
            or group.get("maximize", False) is not False
        ):
            raise GPT2TrainingError("AdamW controls do not match the training contract")


def _validate_cursor_progress(
    cursor: GPT2BatchCursor, config: GPT2TrainingConfig, *, step: int
) -> None:
    """Require the durable sampler boundary implied by completed optimizer steps."""
    if (
        cursor.sequence_length != config.sequence_length
        or cursor.batch_size != config.micro_batch_size
        or cursor.seed != config.seed
    ):
        raise GPT2TrainingError("checkpoint cursor controls do not match training")
    consumed = step * config.accumulation_steps
    batches_per_epoch = cursor.usable_samples // cursor.batch_size
    if consumed:
        epoch, batch_offset = divmod(consumed - 1, batches_per_epoch)
        expected = {"epoch": epoch, "offset": (batch_offset + 1) * cursor.batch_size}
    else:
        expected = {"epoch": 0, "offset": 0}
    if cursor.state_dict() != expected:
        raise GPT2TrainingError(
            "checkpoint cursor progress does not match its optimizer step"
        )


def _gradient_norm(parameters: tuple[Tensor, ...]) -> float:
    total = math.fsum(
        float(parameter.grad.detach().double().pow(2).sum()) for parameter in parameters
    )
    return math.sqrt(total)


def _require_finite_state(value: Any, name: str) -> None:
    if isinstance(value, Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise GPT2TrainingError(f"{name} must be finite")
    elif isinstance(value, float) and not math.isfinite(value):
        raise GPT2TrainingError(f"{name} must be finite")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_state(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_state(item, f"{name}[{index}]")


def train_optimizer_step(
    model: TrainingModel,
    cursor: GPT2BatchCursor,
    optimizer: torch.optim.Optimizer,
    config: GPT2TrainingConfig,
    *,
    step: int,
) -> GPT2StepMetric:
    """Accumulate exact mean gradients and apply one scheduled clipped update."""

    if not isinstance(model, (GPT2LanguageModel, DecoderVariantModel)):
        raise TypeError("model must be GPT2LanguageModel or DecoderVariantModel")
    if not isinstance(cursor, GPT2BatchCursor):
        raise TypeError("cursor must be GPT2BatchCursor")
    if not isinstance(config, GPT2TrainingConfig):
        raise TypeError("config must be GPT2TrainingConfig")
    if cursor.sequence_length > base_model_config(model).context_length:
        raise GPT2TrainingError("cursor sequence exceeds model context length")
    _validate_optimizer_binding(model, optimizer)
    parameters = tuple(model.parameters())
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_cursor = cursor.state_dict()
    saved_rng = torch.get_rng_state().clone()
    saved_gradients = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]
    caller_modes = [(module, module.training) for module in model.modules()]
    try:
        model.train()
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
        _require_finite_state(model.state_dict(), "updated model")
        _require_finite_state(optimizer.state_dict(), "updated optimizer")
        metric = GPT2StepMetric(
            step=step + 1,
            mean_loss=math.fsum(losses) / len(losses),
            learning_rate=learning_rate,
            gradient_norm_before_clip=norm_before,
            gradient_norm_after_clip=norm_after,
            clipped=norm_before > config.gradient_clip,
            micro_batches=config.accumulation_steps,
            target_tokens=config.tokens_per_step,
        )
        optimizer.zero_grad(set_to_none=True)
    except Exception:
        model.load_state_dict(saved_model)
        optimizer.load_state_dict(saved_optimizer)
        cursor.load_state_dict(saved_cursor)
        torch.set_rng_state(saved_rng)
        for parameter, gradient in zip(parameters, saved_gradients, strict=True):
            parameter.grad = gradient
        raise
    finally:
        for module, mode in caller_modes:
            module.training = mode
    return metric


def gpt2_model_fingerprint(model: TrainingModel) -> str:
    """Hash decoder state, preserving the published baseline fingerprint contract."""
    return training_model_fingerprint(model)


def train_optimizer_steps(
    model: TrainingModel,
    cursor: GPT2BatchCursor,
    optimizer: torch.optim.Optimizer,
    config: GPT2TrainingConfig,
    *,
    start_step: int = 0,
    step_count: int | None = None,
) -> tuple[GPT2StepMetric, ...]:
    """Run a bounded contiguous segment of the configured optimizer schedule."""

    if isinstance(start_step, bool) or not isinstance(start_step, int):
        raise TypeError("start_step must be an integer")
    if not 0 <= start_step < config.optimizer_steps:
        raise GPT2TrainingError("start_step must be inside the optimizer schedule")
    count = config.optimizer_steps - start_step if step_count is None else step_count
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("step_count must be an integer")
    if count <= 0 or start_step + count > config.optimizer_steps:
        raise GPT2TrainingError("step_count must fit the remaining optimizer schedule")
    return tuple(
        train_optimizer_step(
            model,
            cursor,
            optimizer,
            config,
            step=step,
        )
        for step in range(start_step, start_step + count)
    )


def save_training_checkpoint(
    path: Path,
    *,
    model: TrainingModel,
    optimizer: torch.optim.Optimizer,
    cursor: GPT2BatchCursor,
    config: GPT2TrainingConfig,
    corpus_fingerprint: str,
    step: int,
) -> None:
    """Atomically and durably save exact GPT-2 training restart state."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer")
    if not 0 <= step <= config.optimizer_steps:
        raise GPT2TrainingError("checkpoint step must fit the optimizer schedule")
    if (
        not isinstance(corpus_fingerprint, str)
        or len(corpus_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in corpus_fingerprint)
    ):
        raise GPT2TrainingError("corpus_fingerprint must be a SHA-256 digest")
    _validate_optimizer_binding(model, optimizer)
    _validate_optimizer_contract(model, optimizer, config, step=step)
    _validate_cursor_progress(cursor, config, step=step)
    payload = {
        "schema_version": GPT2_TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "model_config": model_config_mapping(model.config),
        "training_config": asdict(config),
        "training_fingerprint": config.fingerprint(),
        "corpus_fingerprint": corpus_fingerprint,
        "cursor_fingerprint": cursor.fingerprint(),
        "model_fingerprint": gpt2_model_fingerprint(model),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cursor_state": cursor.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    _require_finite_state(payload, "checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
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
    model: TrainingModel,
    optimizer: torch.optim.Optimizer,
    cursor: GPT2BatchCursor,
    config: GPT2TrainingConfig,
    corpus_fingerprint: str,
    max_checkpoint_bytes: int = 64 * 1024 * 1024,
) -> int:
    """Validate and transactionally restore an exact GPT-2 training checkpoint."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(max_checkpoint_bytes, bool) or not isinstance(
        max_checkpoint_bytes, int
    ):
        raise TypeError("max_checkpoint_bytes must be an integer")
    if max_checkpoint_bytes <= 0:
        raise GPT2TrainingError("max_checkpoint_bytes must be positive")
    if path.is_symlink() or not path.is_file():
        raise GPT2TrainingError("checkpoint must be a regular non-symlink file")
    if path.stat().st_size <= 0 or path.stat().st_size > max_checkpoint_bytes:
        raise GPT2TrainingError("checkpoint exceeds its byte budget")
    _validate_optimizer_binding(model, optimizer)
    try:
        payload = load_checkpoint_archive(path, max_bytes=max_checkpoint_bytes)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
        pickle.UnpicklingError,
        EOFError,
    ) as exc:
        raise GPT2TrainingError("checkpoint is not a weights-only archive") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != _GPT2_TRAINING_CHECKPOINT_FIELDS
    ):
        raise GPT2TrainingError("checkpoint fields do not match the schema")
    if payload["schema_version"] != GPT2_TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise GPT2TrainingError("checkpoint schema version is unsupported")
    if payload["model_config"] != model_config_mapping(model.config):
        raise GPT2TrainingError("checkpoint model configuration does not match")
    if (
        payload["training_config"] != asdict(config)
        or payload["training_fingerprint"] != config.fingerprint()
    ):
        raise GPT2TrainingError("checkpoint training configuration does not match")
    if payload["corpus_fingerprint"] != corpus_fingerprint:
        raise GPT2TrainingError("checkpoint corpus fingerprint does not match")
    if payload["cursor_fingerprint"] != cursor.fingerprint():
        raise GPT2TrainingError("checkpoint cursor fingerprint does not match")
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("checkpoint step must be an integer")
    if not 0 <= step <= config.optimizer_steps:
        raise GPT2TrainingError("checkpoint step is outside the optimizer schedule")
    rng_state = payload["torch_rng_state"]
    if (
        not isinstance(rng_state, Tensor)
        or rng_state.device.type != "cpu"
        or rng_state.dtype != torch.uint8
        or rng_state.ndim != 1
    ):
        raise GPT2TrainingError("checkpoint RNG state is invalid")
    _require_finite_state(payload, "checkpoint")
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_cursor = cursor.state_dict()
    saved_rng = torch.get_rng_state().clone()
    try:
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        cursor.load_state_dict(payload["cursor_state"])
        _validate_cursor_progress(cursor, config, step=step)
        torch.set_rng_state(rng_state)
        _validate_optimizer_binding(model, optimizer)
        _validate_optimizer_contract(model, optimizer, config, step=step)
        _require_finite_state(optimizer.state_dict(), "restored optimizer")
        if gpt2_model_fingerprint(model) != payload["model_fingerprint"]:
            raise GPT2TrainingError("checkpoint model fingerprint does not match")
    except Exception:
        model.load_state_dict(saved_model)
        optimizer.load_state_dict(saved_optimizer)
        cursor.load_state_dict(saved_cursor)
        torch.set_rng_state(saved_rng)
        raise
    return step


def training_spec_from_mapping(payload: Mapping[str, Any]) -> GPT2TrainingSpec:
    """Parse a complete spec without ignoring unknown or misspelled controls."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "model",
        "training",
        "source",
    }:
        raise GPT2TrainingError("training spec fields do not match the schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] not in (
        1,
        2,
    ):
        raise GPT2TrainingError("training spec schema version is unsupported")
    model = model_config_from_mapping(payload["model"])
    if (payload["schema_version"] == 2) != isinstance(model, DecoderVariantConfig):
        raise GPT2TrainingError("training spec schema does not match its architecture")
    components = {
        "training": (payload["training"], GPT2TrainingConfig),
        "source": (payload["source"], GPT2TrainingSource),
    }
    parsed: dict[str, Any] = {}
    for name, (value, kind) in components.items():
        expected = set(kind.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != expected:
            raise GPT2TrainingError(f"training spec {name} fields are invalid")
        parsed[name] = kind(**value)
    return GPT2TrainingSpec(model=model, **parsed)


def load_training_spec(path: Path, *, max_bytes: int = 1024 * 1024) -> GPT2TrainingSpec:
    """Load a bounded regular JSON training specification."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes <= 0:
        raise GPT2TrainingError("max_bytes must be positive")
    if path.is_symlink() or not path.is_file():
        raise GPT2TrainingError("training spec must be a regular non-symlink file")
    if path.stat().st_size <= 0 or path.stat().st_size > max_bytes:
        raise GPT2TrainingError("training spec exceeds its byte budget")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GPT2TrainingError("training spec must be valid UTF-8 JSON") from exc
    return training_spec_from_mapping(payload)


def run_training_experiment(
    source: bytes,
    spec: GPT2TrainingSpec,
    *,
    checkpoint_path: Path,
    resume: bool = False,
    max_parameters: int = 5_000_000,
    step_count: int | None = None,
) -> GPT2TrainingRun:
    """Run or resume a bounded deterministic byte-level GPT-2 experiment."""

    if not isinstance(spec, GPT2TrainingSpec):
        raise TypeError("spec must be GPT2TrainingSpec")
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be pathlib.Path")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a boolean")
    if step_count is not None and (type(step_count) is not int or step_count <= 0):
        raise GPT2TrainingError("step_count must be a positive integer")
    if isinstance(max_parameters, bool) or not isinstance(max_parameters, int):
        raise TypeError("max_parameters must be an integer")
    if max_parameters <= 0:
        raise GPT2TrainingError("max_parameters must be positive")
    parameter_count = expected_training_parameters(spec.model)
    if parameter_count > max_parameters:
        raise GPT2TrainingError("model exceeds the declared parameter budget")
    corpus = GPT2ByteCorpus.from_bytes(source)
    if corpus.source_sha256 != spec.source.sha256:
        raise GPT2TrainingError("source bytes do not match the training spec")
    caller_rng = torch.get_rng_state().clone()
    caller_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(GPT2_TRAINING_TORCH_THREADS)
        torch.manual_seed(spec.training.seed)
        model = build_training_model(spec.model, max_parameters=max_parameters)
        cursor = GPT2BatchCursor(corpus, spec.training)
        optimizer = build_gpt2_optimizer(model, spec.training)
        started_step = 0
        if resume:
            started_step = load_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                config=spec.training,
                corpus_fingerprint=corpus.fingerprint(),
            )
        initial_fingerprint = gpt2_model_fingerprint(model)
        if (
            step_count is not None
            and started_step + step_count > spec.training.optimizer_steps
        ):
            raise GPT2TrainingError(
                "step_count exceeds the remaining optimizer schedule"
            )
        metrics = (
            ()
            if started_step == spec.training.optimizer_steps
            else train_optimizer_steps(
                model,
                cursor,
                optimizer,
                spec.training,
                start_step=started_step,
                step_count=step_count,
            )
        )
        completed_step = metrics[-1].step if metrics else started_step
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            cursor=cursor,
            config=spec.training,
            corpus_fingerprint=corpus.fingerprint(),
            step=completed_step,
        )
        return GPT2TrainingRun(
            spec=spec,
            corpus_fingerprint=corpus.fingerprint(),
            corpus_bytes=len(source),
            discarded_samples_per_epoch=cursor.discarded_samples_per_epoch,
            optimizer_groups=optimizer_group_summary(model),
            parameter_count=parameter_count,
            torch_threads=GPT2_TRAINING_TORCH_THREADS,
            started_step=started_step,
            completed_step=completed_step,
            initial_model_fingerprint=initial_fingerprint,
            final_model_fingerprint=gpt2_model_fingerprint(model),
            metrics=metrics,
        )
    finally:
        torch.set_rng_state(caller_rng)
        torch.set_num_threads(caller_threads)
