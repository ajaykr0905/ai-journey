"""Controlled fixed-normal versus Kaiming initialization experiments."""

from __future__ import annotations

import json
import math
import platform
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    StepMetric,
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    TransformerLabError,
    build_optimizer,
    evaluate_nll,
    expected_initialization_std,
    model_fingerprint,
    seed_everything,
    train_steps,
)

COMPARISON_SCHEMA_VERSION = 1


@contextmanager
def _preserve_random_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@dataclass(frozen=True)
class ParameterInitializationAudit:
    """Observed and theoretical statistics for one initialized weight matrix."""

    name: str
    module_type: str
    shape: tuple[int, ...]
    fan_in: int | None
    expected_std: float
    observed_mean: float
    observed_std: float
    relative_std_error: float
    bias_is_zero: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_model_initialization(
    model: DecoderLanguageModel,
) -> tuple[ParameterInitializationAudit, ...]:
    """Inspect every explicitly initialized matrix in stable module order."""

    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be a DecoderLanguageModel")
    audits: list[ParameterInitializationAudit] = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                continue
            weight = module.weight.detach()
            expected = expected_initialization_std(module, model.config)
            observed = float(weight.std(unbiased=False))
            bias_is_zero = None
            fan_in = None
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                bias_is_zero = module.bias is None or bool(
                    torch.count_nonzero(module.bias.detach()) == 0
                )
            audits.append(
                ParameterInitializationAudit(
                    name=f"{name}.weight",
                    module_type=type(module).__name__,
                    shape=tuple(weight.shape),
                    fan_in=fan_in,
                    expected_std=expected,
                    observed_mean=float(weight.mean()),
                    observed_std=observed,
                    relative_std_error=abs(observed - expected) / expected,
                    bias_is_zero=bias_is_zero,
                )
            )
    return tuple(audits)


@dataclass(frozen=True)
class KaimingVariantResult:
    """Training evidence for one controlled initialization policy."""

    name: str
    model_config: TransformerConfig
    initialization_audit: tuple[ParameterInitializationAudit, ...]
    embedding_fingerprint: str
    first_batch_fingerprint: str
    initial_train_nll: float
    initial_validation_nll: float
    trace: tuple[StepMetric, ...]
    loss_curve: LossCurveMetrics
    final_train_nll: float
    final_validation_nll: float
    model_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_config": asdict(self.model_config),
            "initialization_audit": [
                item.to_dict() for item in self.initialization_audit
            ],
            "embedding_fingerprint": self.embedding_fingerprint,
            "first_batch_fingerprint": self.first_batch_fingerprint,
            "initial_train_nll": self.initial_train_nll,
            "initial_validation_nll": self.initial_validation_nll,
            "trace": [asdict(metric) for metric in self.trace],
            "loss_curve": asdict(self.loss_curve),
            "final_train_nll": self.final_train_nll,
            "final_validation_nll": self.final_validation_nll,
            "model_fingerprint": self.model_fingerprint,
        }


@dataclass(frozen=True)
class KaimingComparisonResult:
    """Fixed-normal and Kaiming runs with all training controls held constant."""

    corpus_fingerprint: str
    training_config: TrainingConfig
    variants: tuple[KaimingVariantResult, KaimingVariantResult]
    contrast: KaimingContrast
    runtime: RuntimeMetadata

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "corpus_fingerprint": self.corpus_fingerprint,
            "training_config": asdict(self.training_config),
            "variants": [variant.to_dict() for variant in self.variants],
            "contrast": asdict(self.contrast),
            "runtime": asdict(self.runtime),
        }

    def evidence_fingerprint(self) -> str:
        """Hash the canonical evidence payload for later integrity checks."""

        encoded = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["evidence_fingerprint"] = self.evidence_fingerprint()
        return payload


@dataclass(frozen=True)
class RuntimeMetadata:
    """Public-safe runtime provenance for reproducing the CPU comparison."""

    python_version: str
    torch_version: str
    numpy_version: str
    device: str
    machine: str
    deterministic_algorithms: bool
    intraop_threads: int


@dataclass(frozen=True)
class LossCurveMetrics:
    """Stable summary of a finite per-step training-loss trace."""

    step_count: int
    initial_loss: float
    final_loss: float
    best_loss: float
    best_step: int
    mean_loss: float
    relative_loss_reduction: float
    improving_transition_fraction: float


@dataclass(frozen=True)
class KaimingContrast:
    """Direct comparison of the two matched loss curves and held-out results."""

    fixed_mean_loss: float
    kaiming_mean_loss: float
    kaiming_to_fixed_mean_loss_ratio: float
    fixed_final_train_nll: float
    kaiming_final_train_nll: float
    final_train_nll_delta: float
    fixed_final_validation_nll: float
    kaiming_final_validation_nll: float
    final_validation_nll_delta: float


@dataclass(frozen=True)
class ComparisonCriteria:
    """Explicit evidence thresholds for the controlled comparison."""

    max_relative_std_error: float = 0.25
    min_variant_loss_reduction: float = 0.01
    min_kaiming_mean_loss_improvement: float = 0.01

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_relative_std_error, (int, float))
            or isinstance(self.max_relative_std_error, bool)
            or not math.isfinite(self.max_relative_std_error)
            or self.max_relative_std_error < 0
        ):
            raise TransformerLabError(
                "max_relative_std_error must be finite and non-negative"
            )
        for name in (
            "min_variant_loss_reduction",
            "min_kaiming_mean_loss_improvement",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                raise TransformerLabError(f"{name} must be finite and in [0, 1)")


@dataclass(frozen=True)
class ComparisonEvaluation:
    """Measured gate values and named violations for review and CI."""

    passed: bool
    maximum_relative_std_error: float
    minimum_variant_loss_reduction: float
    kaiming_mean_loss_improvement: float
    violations: tuple[str, ...]


def evaluate_comparison(
    result: KaimingComparisonResult,
    criteria: ComparisonCriteria | None = None,
) -> ComparisonEvaluation:
    """Evaluate initialization fidelity and the predeclared loss-curve effect."""

    if not isinstance(result, KaimingComparisonResult):
        raise TypeError("result must be a KaimingComparisonResult")
    policy = criteria or ComparisonCriteria()
    audits = tuple(
        audit for variant in result.variants for audit in variant.initialization_audit
    )
    maximum_error = max(audit.relative_std_error for audit in audits)
    minimum_reduction = min(
        variant.loss_curve.relative_loss_reduction for variant in result.variants
    )
    improvement = 1 - result.contrast.kaiming_to_fixed_mean_loss_ratio
    violations: list[str] = []
    if any(audit.bias_is_zero is False for audit in audits):
        violations.append("linear_bias_not_zero")
    if maximum_error > policy.max_relative_std_error:
        violations.append("initialization_std_error")
    if minimum_reduction < policy.min_variant_loss_reduction:
        violations.append("insufficient_variant_loss_reduction")
    if improvement < policy.min_kaiming_mean_loss_improvement:
        violations.append("insufficient_kaiming_mean_loss_improvement")
    return ComparisonEvaluation(
        passed=not violations,
        maximum_relative_std_error=maximum_error,
        minimum_variant_loss_reduction=minimum_reduction,
        kaiming_mean_loss_improvement=improvement,
        violations=tuple(violations),
    )


def build_comparison_report(
    result: KaimingComparisonResult,
    criteria: ComparisonCriteria | None = None,
) -> dict[str, Any]:
    """Build a self-verifying report containing experiment and gate evidence."""

    if not isinstance(result, KaimingComparisonResult):
        raise TypeError("result must be a KaimingComparisonResult")
    policy = criteria or ComparisonCriteria()
    evaluation = evaluate_comparison(result, policy)
    payload = {
        "experiment": result.to_dict(),
        "criteria": asdict(policy),
        "evaluation": asdict(evaluation),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["report_fingerprint"] = sha256(encoded).hexdigest()
    return payload


def write_comparison_report(
    path: Path,
    result: KaimingComparisonResult,
    criteria: ComparisonCriteria | None = None,
) -> None:
    """Atomically write stable JSON evidence for the controlled comparison."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    payload = build_comparison_report(result, criteria)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def render_loss_curves(path: Path, result: KaimingComparisonResult) -> None:
    """Render deterministic matched training-loss curves as an atomic SVG."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(result, KaimingComparisonResult):
        raise TypeError("result must be a KaimingComparisonResult")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    colors = {"fixed_normal": "#DC2626", "kaiming_normal": "#2563EB"}
    with matplotlib.rc_context({"svg.hashsalt": "ai-journey-day-26-loss-curves"}):
        figure, axis = plt.subplots(figsize=(9, 5))
        for variant in result.variants:
            axis.plot(
                [metric.step for metric in variant.trace],
                [metric.loss for metric in variant.trace],
                label=(f"{variant.name} · mean {variant.loss_curve.mean_loss:.4f}"),
                color=colors[variant.name],
                linewidth=1.8,
            )
        axis.set_title("Matched transformer loss curves")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Batch cross-entropy")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(temporary, format="svg", metadata={"Date": None})
        plt.close(figure)
    temporary.replace(path)


def render_initialization_audit(path: Path, result: KaimingComparisonResult) -> None:
    """Render expected and observed standard deviations for linear weights."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(result, KaimingComparisonResult):
        raise TypeError("result must be a KaimingComparisonResult")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    linear_names = tuple(
        item.name
        for item in result.variants[0].initialization_audit
        if item.module_type == "Linear"
    )
    if not linear_names:
        raise TransformerLabError("comparison contains no linear initialization audits")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with matplotlib.rc_context({"svg.hashsalt": "ai-journey-day-26-init-audit"}):
        figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        positions = list(range(len(linear_names)))
        for axis, variant in zip(axes, result.variants, strict=True):
            by_name = {item.name: item for item in variant.initialization_audit}
            expected = [by_name[name].expected_std for name in linear_names]
            observed = [by_name[name].observed_std for name in linear_names]
            axis.bar(
                [position - 0.2 for position in positions],
                expected,
                width=0.4,
                label="expected",
                color="#94A3B8",
            )
            axis.bar(
                [position + 0.2 for position in positions],
                observed,
                width=0.4,
                label="observed",
                color="#2563EB",
            )
            axis.set_title(variant.name)
            axis.set_ylabel("Weight standard deviation")
            axis.grid(axis="y", alpha=0.2)
            axis.legend()
        axes[-1].set_xticks(positions, linear_names, rotation=35, ha="right")
        figure.suptitle("Linear-weight initialization audit")
        figure.tight_layout()
        figure.savefig(temporary, format="svg", metadata={"Date": None})
        plt.close(figure)
    temporary.replace(path)


def summarize_loss_curve(trace: tuple[StepMetric, ...]) -> LossCurveMetrics:
    """Summarize a non-empty, finite, strictly ordered loss trace."""

    if not trace:
        raise TransformerLabError("loss trace must not be empty")
    if any(not math.isfinite(metric.loss) for metric in trace):
        raise TransformerLabError("loss trace must contain only finite losses")
    if any(right.step <= left.step for left, right in pairwise(trace)):
        raise TransformerLabError("loss trace steps must be strictly increasing")
    losses = tuple(metric.loss for metric in trace)
    best_index = min(range(len(trace)), key=lambda index: losses[index])
    improving = sum(right < left for left, right in pairwise(losses))
    transitions = max(len(losses) - 1, 1)
    return LossCurveMetrics(
        step_count=len(trace),
        initial_loss=losses[0],
        final_loss=losses[-1],
        best_loss=losses[best_index],
        best_step=trace[best_index].step,
        mean_loss=math.fsum(losses) / len(losses),
        relative_loss_reduction=(losses[0] - losses[-1]) / losses[0],
        improving_transition_fraction=improving / transitions,
    )


def _run_variant(
    name: str,
    corpus: TokenCorpus,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
) -> KaimingVariantResult:
    seed_everything(training_config.seed)
    model = DecoderLanguageModel(model_config)
    audit = audit_model_initialization(model)
    embedding_digest = sha256()
    embedding_digest.update(model.token_embedding.weight.detach().numpy().tobytes())
    embedding_digest.update(model.position_embedding.weight.detach().numpy().tobytes())
    batch_probe = BatchCursor(
        corpus.train_tokens,
        block_size=model_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    first_inputs, first_targets = batch_probe.next()
    batch_digest = sha256()
    batch_digest.update(first_inputs.numpy().tobytes())
    batch_digest.update(first_targets.numpy().tobytes())
    initial_train = evaluate_nll(model, corpus.train_tokens)
    initial_validation = evaluate_nll(model, corpus.validation_tokens)
    cursor = BatchCursor(
        corpus.train_tokens,
        block_size=model_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    trace = train_steps(
        model,
        cursor,
        build_optimizer(model, training_config),
        training_config,
    )
    return KaimingVariantResult(
        name=name,
        model_config=model_config,
        initialization_audit=audit,
        embedding_fingerprint=embedding_digest.hexdigest(),
        first_batch_fingerprint=batch_digest.hexdigest(),
        initial_train_nll=initial_train,
        initial_validation_nll=initial_validation,
        trace=trace,
        loss_curve=summarize_loss_curve(trace),
        final_train_nll=evaluate_nll(model, corpus.train_tokens),
        final_validation_nll=evaluate_nll(model, corpus.validation_tokens),
        model_fingerprint=model_fingerprint(model),
    )


def run_kaiming_comparison(
    corpus: TokenCorpus,
    *,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
) -> KaimingComparisonResult:
    """Run matched fixed-normal and Kaiming transformer training curves."""

    if model_config.vocab_size != corpus.vocab_size:
        raise TransformerLabError("model vocabulary does not match corpus codec")
    if model_config.initialization_mode != "fixed_normal":
        raise TransformerLabError("comparison baseline must use fixed_normal")
    with _preserve_random_state():
        variants = (
            _run_variant("fixed_normal", corpus, model_config, training_config),
            _run_variant(
                "kaiming_normal",
                corpus,
                replace(model_config, initialization_mode="kaiming_normal"),
                training_config,
            ),
        )
        runtime = RuntimeMetadata(
            python_version=platform.python_version(),
            torch_version=str(torch.__version__),
            numpy_version=str(np.__version__),
            device="cpu",
            machine=platform.machine(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            intraop_threads=torch.get_num_threads(),
        )
    fixed, kaiming = variants
    contrast = KaimingContrast(
        fixed_mean_loss=fixed.loss_curve.mean_loss,
        kaiming_mean_loss=kaiming.loss_curve.mean_loss,
        kaiming_to_fixed_mean_loss_ratio=(
            kaiming.loss_curve.mean_loss / fixed.loss_curve.mean_loss
        ),
        fixed_final_train_nll=fixed.final_train_nll,
        kaiming_final_train_nll=kaiming.final_train_nll,
        final_train_nll_delta=kaiming.final_train_nll - fixed.final_train_nll,
        fixed_final_validation_nll=fixed.final_validation_nll,
        kaiming_final_validation_nll=kaiming.final_validation_nll,
        final_validation_nll_delta=(
            kaiming.final_validation_nll - fixed.final_validation_nll
        ),
    )
    return KaimingComparisonResult(
        corpus_fingerprint=corpus.fingerprint(),
        training_config=training_config,
        variants=variants,
        contrast=contrast,
        runtime=runtime,
    )
