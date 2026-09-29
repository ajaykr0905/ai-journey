"""Controlled initialization-scale diagnostics for the transformer lab."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from ai_journey.training_diagnostics import (
    DiagnosticHealthReport,
    HealthThresholds,
    HistogramSpec,
    ParameterUpdate,
    ParameterUpdateTracker,
    TrainingSnapshot,
    capture_training_snapshot,
    evaluate_snapshot_health,
)
from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    StepMetric,
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    TransformerLabError,
    build_optimizer,
    model_fingerprint,
    seed_everything,
    train_steps,
)

COMPARISON_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InitializationVariantResult:
    """Diagnostics and training evidence for one initialization scale."""

    name: str
    model_config: TransformerConfig
    initial_snapshot: TrainingSnapshot
    final_snapshot: TrainingSnapshot
    initial_health: DiagnosticHealthReport
    final_health: DiagnosticHealthReport
    first_step_updates: tuple[ParameterUpdate, ...]
    trace: tuple[StepMetric, ...]
    model_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_config": asdict(self.model_config),
            "initial_snapshot": self.initial_snapshot.to_dict(),
            "final_snapshot": self.final_snapshot.to_dict(),
            "initial_health": self.initial_health.to_dict(),
            "final_health": self.final_health.to_dict(),
            "first_step_updates": [
                asdict(update) for update in self.first_step_updates
            ],
            "trace": [asdict(metric) for metric in self.trace],
            "model_fingerprint": self.model_fingerprint,
        }


@dataclass(frozen=True)
class InitializationContrast:
    """Directly comparable effect size for the controlled initialization change."""

    activation_module: str
    baseline_initial_rms: float
    stressed_initial_rms: float
    stressed_to_baseline_rms_ratio: float
    baseline_initial_loss: float
    stressed_initial_loss: float


@dataclass(frozen=True)
class RuntimeMetadata:
    """Public-safe software and execution context for the CPU experiment."""

    python_version: str
    torch_version: str
    device: str
    deterministic_algorithms: bool


@dataclass(frozen=True)
class InitializationComparisonResult:
    """Reproducible baseline-versus-stressed initialization comparison."""

    corpus_fingerprint: str
    training_config: TrainingConfig
    activation_modules: tuple[str, ...]
    activation_histogram: HistogramSpec
    gradient_histogram: HistogramSpec
    health_thresholds: HealthThresholds
    variants: tuple[InitializationVariantResult, ...]
    contrast: InitializationContrast
    runtime: RuntimeMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "corpus_fingerprint": self.corpus_fingerprint,
            "training_config": asdict(self.training_config),
            "activation_modules": list(self.activation_modules),
            "activation_histogram": asdict(self.activation_histogram),
            "gradient_histogram": asdict(self.gradient_histogram),
            "health_thresholds": asdict(self.health_thresholds),
            "variants": [variant.to_dict() for variant in self.variants],
            "contrast": asdict(self.contrast),
            "runtime": asdict(self.runtime),
        }


def default_activation_modules(config: TransformerConfig) -> tuple[str, ...]:
    """Return every transformer-block GELU module in stable depth order."""

    return tuple(
        f"blocks.{index}.feed_forward.network.1" for index in range(config.layer_count)
    )


def _run_variant(
    name: str,
    corpus: TokenCorpus,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    module_names: tuple[str, ...],
    activation_spec: HistogramSpec,
    gradient_spec: HistogramSpec,
    thresholds: HealthThresholds,
) -> InitializationVariantResult:
    seed_everything(training_config.seed)
    model = DecoderLanguageModel(model_config)
    optimizer = build_optimizer(model, training_config)
    diagnostic_cursor = BatchCursor(
        corpus.train_tokens,
        block_size=model_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    inputs, targets = diagnostic_cursor.next()
    initial = capture_training_snapshot(
        model,
        inputs,
        targets,
        module_names,
        activation_spec=activation_spec,
        gradient_spec=gradient_spec,
    )
    training_cursor = BatchCursor(
        corpus.train_tokens,
        block_size=model_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    update_tracker = ParameterUpdateTracker(model)
    first = train_steps(
        model,
        training_cursor,
        optimizer,
        training_config,
        step_count=1,
    )
    updates = update_tracker.measure(model)
    remaining = training_config.steps - 1
    rest = (
        train_steps(
            model,
            training_cursor,
            optimizer,
            training_config,
            start_step=1,
            step_count=remaining,
        )
        if remaining
        else ()
    )
    final = capture_training_snapshot(
        model,
        inputs,
        targets,
        module_names,
        activation_spec=activation_spec,
        gradient_spec=gradient_spec,
    )
    return InitializationVariantResult(
        name=name,
        model_config=model_config,
        initial_snapshot=initial,
        final_snapshot=final,
        initial_health=evaluate_snapshot_health(initial, thresholds),
        final_health=evaluate_snapshot_health(final, thresholds),
        first_step_updates=updates,
        trace=first + rest,
        model_fingerprint=model_fingerprint(model),
    )


def run_initialization_comparison(
    corpus: TokenCorpus,
    *,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    stressed_initialization_std: float = 1.0,
    activation_modules: tuple[str, ...] | None = None,
    activation_histogram: HistogramSpec | None = None,
    gradient_histogram: HistogramSpec | None = None,
    health_thresholds: HealthThresholds | None = None,
) -> InitializationComparisonResult:
    """Compare one controlled initialization-scale change on the same batch."""

    if model_config.vocab_size != corpus.vocab_size:
        raise TransformerLabError("model vocabulary does not match corpus codec")
    if (
        isinstance(stressed_initialization_std, bool)
        or not isinstance(stressed_initialization_std, (int, float))
        or stressed_initialization_std <= 0
    ):
        raise TransformerLabError("stressed_initialization_std must be positive")
    if stressed_initialization_std == model_config.initialization_std:
        raise TransformerLabError("stressed initialization must differ from baseline")
    modules = activation_modules or default_activation_modules(model_config)
    activation_spec = activation_histogram or HistogramSpec(
        lower=-5, upper=5, bins=50, near_zero=1e-6
    )
    gradient_spec = gradient_histogram or HistogramSpec(
        lower=-1, upper=1, bins=50, near_zero=1e-10
    )
    thresholds = health_thresholds or HealthThresholds(
        max_out_of_range_fraction=0.1,
        max_near_zero_fraction=0.999,
        min_standard_deviation=1e-12,
    )
    variants = (
        _run_variant(
            "baseline",
            corpus,
            model_config,
            training_config,
            modules,
            activation_spec,
            gradient_spec,
            thresholds,
        ),
        _run_variant(
            "stressed",
            corpus,
            replace(model_config, initialization_std=stressed_initialization_std),
            training_config,
            modules,
            activation_spec,
            gradient_spec,
            thresholds,
        ),
    )
    baseline_activation = variants[0].initial_snapshot.activations[0].distribution
    stressed_activation = variants[1].initial_snapshot.activations[0].distribution
    if baseline_activation.rms == 0:
        raise TransformerLabError("baseline activation RMS is zero")
    contrast = InitializationContrast(
        activation_module=modules[0],
        baseline_initial_rms=baseline_activation.rms,
        stressed_initial_rms=stressed_activation.rms,
        stressed_to_baseline_rms_ratio=(
            stressed_activation.rms / baseline_activation.rms
        ),
        baseline_initial_loss=variants[0].initial_snapshot.loss,
        stressed_initial_loss=variants[1].initial_snapshot.loss,
    )
    return InitializationComparisonResult(
        corpus_fingerprint=corpus.fingerprint(),
        training_config=training_config,
        activation_modules=modules,
        activation_histogram=activation_spec,
        gradient_histogram=gradient_spec,
        health_thresholds=thresholds,
        variants=variants,
        contrast=contrast,
        runtime=RuntimeMetadata(
            python_version=platform.python_version(),
            torch_version=str(torch.__version__),
            device="cpu",
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        ),
    )


def contrast_meets_minimum(
    result: InitializationComparisonResult, minimum_rms_ratio: float
) -> bool:
    """Return whether the stressed initialization produced the required effect."""

    if (
        isinstance(minimum_rms_ratio, bool)
        or not isinstance(minimum_rms_ratio, (int, float))
        or minimum_rms_ratio <= 1
    ):
        raise TransformerLabError("minimum_rms_ratio must be greater than one")
    return result.contrast.stressed_to_baseline_rms_ratio >= minimum_rms_ratio


def write_comparison_report(path: Path, result: InitializationComparisonResult) -> None:
    """Atomically write stable JSON evidence for a diagnostic comparison."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(result, InitializationComparisonResult):
        raise TypeError("result must be an InitializationComparisonResult")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def render_activation_histograms(
    path: Path,
    result: InitializationComparisonResult,
    *,
    module_name: str | None = None,
) -> None:
    """Render deterministic initial/final activation histograms as SVG."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(result, InitializationComparisonResult):
        raise TypeError("result must be an InitializationComparisonResult")
    selected_module = module_name or result.activation_modules[0]
    if selected_module not in result.activation_modules:
        raise TransformerLabError(
            f"activation module was not captured: {selected_module}"
        )

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with matplotlib.rc_context({"svg.hashsalt": "ai-journey-day-25"}):
        figure, axes = plt.subplots(
            len(result.variants),
            2,
            figsize=(10, 3.5 * len(result.variants)),
            squeeze=False,
        )
        for row, variant in enumerate(result.variants):
            for column, (phase, snapshot) in enumerate(
                (
                    ("initial", variant.initial_snapshot),
                    ("final", variant.final_snapshot),
                )
            ):
                distributions = {
                    item.name: item.distribution for item in snapshot.activations
                }
                distribution = distributions[selected_module]
                edges = distribution.histogram_edges
                widths = [right - left for left, right in pairwise(edges)]
                axis = axes[row][column]
                axis.bar(
                    edges[:-1],
                    distribution.histogram_counts,
                    width=widths,
                    align="edge",
                    color="#2563EB" if variant.name == "baseline" else "#DC2626",
                    edgecolor="white",
                    linewidth=0.3,
                )
                axis.set_title(
                    f"{variant.name} {phase} · outside range "
                    f"{distribution.underflow_count + distribution.overflow_count}"
                )
                axis.set_xlabel("Activation value")
                axis.set_ylabel("Count")
                axis.grid(axis="y", alpha=0.2)
        figure.suptitle(f"Activation distributions · {selected_module}")
        figure.tight_layout()
        figure.savefig(temporary, format="svg", metadata={"Date": None})
        plt.close(figure)
    temporary.replace(path)


def render_gradient_histograms(
    path: Path,
    result: InitializationComparisonResult,
    *,
    parameter_name: str | None = None,
) -> None:
    """Render deterministic initial/final parameter-gradient histograms as SVG."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(result, InitializationComparisonResult):
        raise TypeError("result must be an InitializationComparisonResult")
    first_gradients = result.variants[0].initial_snapshot.parameter_gradients.gradients
    if not first_gradients:
        raise TransformerLabError("comparison contains no parameter gradients")
    selected_parameter = parameter_name or first_gradients[0].name
    available = {item.name for item in first_gradients}
    if selected_parameter not in available:
        raise TransformerLabError(
            f"gradient parameter was not captured: {selected_parameter}"
        )

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with matplotlib.rc_context({"svg.hashsalt": "ai-journey-day-25-gradients"}):
        figure, axes = plt.subplots(
            len(result.variants),
            2,
            figsize=(10, 3.5 * len(result.variants)),
            squeeze=False,
        )
        for row, variant in enumerate(result.variants):
            for column, (phase, snapshot) in enumerate(
                (
                    ("initial", variant.initial_snapshot),
                    ("final", variant.final_snapshot),
                )
            ):
                distributions = {
                    item.name: item.distribution
                    for item in snapshot.parameter_gradients.gradients
                }
                if selected_parameter not in distributions:
                    raise TransformerLabError(
                        f"gradient parameter was not captured: {selected_parameter}"
                    )
                distribution = distributions[selected_parameter]
                edges = distribution.histogram_edges
                widths = [right - left for left, right in pairwise(edges)]
                axis = axes[row][column]
                axis.bar(
                    edges[:-1],
                    distribution.histogram_counts,
                    width=widths,
                    align="edge",
                    color="#2563EB" if variant.name == "baseline" else "#DC2626",
                    edgecolor="white",
                    linewidth=0.3,
                )
                axis.set_title(f"{variant.name} {phase}")
                axis.set_xlabel("Gradient value")
                axis.set_ylabel("Count")
                axis.grid(axis="y", alpha=0.2)
        figure.suptitle(f"Gradient distributions · {selected_parameter}")
        figure.tight_layout()
        figure.savefig(temporary, format="svg", metadata={"Date": None})
        plt.close(figure)
    temporary.replace(path)
