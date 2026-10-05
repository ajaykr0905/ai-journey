"""Predeclared, one-variable transformer ablation protocols."""

from __future__ import annotations

import json
import math
import platform
import random
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    StepMetric,
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    TransformerLabError,
    build_optimizer,
    evaluate_nll,
    model_fingerprint,
    seed_everything,
    train_steps,
)

ABLATION_SCHEMA_VERSION = 1
_LABEL_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_METRICS = {"train_nll", "validation_nll"}
_DIRECTIONS = {"lower", "higher"}


@contextmanager
def _preserve_random_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic)


def _batch_fingerprint(inputs: Tensor, targets: Tensor) -> str:
    digest = sha256()
    for tensor in (inputs, targets):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise TransformerLabError(f"{name} must not be blank")


def _flatten_config(
    model_config: TransformerConfig, training_config: TrainingConfig
) -> dict[str, Any]:
    values = {
        f"model.{field.name}": getattr(model_config, field.name)
        for field in fields(model_config)
    }
    values.update(
        {
            f"training.{field.name}": getattr(training_config, field.name)
            for field in fields(training_config)
        }
    )
    return values


@dataclass(frozen=True)
class AblationArm:
    """One named configuration in a controlled ablation."""

    label: str
    model_config: TransformerConfig
    training_config: TrainingConfig

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        if not _LABEL_PATTERN.fullmatch(self.label):
            raise TransformerLabError(
                "label must use lowercase letters, digits, and single hyphens"
            )
        if not isinstance(self.model_config, TransformerConfig):
            raise TypeError("model_config must be TransformerConfig")
        if not isinstance(self.training_config, TrainingConfig):
            raise TypeError("training_config must be TrainingConfig")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
        }


@dataclass(frozen=True)
class ControlledAblation:
    """A hypothesis and configurations fixed before measurements are produced."""

    name: str
    hypothesis: str
    independent_variable: str
    primary_metric: str
    expected_direction: str
    minimum_effect: float
    baseline_label: str
    trial_seeds: tuple[int, ...]
    arms: tuple[AblationArm, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not _LABEL_PATTERN.fullmatch(self.name):
            raise TransformerLabError(
                "name must use lowercase letters, digits, and single hyphens"
            )
        _require_text(self.hypothesis, "hypothesis")
        if self.primary_metric not in _METRICS:
            raise TransformerLabError(
                f"primary_metric must be one of {sorted(_METRICS)}"
            )
        if self.expected_direction not in _DIRECTIONS:
            raise TransformerLabError(
                f"expected_direction must be one of {sorted(_DIRECTIONS)}"
            )
        if (
            isinstance(self.minimum_effect, bool)
            or not isinstance(self.minimum_effect, (int, float))
            or not math.isfinite(self.minimum_effect)
            or self.minimum_effect < 0
        ):
            raise TransformerLabError("minimum_effect must be finite and non-negative")
        if not isinstance(self.trial_seeds, tuple):
            raise TypeError("trial_seeds must be a tuple")
        if not self.trial_seeds:
            raise TransformerLabError("trial_seeds must not be empty")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in self.trial_seeds
        ):
            raise TypeError("trial_seeds must contain integers")
        if len(set(self.trial_seeds)) != len(self.trial_seeds):
            raise TransformerLabError("trial_seeds must be unique")
        if not isinstance(self.arms, tuple):
            raise TypeError("arms must be a tuple")
        if len(self.arms) < 2:
            raise TransformerLabError("an ablation requires at least two arms")
        if any(not isinstance(arm, AblationArm) for arm in self.arms):
            raise TypeError("arms must contain AblationArm values")
        labels = tuple(arm.label for arm in self.arms)
        if len(set(labels)) != len(labels):
            raise TransformerLabError("arm labels must be unique")
        if self.baseline_label not in labels:
            raise TransformerLabError("baseline_label must identify an arm")

        baseline = self.baseline_arm
        baseline_values = _flatten_config(
            baseline.model_config, baseline.training_config
        )
        if self.independent_variable not in baseline_values:
            raise TransformerLabError("independent_variable is not a config field")
        if self.independent_variable in {"model.vocab_size", "training.seed"}:
            raise TransformerLabError(
                "vocab_size and seed are paired controls, not ablation variables"
            )
        independent_values: list[Any] = []
        for arm in self.arms:
            values = _flatten_config(arm.model_config, arm.training_config)
            differences = {
                name for name, value in values.items() if value != baseline_values[name]
            }
            if arm.label == self.baseline_label:
                if differences:
                    raise RuntimeError("baseline comparison is inconsistent")
            elif differences != {self.independent_variable}:
                rendered = ", ".join(sorted(differences)) or "none"
                raise TransformerLabError(
                    "every non-baseline arm must change only "
                    f"{self.independent_variable}; {arm.label} changed {rendered}"
                )
            independent_values.append(values[self.independent_variable])
        if len(
            {json.dumps(value, sort_keys=True) for value in independent_values}
        ) != len(independent_values):
            raise TransformerLabError("independent variable values must be unique")

    @property
    def baseline_arm(self) -> AblationArm:
        return next(arm for arm in self.arms if arm.label == self.baseline_label)

    def fixed_controls(self) -> dict[str, Any]:
        controls = _flatten_config(
            self.baseline_arm.model_config, self.baseline_arm.training_config
        )
        del controls[self.independent_variable]
        controls["paired_trial_seeds"] = list(self.trial_seeds)
        return controls

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ABLATION_SCHEMA_VERSION,
            "name": self.name,
            "hypothesis": self.hypothesis.strip(),
            "independent_variable": self.independent_variable,
            "primary_metric": self.primary_metric,
            "expected_direction": self.expected_direction,
            "minimum_effect": float(self.minimum_effect),
            "baseline_label": self.baseline_label,
            "trial_seeds": list(self.trial_seeds),
            "arms": [arm.to_dict() for arm in self.arms],
            "fixed_controls": self.fixed_controls(),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class AblationTrial:
    """One measured arm/seed pair with audit fingerprints."""

    arm_label: str
    seed: int
    independent_value: Any
    initial_model_fingerprint: str
    first_batch_fingerprint: str
    trace: tuple[StepMetric, ...]
    train_nll: float
    validation_nll: float
    final_model_fingerprint: str

    def metric(self, name: str) -> float:
        if name == "train_nll":
            return self.train_nll
        if name == "validation_nll":
            return self.validation_nll
        raise TransformerLabError(f"unsupported metric: {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_label": self.arm_label,
            "seed": self.seed,
            "independent_value": self.independent_value,
            "initial_model_fingerprint": self.initial_model_fingerprint,
            "first_batch_fingerprint": self.first_batch_fingerprint,
            "trace": [asdict(metric) for metric in self.trace],
            "train_nll": self.train_nll,
            "validation_nll": self.validation_nll,
            "final_model_fingerprint": self.final_model_fingerprint,
        }


@dataclass(frozen=True)
class AblationArmSummary:
    """Across-seed summary for one arm."""

    arm_label: str
    independent_value: Any
    trial_count: int
    mean_train_nll: float
    train_nll_sample_std: float
    mean_validation_nll: float
    validation_nll_sample_std: float


@dataclass(frozen=True)
class PairedContrast:
    """Per-seed primary-metric differences from the baseline arm."""

    arm_label: str
    baseline_label: str
    paired_deltas: tuple[float, ...]
    mean_delta: float
    sample_std: float


@dataclass(frozen=True)
class AblationRuntime:
    """Public-safe runtime provenance for the CPU experiment."""

    python_version: str
    torch_version: str
    numpy_version: str
    device: str
    machine: str
    deterministic_algorithms: bool
    intraop_threads: int


@dataclass(frozen=True)
class AblationResult:
    """Deterministic paired measurements and aggregate contrasts."""

    protocol: ControlledAblation
    corpus_fingerprint: str
    trials: tuple[AblationTrial, ...]
    summaries: tuple[AblationArmSummary, ...]
    contrasts: tuple[PairedContrast, ...]
    runtime: AblationRuntime

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": ABLATION_SCHEMA_VERSION,
            "protocol": self.protocol.to_dict(),
            "protocol_fingerprint": self.protocol.fingerprint(),
            "corpus_fingerprint": self.corpus_fingerprint,
            "trials": [trial.to_dict() for trial in self.trials],
            "summaries": [asdict(summary) for summary in self.summaries],
            "contrasts": [asdict(contrast) for contrast in self.contrasts],
            "runtime": asdict(self.runtime),
        }

    def evidence_fingerprint(self) -> str:
        payload = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["evidence_fingerprint"] = self.evidence_fingerprint()
        return payload


def _summarize_arm(
    arm: AblationArm,
    trials: tuple[AblationTrial, ...],
    independent_variable: str,
) -> AblationArmSummary:
    selected = tuple(trial for trial in trials if trial.arm_label == arm.label)
    train_values = tuple(trial.train_nll for trial in selected)
    validation_values = tuple(trial.validation_nll for trial in selected)
    independent_value = _flatten_config(arm.model_config, arm.training_config)[
        independent_variable
    ]
    return AblationArmSummary(
        arm_label=arm.label,
        independent_value=independent_value,
        trial_count=len(selected),
        mean_train_nll=fmean(train_values),
        train_nll_sample_std=stdev(train_values) if len(train_values) > 1 else 0.0,
        mean_validation_nll=fmean(validation_values),
        validation_nll_sample_std=(
            stdev(validation_values) if len(validation_values) > 1 else 0.0
        ),
    )


def _build_contrasts(
    protocol: ControlledAblation, trials: tuple[AblationTrial, ...]
) -> tuple[PairedContrast, ...]:
    indexed = {(trial.arm_label, trial.seed): trial for trial in trials}
    contrasts: list[PairedContrast] = []
    for arm in protocol.arms:
        if arm.label == protocol.baseline_label:
            continue
        deltas = tuple(
            indexed[(arm.label, seed)].metric(protocol.primary_metric)
            - indexed[(protocol.baseline_label, seed)].metric(protocol.primary_metric)
            for seed in protocol.trial_seeds
        )
        contrasts.append(
            PairedContrast(
                arm_label=arm.label,
                baseline_label=protocol.baseline_label,
                paired_deltas=deltas,
                mean_delta=fmean(deltas),
                sample_std=stdev(deltas) if len(deltas) > 1 else 0.0,
            )
        )
    return tuple(contrasts)


def _run_ablation(corpus: TokenCorpus, protocol: ControlledAblation) -> AblationResult:
    if not isinstance(corpus, TokenCorpus):
        raise TypeError("corpus must be TokenCorpus")
    if not isinstance(protocol, ControlledAblation):
        raise TypeError("protocol must be ControlledAblation")
    for arm in protocol.arms:
        if arm.model_config.vocab_size != corpus.vocab_size:
            raise TransformerLabError(
                f"{arm.label} vocabulary does not match the corpus"
            )

    trials: list[AblationTrial] = []
    for seed in protocol.trial_seeds:
        for arm in protocol.arms:
            training_config = TrainingConfig(
                **{**asdict(arm.training_config), "seed": seed}
            )
            seed_everything(seed)
            model = DecoderLanguageModel(arm.model_config)
            initial_fingerprint = model_fingerprint(model)
            preview = BatchCursor(
                corpus.train_tokens,
                block_size=arm.model_config.block_size,
                batch_size=training_config.batch_size,
                seed=seed,
            )
            first_inputs, first_targets = preview.next()
            cursor = BatchCursor(
                corpus.train_tokens,
                block_size=arm.model_config.block_size,
                batch_size=training_config.batch_size,
                seed=seed,
            )
            trace = train_steps(
                model,
                cursor,
                build_optimizer(model, training_config),
                training_config,
            )
            train_nll = evaluate_nll(model, corpus.train_tokens)
            validation_nll = evaluate_nll(model, corpus.validation_tokens)
            if not math.isfinite(train_nll) or not math.isfinite(validation_nll):
                raise TransformerLabError("ablation produced non-finite NLL")
            independent_value = _flatten_config(arm.model_config, arm.training_config)[
                protocol.independent_variable
            ]
            trials.append(
                AblationTrial(
                    arm_label=arm.label,
                    seed=seed,
                    independent_value=independent_value,
                    initial_model_fingerprint=initial_fingerprint,
                    first_batch_fingerprint=_batch_fingerprint(
                        first_inputs, first_targets
                    ),
                    trace=trace,
                    train_nll=train_nll,
                    validation_nll=validation_nll,
                    final_model_fingerprint=model_fingerprint(model),
                )
            )
    frozen_trials = tuple(trials)
    summaries = tuple(
        _summarize_arm(arm, frozen_trials, protocol.independent_variable)
        for arm in protocol.arms
    )
    return AblationResult(
        protocol=protocol,
        corpus_fingerprint=corpus.fingerprint(),
        trials=frozen_trials,
        summaries=summaries,
        contrasts=_build_contrasts(protocol, frozen_trials),
        runtime=AblationRuntime(
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            device="cpu",
            machine=platform.machine() or "unknown",
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            intraop_threads=torch.get_num_threads(),
        ),
    )


def run_ablation(corpus: TokenCorpus, protocol: ControlledAblation) -> AblationResult:
    """Run paired CPU trials without changing caller random state."""

    with _preserve_random_state():
        return _run_ablation(corpus, protocol)


@dataclass(frozen=True)
class AblationConclusion:
    """Direction-aware interpretation of one baseline contrast."""

    arm_label: str
    mean_delta: float
    oriented_effect: float
    outcome: str


@dataclass(frozen=True)
class AblationEvaluation:
    """Protocol-control checks and hypothesis outcomes."""

    outcome: str
    complete_pairs: bool
    matched_initial_models: bool | None
    matched_first_batches: bool | None
    conclusions: tuple[AblationConclusion, ...]


def _matched_within_seed(result: AblationResult, attribute: str) -> bool:
    return all(
        len(
            {getattr(trial, attribute) for trial in result.trials if trial.seed == seed}
        )
        == 1
        for seed in result.protocol.trial_seeds
    )


def evaluate_ablation(result: AblationResult) -> AblationEvaluation:
    """Interpret predeclared effects without hiding negative findings."""

    if not isinstance(result, AblationResult):
        raise TypeError("result must be AblationResult")
    protocol = result.protocol
    expected_pairs = {
        (arm.label, seed) for arm in protocol.arms for seed in protocol.trial_seeds
    }
    observed_pairs = {(trial.arm_label, trial.seed) for trial in result.trials}
    complete_pairs = observed_pairs == expected_pairs and len(result.trials) == len(
        expected_pairs
    )
    model_start_should_match = not protocol.independent_variable.startswith("model.")
    batch_should_match = protocol.independent_variable not in {
        "model.block_size",
        "training.batch_size",
    }
    matched_models = (
        _matched_within_seed(result, "initial_model_fingerprint")
        if model_start_should_match
        else None
    )
    matched_batches = (
        _matched_within_seed(result, "first_batch_fingerprint")
        if batch_should_match
        else None
    )
    conclusions: list[AblationConclusion] = []
    for contrast in result.contrasts:
        oriented_effect = (
            -contrast.mean_delta
            if protocol.expected_direction == "lower"
            else contrast.mean_delta
        )
        if oriented_effect > 0 and oriented_effect >= protocol.minimum_effect:
            outcome = "supports"
        elif oriented_effect < 0 and -oriented_effect >= protocol.minimum_effect:
            outcome = "contradicts"
        else:
            outcome = "inconclusive"
        conclusions.append(
            AblationConclusion(
                arm_label=contrast.arm_label,
                mean_delta=contrast.mean_delta,
                oriented_effect=oriented_effect,
                outcome=outcome,
            )
        )
    outcomes = {conclusion.outcome for conclusion in conclusions}
    if outcomes == {"supports"}:
        overall = "supports"
    elif outcomes == {"contradicts"}:
        overall = "contradicts"
    elif outcomes == {"inconclusive"}:
        overall = "inconclusive"
    else:
        overall = "mixed"
    return AblationEvaluation(
        outcome=overall,
        complete_pairs=complete_pairs,
        matched_initial_models=matched_models,
        matched_first_batches=matched_batches,
        conclusions=tuple(conclusions),
    )


def build_ablation_report(result: AblationResult) -> dict[str, Any]:
    """Build a canonical report with measurement and interpretation fingerprints."""

    evaluation = evaluate_ablation(result)
    payload: dict[str, Any] = {
        "result": result.to_dict(),
        "evaluation": asdict(evaluation),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["report_fingerprint"] = sha256(encoded).hexdigest()
    return payload


def verify_ablation_report(payload: dict[str, Any]) -> None:
    """Reject report payloads whose canonical fingerprint does not match."""

    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    supplied = payload.get("report_fingerprint")
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise TransformerLabError("report_fingerprint must be a SHA-256 digest")
    unsigned = {
        key: value for key, value in payload.items() if key != "report_fingerprint"
    }
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    expected = sha256(encoded).hexdigest()
    if supplied != expected:
        raise TransformerLabError("ablation report fingerprint does not match")


def write_ablation_report(path: Path, result: AblationResult) -> None:
    """Atomically publish a canonical JSON report."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    payload = build_ablation_report(result)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TransformerLabError(
            f"{name} fields do not match schema; missing={missing}, extra={extra}"
        )


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise TransformerLabError(f"duplicate JSON field: {key}")
        payload[key] = value
    return payload


def load_ablation_protocol(path: Path, *, vocab_size: int) -> ControlledAblation:
    """Load a strict JSON protocol and bind it to the observed vocabulary."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise TransformerLabError("vocab_size must be positive")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransformerLabError(f"cannot read ablation protocol: {exc}") from exc
    if not isinstance(payload, dict):
        raise TransformerLabError("ablation protocol root must be an object")
    top_fields = {
        "schema_version",
        "name",
        "hypothesis",
        "independent_variable",
        "primary_metric",
        "expected_direction",
        "minimum_effect",
        "baseline_label",
        "trial_seeds",
        "model_config",
        "training_config",
        "arms",
    }
    _require_exact_keys(payload, top_fields, "protocol")
    if payload["schema_version"] != ABLATION_SCHEMA_VERSION:
        raise TransformerLabError("unsupported ablation schema_version")
    model_payload = payload["model_config"]
    training_payload = payload["training_config"]
    if not isinstance(model_payload, dict) or not isinstance(training_payload, dict):
        raise TransformerLabError("model_config and training_config must be objects")
    model_fields = {field.name for field in fields(TransformerConfig)} - {"vocab_size"}
    training_fields = {field.name for field in fields(TrainingConfig)} - {"seed"}
    _require_exact_keys(model_payload, model_fields, "model_config")
    _require_exact_keys(training_payload, training_fields, "training_config")
    seeds = payload["trial_seeds"]
    if not isinstance(seeds, list):
        raise TransformerLabError("trial_seeds must be an array")
    trial_seeds = tuple(seeds)
    seed = trial_seeds[0] if trial_seeds else 0
    base_model = TransformerConfig(vocab_size=vocab_size, **model_payload)
    base_training = TrainingConfig(seed=seed, **training_payload)
    independent_variable = payload["independent_variable"]
    if not isinstance(independent_variable, str):
        raise TypeError("independent_variable must be a string")
    namespace, separator, field_name = independent_variable.partition(".")
    if separator != ".":
        raise TransformerLabError("independent_variable must include a namespace")
    arm_payloads = payload["arms"]
    if not isinstance(arm_payloads, list):
        raise TransformerLabError("arms must be an array")
    arms: list[AblationArm] = []
    for index, raw_arm in enumerate(arm_payloads):
        if not isinstance(raw_arm, dict):
            raise TransformerLabError(f"arms[{index}] must be an object")
        _require_exact_keys(raw_arm, {"label", "value"}, f"arms[{index}]")
        model = base_model
        training = base_training
        try:
            if namespace == "model":
                model = replace(base_model, **{field_name: raw_arm["value"]})
            elif namespace == "training":
                training = replace(base_training, **{field_name: raw_arm["value"]})
            else:
                raise TransformerLabError(
                    "independent_variable namespace must be model or training"
                )
        except TypeError as exc:
            raise TransformerLabError(
                f"invalid independent_variable field: {independent_variable}"
            ) from exc
        arms.append(AblationArm(raw_arm["label"], model, training))
    baseline_label = payload["baseline_label"]
    baseline_arm = next(
        (arm for arm in arms if arm.label == baseline_label),
        None,
    )
    if baseline_arm is not None:
        configured_value = _flatten_config(base_model, base_training).get(
            independent_variable
        )
        observed_value = _flatten_config(
            baseline_arm.model_config, baseline_arm.training_config
        ).get(independent_variable)
        if configured_value != observed_value:
            raise TransformerLabError(
                "baseline arm value must match the base configuration"
            )
    return ControlledAblation(
        name=payload["name"],
        hypothesis=payload["hypothesis"],
        independent_variable=independent_variable,
        primary_metric=payload["primary_metric"],
        expected_direction=payload["expected_direction"],
        minimum_effect=payload["minimum_effect"],
        baseline_label=baseline_label,
        trial_seeds=trial_seeds,
        arms=tuple(arms),
    )
