"""Predeclared, one-variable transformer ablation protocols."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from typing import Any

from .transformer_lab import TrainingConfig, TransformerConfig, TransformerLabError

ABLATION_SCHEMA_VERSION = 1
_LABEL_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_METRICS = {"train_nll", "validation_nll"}
_DIRECTIONS = {"lower", "higher"}


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
