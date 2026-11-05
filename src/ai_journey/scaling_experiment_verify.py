"""Canonical integrity and independent recomputation for scaling experiments."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .scaling_experiment_analysis import ScalingAcceptancePolicy
from .scaling_experiment_identity import ExperimentInputs
from .scaling_experiment_observation import ScalingObservation
from .scaling_experiment_protocol import ExperimentPoint, ScalingExperimentProtocol
from .scaling_experiment_report import build_experiment_report
from .scaling_laws import ScalingLawError


def _canonical(report: dict[str, object]) -> bytes:
    return json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def seal_experiment_report(report: dict[str, object]) -> dict[str, object]:
    """Attach a deterministic digest without mutating the caller's report."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    sealed = deepcopy(report)
    sealed.pop("integrity", None)
    sealed["integrity"] = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(_canonical(sealed)).hexdigest(),
    }
    return sealed


def _rebuild(report: dict[str, object]) -> dict[str, object]:
    protocol_data = report["protocol"]
    protocol = ScalingExperimentProtocol(
        name=protocol_data["name"],
        parameter_counts=tuple(protocol_data["parameter_counts"]),
        train_token_counts=tuple(protocol_data["train_token_counts"]),
        seeds=tuple(protocol_data["seeds"]),
    )
    inputs = ExperimentInputs(**report["inputs"])
    observations = tuple(
        ScalingObservation(
            run_id=row["run_id"],
            point=ExperimentPoint(**row["point"]),
            validation_loss=row["validation_loss"],
            trained_tokens=row["trained_tokens"],
            optimizer_steps=row["optimizer_steps"],
            elapsed_seconds=row["elapsed_seconds"],
        )
        for row in report["observations"]
    )
    policy = ScalingAcceptancePolicy(**report["policy"])
    return build_experiment_report(
        protocol,
        inputs,
        observations,
        target_compute=report["target"]["compute"],
        policy=policy,
    )


def verify_experiment_report(report: dict[str, object]) -> dict[str, object]:
    """Verify integrity, schema, and every recomputable report field."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    try:
        integrity = report["integrity"]
        if integrity.get("algorithm") != "sha256":
            raise ScalingLawError("unsupported scaling experiment integrity algorithm")
        unsigned = deepcopy(report)
        unsigned.pop("integrity")
        digest = hashlib.sha256(_canonical(unsigned)).hexdigest()
        if digest != integrity.get("digest"):
            raise ScalingLawError("scaling experiment integrity mismatch")
        expected = _rebuild(unsigned)
    except ScalingLawError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ScalingLawError("invalid scaling experiment report structure") from error
    if _canonical(expected) != _canonical(unsigned):
        raise ScalingLawError("scaling experiment report recomputation mismatch")
    return report
