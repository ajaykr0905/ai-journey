"""Seal and independently recompute analytical scaling reports."""

from __future__ import annotations

import hashlib
import json

from .scaling_admission import CorpusAdmissionPolicy
from .scaling_laws import ScalingLawError
from .scaling_protocol import ScalingProtocol
from .scaling_report import build_scaling_report
from .scaling_runtime import RuntimeAssumptions
from .scaling_schedule import TrainingBatchGeometry


def _canonical_report_bytes(report: dict[str, object]) -> bytes:
    unsigned = {key: value for key, value in report.items() if key != "integrity"}
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_scaling_report(report: dict[str, object]) -> dict[str, object]:
    """Attach a stable SHA-256 digest without mutating the caller's mapping."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    sealed = {key: value for key, value in report.items() if key != "integrity"}
    sealed["integrity"] = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(_canonical_report_bytes(sealed)).hexdigest(),
    }
    return sealed


def _protocol_from_report(report: dict[str, object]) -> ScalingProtocol:
    try:
        raw = report["protocol"]
        if not isinstance(raw, dict):
            raise TypeError("protocol must be an object")
        batch = raw["batch"]
        if not isinstance(batch, dict):
            raise TypeError("batch must be an object")
        return ScalingProtocol(
            name=raw["name"],
            parameter_count=raw["parameter_count"],
            tokens_per_parameter=raw["tokens_per_parameter"],
            training_flops_per_parameter_token=raw[
                "training_flops_per_parameter_token"
            ],
            unique_train_tokens=raw["unique_train_tokens"],
            batch=TrainingBatchGeometry(**batch),
            maximum_overshoot_fraction=raw["maximum_overshoot_fraction"],
            isoflop_parameter_counts=tuple(raw["isoflop_parameter_counts"]),
            runtime=RuntimeAssumptions(**raw["runtime"]),
            corpus_policy=CorpusAdmissionPolicy(**raw["corpus_policy"]),
        )
    except (KeyError, TypeError, ScalingLawError) as error:
        raise ScalingLawError("invalid scaling report protocol") from error


def verify_scaling_report(report: dict[str, object]) -> dict[str, object]:
    """Verify integrity and recompute every derived report field."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ScalingLawError("unsupported scaling report schema")
    if report.get("kind") != "analytical_scaling_plan":
        raise ScalingLawError("unexpected scaling report kind")
    if report.get("measured") is not False:
        raise ScalingLawError("scaling report must remain unmeasured")
    try:
        expected_integrity = seal_scaling_report(report)["integrity"]
    except (TypeError, ValueError, OverflowError) as error:
        raise ScalingLawError("scaling report cannot be canonicalized") from error
    if report.get("integrity") != expected_integrity:
        raise ScalingLawError("scaling report integrity does not match")
    expected = seal_scaling_report(build_scaling_report(_protocol_from_report(report)))
    if report != expected:
        raise ScalingLawError("scaling report does not match recomputation")
    return report
