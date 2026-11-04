"""Independent recomputation and integrity checks for inference reports."""

from __future__ import annotations

from .inference_protocol import InferenceProtocol
from .inference_report import build_inference_report
from .inference_seal import (
    canonical_inference_bytes,
    inference_report_digest,
    seal_inference_report,
)
from .inference_speculative import SpeculativePlan


class InferenceVerificationError(ValueError):
    """Raised when saved inference evidence cannot be reproduced exactly."""


def _reconstruct_protocol(mapping: object) -> InferenceProtocol:
    if type(mapping) is not dict:
        raise InferenceVerificationError("protocol must be an object")
    values = dict(mapping)
    speculative = values.get("speculative")
    if type(speculative) is not dict:
        raise InferenceVerificationError("speculative protocol must be an object")
    values["speculative"] = SpeculativePlan(**speculative)
    request_lengths = values.get("request_lengths")
    if type(request_lengths) is not list:
        raise InferenceVerificationError("request_lengths must be a JSON array")
    values["request_lengths"] = tuple(request_lengths)
    try:
        return InferenceProtocol(**values)
    except (TypeError, ValueError) as error:
        raise InferenceVerificationError(f"invalid protocol: {error}") from error


def verify_inference_report(report: object) -> InferenceProtocol:
    """Verify schema, evidence boundaries, digest, and all derived fields."""

    if type(report) is not dict:
        raise InferenceVerificationError("report must be an object")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "analytical_inference_capacity_plan"
        or report.get("measured") is not False
    ):
        raise InferenceVerificationError("unsupported inference report contract")
    boundaries = report.get("evidence_boundaries")
    if type(boundaries) is not dict or boundaries != {
        "analytical_only": True,
        "gpu_executed": False,
        "server_executed": False,
        "latency_measured": False,
        "learner_completion": False,
    }:
        raise InferenceVerificationError("evidence boundaries were weakened")
    digest = report.get("report_sha256")
    if type(digest) is not str or digest != inference_report_digest(report):
        raise InferenceVerificationError("report digest mismatch")
    protocol = _reconstruct_protocol(report.get("protocol"))
    expected = seal_inference_report(build_inference_report(protocol))
    if canonical_inference_bytes(report) != canonical_inference_bytes(expected):
        raise InferenceVerificationError("derived inference fields do not recompute")
    return protocol
