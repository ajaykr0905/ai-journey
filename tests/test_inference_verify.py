from copy import deepcopy

import pytest

from ai_journey.inference_protocol import InferenceProtocol
from ai_journey.inference_report import build_inference_report
from ai_journey.inference_seal import seal_inference_report
from ai_journey.inference_speculative import SpeculativePlan
from ai_journey.inference_verify import (
    InferenceVerificationError,
    verify_inference_report,
)


def sealed_report():
    gib = 1024**3
    protocol = InferenceProtocol(
        "verify-inference",
        7_000_000_000,
        32,
        32,
        8,
        128,
        32_768,
        (32_768,) * 16,
        2,
        80 * gib,
        14 * gib,
        6 * gib,
        16,
        SpeculativePlan(4, 4, 5),
    )
    return seal_inference_report(build_inference_report(protocol))


def test_verifier_recomputes_every_report_field():
    assert verify_inference_report(sealed_report()).name == "verify-inference"


def test_verifier_rejects_tampered_derived_values_even_with_stale_digest():
    report = deepcopy(sealed_report())
    report["kv_cache"]["gqa_total_bytes"] += 1

    with pytest.raises(InferenceVerificationError, match="digest"):
        verify_inference_report(report)


def test_verifier_rejects_weakened_evidence_boundaries():
    report = deepcopy(sealed_report())
    report["evidence_boundaries"]["gpu_executed"] = True

    with pytest.raises(InferenceVerificationError, match="boundaries"):
        verify_inference_report(report)
