from copy import deepcopy

import pytest

from ai_journey.scaling_laws import ScalingLawError
from ai_journey.scaling_report import build_scaling_report
from ai_journey.scaling_verify import (
    seal_scaling_report,
    verify_scaling_report,
)
from test_scaling_report import make_protocol


def test_scaling_report_seal_is_stable_and_nonmutating():
    report = build_scaling_report(make_protocol())

    first = seal_scaling_report(report)
    second = seal_scaling_report(report)

    assert first == second
    assert "integrity" not in report
    assert len(first["integrity"]["digest"]) == 64
    assert verify_scaling_report(first) is first


def test_verifier_rejects_tampering_even_after_resealing():
    report = seal_scaling_report(build_scaling_report(make_protocol()))
    changed = deepcopy(report)
    changed["target"]["tokens"] += 1

    with pytest.raises(ScalingLawError, match="integrity"):
        verify_scaling_report(changed)

    changed = seal_scaling_report(changed)
    with pytest.raises(ScalingLawError, match="recomputation"):
        verify_scaling_report(changed)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("schema_version", True, "unsupported"),
        ("kind", "measured_scaling", "unexpected"),
        ("measured", 0, "remain unmeasured"),
    ],
)
def test_verifier_rejects_serially_distinct_contract_values(field, value, message):
    report = seal_scaling_report(build_scaling_report(make_protocol()))
    report[field] = value
    report = seal_scaling_report(report)

    with pytest.raises(ScalingLawError, match=message):
        verify_scaling_report(report)
