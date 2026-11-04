from dataclasses import replace

import pytest

from ai_journey.inference_kv import InferenceCapacityError
from ai_journey.inference_protocol import InferenceProtocol
from ai_journey.inference_speculative import SpeculativePlan


def protocol():
    gib = 1024**3
    return InferenceProtocol(
        name="day-61-7b-inference",
        parameter_count=7_000_000_000,
        layers=32,
        query_heads=32,
        kv_heads=8,
        head_dim=128,
        context_tokens=32_768,
        request_lengths=(32_768,) * 16,
        bytes_per_element=2,
        total_device_bytes=80 * gib,
        model_bytes=14 * gib,
        runtime_reserve_bytes=6 * gib,
        block_tokens=16,
        speculative=SpeculativePlan(4, 4, 5),
    )


def test_protocol_binds_model_memory_workload_paging_and_speculation():
    declared = protocol()

    assert declared.kv_layout.total_bytes == 64 * 1024**3
    assert declared.device_budget.kv_bytes == 60 * 1024**3


@pytest.mark.parametrize(
    "change,match",
    [
        ({"name": "Day 61"}, "lowercase"),
        ({"kv_heads": 3}, "divide"),
        ({"request_lengths": (32_769,)}, "fit"),
    ],
)
def test_protocol_rejects_ambiguous_or_incompatible_controls(change, match):
    with pytest.raises(InferenceCapacityError, match=match):
        replace(protocol(), **change)
