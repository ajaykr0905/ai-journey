import pytest

from ai_journey.recovery_metrics import measure_recovery_timing


def test_timing_uses_monotonic_boundaries_without_rounding_inputs():
    timing = measure_recovery_timing(
        fault_injected_ns=1_000_000,
        failure_detected_ns=2_250_000,
        restart_started_ns=3_500_000,
        recovery_completed_ns=8_750_000,
    )

    assert timing.detection_time_ms == 1.25
    assert timing.restart_delay_ms == 1.25
    assert timing.recovery_time_ms == 7.75
    assert timing.recovery_completed_ns == 8_750_000


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((True, 2, 3, 4), "integers"),
        ((-1, 2, 3, 4), "non-negative"),
        ((1, 3, 2, 4), "monotonic"),
        ((2, 1, 3, 4), "monotonic"),
    ],
)
def test_timing_rejects_ambiguous_or_backward_boundaries(values, message):
    with pytest.raises((TypeError, ValueError), match=message):
        measure_recovery_timing(
            fault_injected_ns=values[0],
            failure_detected_ns=values[1],
            restart_started_ns=values[2],
            recovery_completed_ns=values[3],
        )
