"""Measured monotonic timing metrics for worker recovery."""

from __future__ import annotations

from dataclasses import dataclass

from .recovery_config import RecoveryExperimentError


@dataclass(frozen=True)
class RecoveryTiming:
    """Controller-observed detection, restart, and full recovery durations."""

    fault_injected_ns: int
    failure_detected_ns: int
    restart_started_ns: int
    recovery_completed_ns: int
    detection_time_ms: float
    restart_delay_ms: float
    recovery_time_ms: float


def measure_recovery_timing(
    *,
    fault_injected_ns: int,
    failure_detected_ns: int,
    restart_started_ns: int,
    recovery_completed_ns: int,
) -> RecoveryTiming:
    """Derive durations only from monotonic controller timestamps."""

    values = (
        fault_injected_ns,
        failure_detected_ns,
        restart_started_ns,
        recovery_completed_ns,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("recovery timestamps must be integers")
    if any(value < 0 for value in values):
        raise RecoveryExperimentError("recovery timestamps must be non-negative")
    if list(values) != sorted(values):
        raise RecoveryExperimentError("recovery timestamps must be monotonic")
    return RecoveryTiming(
        fault_injected_ns=fault_injected_ns,
        failure_detected_ns=failure_detected_ns,
        restart_started_ns=restart_started_ns,
        recovery_completed_ns=recovery_completed_ns,
        detection_time_ms=(failure_detected_ns - fault_injected_ns) / 1_000_000,
        restart_delay_ms=(restart_started_ns - failure_detected_ns) / 1_000_000,
        recovery_time_ms=(recovery_completed_ns - fault_injected_ns) / 1_000_000,
    )
