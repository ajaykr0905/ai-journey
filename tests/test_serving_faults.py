from fractions import Fraction

import pytest

from ai_journey.serving_faults import (
    FaultEvent,
    FaultKind,
    RecoveryObservation,
    ServingFaultError,
    fault_coverage_fraction,
    recovery_duration_ns,
    recovery_slo_fraction,
    require_nonoverlapping_targets,
    validate_fault_schedule,
    validate_recovery_binding,
)


def test_fault_kinds_use_a_fixed_low_cardinality_vocabulary():
    assert {kind.value for kind in FaultKind} == {
        "process_kill",
        "engine_restart",
        "network_pause",
        "device_oom",
    }


def test_fault_kind_rejects_unbounded_labels():
    with pytest.raises(ValueError):
        FaultKind("customer-specific-chaos")


def test_fault_event_has_an_explicit_bounded_interval():
    event = FaultEvent("fault-1", FaultKind.PROCESS_KILL, "worker-0", 100, 25)

    assert event.end_ns == 125


def test_fault_event_rejects_zero_duration():
    with pytest.raises(ServingFaultError, match="duration"):
        FaultEvent("fault-1", FaultKind.PROCESS_KILL, "worker-0", 100, 0)


def _fault(fault_id, start_ns, duration_ns=10, target="worker-0"):
    return FaultEvent(fault_id, FaultKind.PROCESS_KILL, target, start_ns, duration_ns)


def test_fault_schedule_accepts_tied_starts_with_unique_ids():
    schedule = (_fault("a", 0), _fault("b", 0, target="worker-1"))

    assert validate_fault_schedule(schedule) == schedule


def test_fault_schedule_rejects_duplicate_ids():
    with pytest.raises(ServingFaultError, match="unique"):
        validate_fault_schedule((_fault("a", 0), _fault("a", 10)))


def test_fault_overlap_is_checked_per_target():
    schedule = (_fault("a", 0, 10), _fault("b", 5, 10, target="worker-1"))

    assert require_nonoverlapping_targets(schedule) == schedule

    with pytest.raises(ServingFaultError, match="overlap"):
        require_nonoverlapping_targets((_fault("a", 0, 10), _fault("b", 5, 10)))


def test_fault_coverage_reports_missing_required_classes():
    events = (_fault("a", 0),)
    required = (FaultKind.PROCESS_KILL, FaultKind.ENGINE_RESTART)

    assert fault_coverage_fraction(events, required) == Fraction(1, 2)


def test_recovery_observation_requires_monotonic_milestones():
    observation = RecoveryObservation("fault-1", 110, 120, 130)

    assert observation.first_success_ns == 130

    with pytest.raises(ServingFaultError, match="monotonic"):
        RecoveryObservation("fault-1", 120, 110, 130)


def test_recovery_binding_rejects_readiness_before_fault_end():
    event = _fault("fault-1", 100, 20)

    assert (
        validate_recovery_binding(
            event, RecoveryObservation("fault-1", 105, 120, 130)
        ).ready_ns
        == 120
    )

    with pytest.raises(ServingFaultError, match="fault end"):
        validate_recovery_binding(event, RecoveryObservation("fault-1", 105, 119, 130))


def test_recovery_duration_includes_detection_and_readiness():
    event = _fault("fault-1", 100, 20)
    observation = RecoveryObservation("fault-1", 105, 120, 145)

    assert recovery_duration_ns(event, observation) == 45


def test_recovery_slo_fraction_requires_complete_matching_evidence():
    events = (_fault("a", 0, 10), _fault("b", 100, 10))
    recoveries = (
        RecoveryObservation("a", 5, 10, 20),
        RecoveryObservation("b", 105, 110, 160),
    )

    assert recovery_slo_fraction(events, recoveries, max_recovery_ns=50) == Fraction(
        1, 2
    )
