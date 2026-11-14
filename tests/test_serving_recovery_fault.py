import signal

import pytest
from test_serving_recovery_supervisor import NO_RESTART, observed, supervisor

from ai_journey.serving_recovery_fault import AcknowledgedResponseFault


def test_fault_kills_the_live_lifetime_exactly_at_the_declared_boundary(tmp_path):
    with supervisor(tmp_path, **NO_RESTART) as owner:
        target = owner.start()
        fault = AcknowledgedResponseFault(owner, after_acknowledged=3)

        assert fault.observe(1) is None
        assert fault.observe(2) is None
        assert owner.current().exit_code is None
        injection = fault.observe(3)
        assert owner.wait_until(
            lambda: observed(owner, "exit_observed"), timeout_ms=5000
        )
        assert fault.observe(3) is None
        assert fault.observe(4) is None

    assert injection == fault.injection
    assert (injection.lifetime_id, injection.pid) == (target.lifetime_id, target.pid)
    assert (injection.lifetime_number, injection.acknowledged_responses) == (1, 3)
    assert injection.signal == "SIGKILL"
    exit_observed = owner.observations()[2]
    assert exit_observed.exit_code == -signal.SIGKILL
    assert exit_observed.monotonic_ns >= injection.injected_ns


def test_fault_refuses_a_target_that_is_not_live_and_ready(tmp_path):
    with supervisor(tmp_path, **NO_RESTART) as owner:
        view = owner.start()
        owner.signal_current(view.lifetime_id, signal.SIGKILL)
        assert owner.wait_until(
            lambda: observed(owner, "exit_observed"), timeout_ms=5000
        )
        fault = AcknowledgedResponseFault(owner, after_acknowledged=1)
        with pytest.raises(ValueError, match="not ready"):
            fault.observe(1)
        assert fault.injection is None


def test_fault_rejects_skipped_boundaries_and_invalid_controls(tmp_path):
    with supervisor(tmp_path) as owner:
        fault = AcknowledgedResponseFault(owner, after_acknowledged=2)
        with pytest.raises(ValueError, match="skipped"):
            fault.observe(3)
        with pytest.raises(TypeError, match="integer"):
            fault.observe(True)
        with pytest.raises(TypeError, match="integer"):
            AcknowledgedResponseFault(owner, after_acknowledged=True)
        with pytest.raises(TypeError, match="ServingSupervisor"):
            AcknowledgedResponseFault(object(), after_acknowledged=2)
        assert owner.observations() == ()
