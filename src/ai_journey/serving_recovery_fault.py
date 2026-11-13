"""Inject one SIGKILL into the supervised server at an acknowledged boundary.

The client ledger calls ``observe`` with its acknowledged-response count while
holding its own lock, so no later acknowledgement is recorded before the kill
is issued: the fault lands exactly after the K-th acknowledged response. The
injector never accepts a pid. It asks the supervisor to signal its current,
ready and live child, which the supervisor refuses for any other target.
"""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass

from .serving_recovery_config import ServingRecoveryError, require_int
from .serving_recovery_supervisor import ServingSupervisor


@dataclass(frozen=True)
class ServingFaultInjection:
    """The killed lifetime, its boundary and the monotonic kill time."""

    lifetime_number: int
    lifetime_id: str
    pid: int
    acknowledged_responses: int
    injected_ns: int
    signal: str = "SIGKILL"


class AcknowledgedResponseFault:
    """Kill the current server lifetime once K responses are acknowledged."""

    def __init__(
        self, supervisor: ServingSupervisor, *, after_acknowledged: int
    ) -> None:
        if not isinstance(supervisor, ServingSupervisor):
            raise TypeError("supervisor must be ServingSupervisor")
        self.after_acknowledged = require_int(
            after_acknowledged, "after_acknowledged", 1, 255
        )
        self._supervisor = supervisor
        self._lock = threading.Lock()
        self._injection: ServingFaultInjection | None = None

    @property
    def injection(self) -> ServingFaultInjection | None:
        with self._lock:
            return self._injection

    def observe(self, acknowledged: int) -> ServingFaultInjection | None:
        """Inject at exactly the declared count; never inject twice."""

        require_int(acknowledged, "acknowledged", 1, 2**31 - 1)
        with self._lock:
            if self._injection is not None:
                return None
            if acknowledged < self.after_acknowledged:
                return None
            if acknowledged > self.after_acknowledged:
                raise ServingRecoveryError("the fault boundary was skipped")
            target = self._supervisor.current()
            view, injected_ns = self._supervisor.signal_current(
                target.lifetime_id, signal.SIGKILL
            )
            self._injection = ServingFaultInjection(
                lifetime_number=view.number,
                lifetime_id=view.lifetime_id,
                pid=view.pid,
                acknowledged_responses=acknowledged,
                injected_ns=injected_ns,
            )
            return self._injection
