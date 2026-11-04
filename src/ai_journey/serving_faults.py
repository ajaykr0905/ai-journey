"""Deterministic fault schedules and recovery observations for serving labs."""

from __future__ import annotations

from enum import Enum


class ServingFaultError(ValueError):
    """Raised when fault or recovery evidence is ambiguous."""


class FaultKind(str, Enum):
    """Fixed fault classes supported by the serving experiment protocol."""

    PROCESS_KILL = "process_kill"
    ENGINE_RESTART = "engine_restart"
    NETWORK_PAUSE = "network_pause"
    DEVICE_OOM = "device_oom"
