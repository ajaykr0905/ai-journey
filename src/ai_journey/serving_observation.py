"""Measured request lifecycle records for bounded serving experiments."""

from __future__ import annotations

from enum import Enum


class ServingObservationError(ValueError):
    """Raised when measured request evidence is ambiguous or inconsistent."""


class ServingOutcome(str, Enum):
    """Fixed-cardinality terminal outcomes for serving requests."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    OVERLOADED = "overloaded"
    CANCELLED = "cancelled"
    ERROR = "error"
