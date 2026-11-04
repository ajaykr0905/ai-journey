"""Immutable contracts for reproducible serving measurements."""

from __future__ import annotations

from dataclasses import dataclass


class ServingContractError(ValueError):
    """Raised when a serving run contract is ambiguous or unbounded."""


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ServingContractError(f"{name} must be positive")
    return value


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not value.strip() or value != value.strip():
        raise ServingContractError(f"{name} must be nonempty canonical text")
    return value


@dataclass(frozen=True)
class EngineIdentity:
    """Pin the serving engine implementation under measurement."""

    name: str
    version: str
    revision: str

    def __post_init__(self) -> None:
        for name in ("name", "version", "revision"):
            _nonempty_text(getattr(self, name), name)


@dataclass(frozen=True)
class ModelIdentity:
    """Pin model, tokenizer, and immutable revision identifiers."""

    repository: str
    revision: str
    tokenizer_revision: str
    parameter_count: int

    def __post_init__(self) -> None:
        for name in ("repository", "revision", "tokenizer_revision"):
            _nonempty_text(getattr(self, name), name)
        _positive_integer(self.parameter_count, "parameter_count")


@dataclass(frozen=True)
class HardwareManifest:
    """Declare the actual accelerator topology available to a run."""

    accelerator: str
    device_count: int
    memory_bytes_per_device: int
    interconnect: str

    def __post_init__(self) -> None:
        for name in ("accelerator", "interconnect"):
            _nonempty_text(getattr(self, name), name)
        _positive_integer(self.device_count, "device_count")
        _positive_integer(self.memory_bytes_per_device, "memory_bytes_per_device")

    @property
    def total_device_memory_bytes(self) -> int:
        return self.device_count * self.memory_bytes_per_device


@dataclass(frozen=True)
class SoftwareManifest:
    """Pin the operating system and execution stack versions."""

    operating_system: str
    python_version: str
    framework_version: str
    driver_version: str

    def __post_init__(self) -> None:
        for name in (
            "operating_system",
            "python_version",
            "framework_version",
            "driver_version",
        ):
            _nonempty_text(getattr(self, name), name)


@dataclass(frozen=True)
class RunLimits:
    """Bound wall-clock execution and deterministic trace generation."""

    max_duration_seconds: int
    random_seed: int

    def __post_init__(self) -> None:
        _positive_integer(self.max_duration_seconds, "max_duration_seconds")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ServingContractError("random_seed must be a nonnegative integer")


@dataclass(frozen=True)
class ClaimBoundary:
    """Enumerate claims supported by one declared execution kind."""

    execution_kind: str
    allowed_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.execution_kind not in {
            "analytical",
            "cpu_fixture",
            "measured_hardware",
        }:
            raise ServingContractError("execution_kind is unsupported")
        if (
            not isinstance(self.allowed_claims, tuple)
            or not self.allowed_claims
            or any(not isinstance(claim, str) for claim in self.allowed_claims)
        ):
            raise TypeError("allowed_claims must be a nonempty text tuple")
        canonical = tuple(sorted(set(self.allowed_claims)))
        if canonical != self.allowed_claims or any(
            _nonempty_text(claim, "claim") != claim for claim in self.allowed_claims
        ):
            raise ServingContractError("allowed_claims must be sorted and unique")
        if "production" in self.allowed_claims:
            raise ServingContractError("bounded lab evidence cannot claim production")


@dataclass(frozen=True)
class WorkloadShape:
    """Bound the request count, token demand, and client concurrency."""

    request_count: int
    prompt_tokens_per_request: int
    output_tokens_per_request: int
    max_concurrency: int

    def __post_init__(self) -> None:
        for name in (
            "request_count",
            "prompt_tokens_per_request",
            "output_tokens_per_request",
            "max_concurrency",
        ):
            _positive_integer(getattr(self, name), name)
        if self.max_concurrency > self.request_count:
            raise ServingContractError("max_concurrency cannot exceed request_count")

    @property
    def total_requested_tokens(self) -> int:
        """Return the bounded prompt-plus-output token demand."""

        return self.request_count * (
            self.prompt_tokens_per_request + self.output_tokens_per_request
        )


@dataclass(frozen=True)
class ServingRunContract:
    """Aggregate every immutable input required to compare serving runs."""

    protocol_version: int
    workload: WorkloadShape
    engine: EngineIdentity
    model: ModelIdentity
    hardware: HardwareManifest
    software: SoftwareManifest
    limits: RunLimits
    claims: ClaimBoundary

    def __post_init__(self) -> None:
        _positive_integer(self.protocol_version, "protocol_version")
        expected = {
            "workload": WorkloadShape,
            "engine": EngineIdentity,
            "model": ModelIdentity,
            "hardware": HardwareManifest,
            "software": SoftwareManifest,
            "limits": RunLimits,
            "claims": ClaimBoundary,
        }
        for name, expected_type in expected.items():
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")
