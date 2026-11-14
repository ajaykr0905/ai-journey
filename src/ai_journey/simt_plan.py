"""Strict declared-input plans for the analytical SIMT execution-model report.

A plan names the pinned architectures and devices it uses, copies their
declared values (which must equal the pinned catalog entries exactly), and
declares every scenario input with a ``basis`` sentence that labels it. Unknown
or missing fields are rejected at every level.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
import re

from .resource_accounting import DTYPE_BYTES
from .simt_device import (
    PINNED_ARCHITECTURES,
    PINNED_DEVICES,
    SimtArchitecture,
    SimtDevice,
    SimtError,
    require_int,
    require_sources,
    require_text,
)
from .simt_json import loads_strict, read_bounded
from .simt_latency import ACCESS_WIDTHS_BYTES, ARITHMETIC_PIPELINES

MAX_SIMT_PLAN_BYTES = 64 * 1024
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_MAX_ITEMS = 16


def _slug(value: object, name: str) -> str:
    if type(value) is not str or _SLUG.fullmatch(value) is None:
        raise SimtError(f"{name} must be a lowercase slug")
    return value


def _keys(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= _MAX_ITEMS:
        raise SimtError(f"{name} must be a tuple of one to {_MAX_ITEMS} entries")
    if len(set(value)) != len(value):
        raise SimtError(f"{name} must not repeat an entry")
    return value


def _increasing(
    value: object, name: str, *, minimum: int, maximum: int
) -> tuple[int, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= _MAX_ITEMS:
        raise SimtError(f"{name} must be a tuple of one to {_MAX_ITEMS} entries")
    for item in value:
        require_int(item, name, minimum=minimum, maximum=maximum)
    if any(left >= right for left, right in zip(value, value[1:])):
        raise SimtError(f"{name} must be strictly increasing")
    return value


def _basis(value: object) -> str:
    return require_text(value, "basis", maximum=600)


@dataclass(frozen=True)
class VectorAddScenario:
    """``c[i] = a[i] + b[i]`` over a 1D grid of equal blocks."""

    elements: int
    dtype: str
    threads_per_block: int
    registers_per_thread: int
    devices: tuple[str, ...]
    basis: str

    def __post_init__(self) -> None:
        require_int(self.elements, "elements", minimum=1, maximum=2**40)
        if type(self.dtype) is not str or self.dtype not in DTYPE_BYTES:
            raise SimtError("dtype must be a supported tensor dtype")
        require_int(
            self.threads_per_block, "threads_per_block", minimum=1, maximum=1024
        )
        require_int(
            self.registers_per_thread, "registers_per_thread", minimum=1, maximum=255
        )
        _keys(self.devices, "vector_add devices")
        _basis(self.basis)


@dataclass(frozen=True)
class GrayscaleScenario:
    """RGB-to-grayscale over a row-major image with a 2D grid of 2D blocks."""

    width: int
    height: int
    block_x: int
    block_y: int
    input_bytes_per_pixel: int
    output_bytes_per_pixel: int
    flops_per_pixel: int
    registers_per_thread: int
    devices: tuple[str, ...]
    basis: str

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            require_int(getattr(self, name), name, minimum=1, maximum=2**20)
        for name in ("block_x", "block_y"):
            require_int(getattr(self, name), name, minimum=1, maximum=1024)
        for name in ("input_bytes_per_pixel", "output_bytes_per_pixel"):
            require_int(getattr(self, name), name, minimum=1, maximum=64)
        require_int(self.flops_per_pixel, "flops_per_pixel", minimum=1, maximum=1024)
        require_int(
            self.registers_per_thread, "registers_per_thread", minimum=1, maximum=255
        )
        _keys(self.devices, "rgb_to_grayscale devices")
        _basis(self.basis)


@dataclass(frozen=True)
class OccupancySweepScenario:
    """Theoretical occupancy over every block size and register count pair."""

    devices: tuple[str, ...]
    threads_per_block: tuple[int, ...]
    registers_per_thread: tuple[int, ...]
    static_shared_bytes: int
    dynamic_shared_bytes: int
    basis: str

    def __post_init__(self) -> None:
        _keys(self.devices, "occupancy_sweep devices")
        _increasing(
            self.threads_per_block, "threads_per_block", minimum=1, maximum=1024
        )
        _increasing(
            self.registers_per_thread, "registers_per_thread", minimum=1, maximum=255
        )
        for name in ("static_shared_bytes", "dynamic_shared_bytes"):
            require_int(getattr(self, name), name, minimum=0, maximum=48 * 1024)
        _basis(self.basis)


@dataclass(frozen=True)
class AccessPatternSpec:
    """A named per-warp memory access: width x active lanes x independent loads."""

    name: str
    bytes_per_lane: int
    active_lanes: int
    independent_loads: int

    def __post_init__(self) -> None:
        _slug(self.name, "access pattern name")
        if (
            type(self.bytes_per_lane) is not int
            or self.bytes_per_lane not in ACCESS_WIDTHS_BYTES
        ):
            raise SimtError("bytes_per_lane must be a 1, 2, 4, 8, or 16 byte access")
        require_int(self.active_lanes, "active_lanes", minimum=1, maximum=32)
        require_int(self.independent_loads, "independent_loads", minimum=1, maximum=64)


@dataclass(frozen=True)
class MemoryLatencyScenario:
    """Little's-law rows for each device, declared latency, and access pattern."""

    devices: tuple[str, ...]
    latency_cycles: tuple[int, ...]
    access_patterns: tuple[AccessPatternSpec, ...]
    basis: str

    def __post_init__(self) -> None:
        _keys(self.devices, "memory_latency devices")
        _increasing(self.latency_cycles, "latency_cycles", minimum=1, maximum=100_000)
        if (
            type(self.access_patterns) is not tuple
            or not 1 <= len(self.access_patterns) <= _MAX_ITEMS
            or any(
                not isinstance(pattern, AccessPatternSpec)
                for pattern in self.access_patterns
            )
        ):
            raise SimtError("access_patterns must be a tuple of AccessPatternSpec")
        _keys(
            tuple(pattern.name for pattern in self.access_patterns),
            "access pattern names",
        )
        _basis(self.basis)


@dataclass(frozen=True)
class ArithmeticLatencyScenario:
    """Pipeline latency-hiding rows for architectures with a stated latency."""

    architectures: tuple[str, ...]
    pipelines: tuple[str, ...]
    latency_cycles: int
    independent_instructions: tuple[int, ...]
    basis: str

    def __post_init__(self) -> None:
        _keys(self.architectures, "arithmetic_latency architectures")
        _keys(self.pipelines, "arithmetic_latency pipelines")
        if any(pipeline not in ARITHMETIC_PIPELINES for pipeline in self.pipelines):
            raise SimtError(f"pipelines must be drawn from {ARITHMETIC_PIPELINES}")
        require_int(self.latency_cycles, "latency_cycles", minimum=1, maximum=10_000)
        _increasing(
            self.independent_instructions,
            "independent_instructions",
            minimum=1,
            maximum=64,
        )
        _basis(self.basis)


@dataclass(frozen=True)
class WaveScenario:
    """A fixed grid whose blocks are quantized into waves on each device."""

    total_blocks: int
    threads_per_block: int
    registers_per_thread: int
    devices: tuple[str, ...]
    basis: str

    def __post_init__(self) -> None:
        require_int(self.total_blocks, "total_blocks", minimum=1, maximum=2**40)
        require_int(
            self.threads_per_block, "threads_per_block", minimum=1, maximum=1024
        )
        require_int(
            self.registers_per_thread, "registers_per_thread", minimum=1, maximum=255
        )
        _keys(self.devices, "wave_quantization devices")
        _basis(self.basis)


_SCENARIOS = {
    "vector_add": VectorAddScenario,
    "rgb_to_grayscale": GrayscaleScenario,
    "occupancy_sweep": OccupancySweepScenario,
    "memory_latency": MemoryLatencyScenario,
    "arithmetic_latency": ArithmeticLatencyScenario,
    "wave_quantization": WaveScenario,
}
_PLAN_FIELDS = {
    "schema_version",
    "name",
    "statement",
    "architectures",
    "devices",
    *_SCENARIOS,
    "sources",
}


@dataclass(frozen=True)
class SimtPlan:
    """Every declared input of one analytical SIMT execution-model report."""

    name: str
    statement: str
    architectures: tuple[SimtArchitecture, ...]
    devices: tuple[SimtDevice, ...]
    vector_add: VectorAddScenario
    rgb_to_grayscale: GrayscaleScenario
    occupancy_sweep: OccupancySweepScenario
    memory_latency: MemoryLatencyScenario
    arithmetic_latency: ArithmeticLatencyScenario
    wave_quantization: WaveScenario
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        _slug(self.name, "plan name")
        require_text(self.statement, "statement", maximum=600)
        if (
            type(self.architectures) is not tuple
            or not self.architectures
            or any(
                not isinstance(item, SimtArchitecture) for item in self.architectures
            )
        ):
            raise SimtError(
                "architectures must be a nonempty tuple of SimtArchitecture"
            )
        if (
            type(self.devices) is not tuple
            or not self.devices
            or any(not isinstance(item, SimtDevice) for item in self.devices)
        ):
            raise SimtError("devices must be a nonempty tuple of SimtDevice")
        capabilities = _keys(
            tuple(item.compute_capability for item in self.architectures),
            "architecture compute capabilities",
        )
        keys = _keys(tuple(item.key for item in self.devices), "device keys")
        for architecture in self.architectures:
            if (
                PINNED_ARCHITECTURES.get(architecture.compute_capability)
                != architecture
            ):
                raise SimtError(
                    f"architecture {architecture.compute_capability} must equal the "
                    "pinned source-backed record"
                )
        for device in self.devices:
            if PINNED_DEVICES.get(device.key) != device:
                raise SimtError(
                    f"device {device.key} must equal the pinned source-backed record"
                )
            if device.compute_capability not in capabilities:
                raise SimtError(f"device {device.key} needs a declared architecture")
        for name, cls in _SCENARIOS.items():
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be {cls.__name__}")
        for name in (
            "vector_add",
            "rgb_to_grayscale",
            "occupancy_sweep",
            "memory_latency",
            "wave_quantization",
        ):
            unknown = sorted(set(getattr(self, name).devices) - set(keys))
            if unknown:
                raise SimtError(f"{name} names undeclared devices: {unknown}")
        unknown = sorted(set(self.arithmetic_latency.architectures) - set(capabilities))
        if unknown:
            raise SimtError(
                f"arithmetic_latency names undeclared architectures: {unknown}"
            )
        require_sources(self.sources, "plan sources")

    def device(self, key: str) -> SimtDevice:
        for device in self.devices:
            if device.key == key:
                return device
        raise SimtError(f"device {key!r} is not declared in the plan")

    def architecture(self, capability: str) -> SimtArchitecture:
        for item in self.architectures:
            if item.compute_capability == capability:
                return item
        raise SimtError(f"architecture {capability!r} is not declared in the plan")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical JSON-safe declared-input mapping."""

        mapping: dict[str, object] = {
            "schema_version": 1,
            "name": self.name,
            "statement": self.statement,
            "architectures": [item.to_mapping() for item in self.architectures],
            "devices": [item.to_mapping() for item in self.devices],
            "sources": list(self.sources),
        }
        for name in _SCENARIOS:
            mapping[name] = _scenario_mapping(getattr(self, name))
        return mapping

    @classmethod
    def from_mapping(cls, mapping: object) -> SimtPlan:
        """Parse a declared-input mapping without ignoring any field."""

        if not isinstance(mapping, Mapping):
            raise SimtError("SIMT plan must be an object")
        if set(mapping) != _PLAN_FIELDS:
            missing = sorted(_PLAN_FIELDS - set(mapping))
            extra = sorted(str(key) for key in set(mapping) - _PLAN_FIELDS)
            raise SimtError(
                f"SIMT plan fields mismatch: missing={missing}, extra={extra}"
            )
        if type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1:
            raise SimtError("unsupported SIMT plan schema_version")
        architectures = _records(mapping["architectures"], "architectures")
        parsed_architectures = tuple(
            SimtArchitecture.from_mapping(item) for item in architectures
        )
        by_capability = {item.compute_capability: item for item in parsed_architectures}
        devices = tuple(
            SimtDevice.from_mapping(item, by_capability)
            for item in _records(mapping["devices"], "devices")
        )
        scenarios = {
            name: _scenario_from_mapping(scenario, mapping[name], name)
            for name, scenario in _SCENARIOS.items()
        }
        return cls(
            name=mapping["name"],
            statement=mapping["statement"],
            architectures=parsed_architectures,
            devices=devices,
            sources=_array(mapping["sources"], "sources"),
            **scenarios,
        )


def _array(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise SimtError(f"{label} must be a JSON array")
    return tuple(value)


def _records(value: object, label: str) -> tuple[object, ...]:
    records = _array(value, label)
    if not 1 <= len(records) <= _MAX_ITEMS:
        raise SimtError(f"{label} must hold one to {_MAX_ITEMS} records")
    return records


def _scenario_mapping(scenario: object) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for field in fields(scenario):
        value = getattr(scenario, field.name)
        if type(value) is tuple:
            value = [
                _scenario_mapping(item) if isinstance(item, AccessPatternSpec) else item
                for item in value
            ]
        mapping[field.name] = value
    return mapping


def _scenario_from_mapping(cls: type, value: object, label: str) -> object:
    if not isinstance(value, Mapping):
        raise SimtError(f"{label} must be an object")
    expected = {field.name for field in fields(cls)}
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(str(key) for key in set(value) - expected)
        raise SimtError(f"{label} fields mismatch: missing={missing}, extra={extra}")
    values = dict(value)
    for field in fields(cls):
        if type(values[field.name]) is list:
            values[field.name] = _array(values[field.name], field.name)
    if cls is MemoryLatencyScenario:
        values["access_patterns"] = tuple(
            _scenario_from_mapping(AccessPatternSpec, item, "access pattern")
            for item in values["access_patterns"]
        )
    return cls(**values)


def load_simt_plan(path: Path) -> SimtPlan:
    """Load one bounded, duplicate-free, exact-field SIMT plan file."""

    payload = read_bounded(path, limit=MAX_SIMT_PLAN_BYTES, label="SIMT plan")
    try:
        return SimtPlan.from_mapping(loads_strict(payload, "SIMT plan"))
    except TypeError as error:
        raise SimtError(f"invalid SIMT plan: {error}") from error
