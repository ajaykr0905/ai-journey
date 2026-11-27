"""Strict loader of the declared A100 and H100 inputs of the Day 82 speedup model.

Every device number in ``config/day-82-triton-tc.json`` is a declared input:
it names its source page, the words that the page states, and the status
``declared``. No number is a measurement. The H100 boost clock is the one
``not_stated`` value: the cited pages give no final H100 SXM clock.

``load_config`` refuses a missing or unknown field at every level, a bool as
an integer, any math path other than fp32, tf32, fp16 and bf16, a source
that the sources table does not name, and a URL that is not https. Two
checks tie the inputs together:

- a declared boost clock must reproduce each listed dense peak from the
  per-SM rate and the SM count, within the rounding of the listed figure,
  as the Day 71 FP32 check does: 108 x 64 x 2 x 1410 MHz is 19.49 TFLOPS,
  listed as 19.5;
- the A100 record must agree with the Day 71 pinned ``a100-sxm4-80gb``
  record: 108 SMs, 1410 MHz, 64 FP32 lanes per SM, 2039 GB/s and 19.5
  TFLOPS FP32.

A listed Tensor Core figure ``with_sparsity`` counts 2:4 structured
sparsity, which doubles the rate, so the model divides it by the declared
factor 2 to get the dense peak.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import loads_strict, read_bounded
from .kernel_launch_plan import _fields
from .matmul_model_reference import MatmulModelError
from .matmul_model_roofline import parse_decimal
from .simt_device import PINNED_DEVICES
from .triton_tc_compile import TARGETS
from .triton_tc_kernel import TritonTcError

SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
PATHS = ("fp32", "tf32", "fp16", "bf16")
STATUSES = ("declared", "not_stated")
OUTPUT_BYTES = 4
MAX_SIZE = 1 << 16
DAY71_A100 = "a100-sxm4-80gb"
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CONFIG_FIELDS = {
    "schema_version",
    "name",
    "statement",
    "sizes",
    "output_bytes",
    "sources",
    "sparsity",
    "devices",
}
_SOURCE_FIELDS = {"title", "url", "read_on"}
_DECLARED_FIELDS = {"value", "status", "source", "states"}
_DEVICE_FIELDS = {
    "key",
    "name",
    "target",
    "sm_count",
    "boost_clock_mhz",
    "memory_bandwidth_gb_per_second",
    "paths",
}
_PATH_FIELDS = {"fma_per_sm_per_clock", "listed_tflops"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TritonTcError(message)


def _text(value: object, name: str, limit: int = 400) -> str:
    _require(
        type(value) is str
        and 0 < len(value) <= limit
        and value == value.strip()
        and value.isascii()
        and value.isprintable(),
        f"{name} must be printable ASCII text of at most {limit} characters",
    )
    return value


def _object(value: object, expected: set[str], label: str) -> Mapping:
    try:
        return _fields(value, expected, label)
    except KernelLaunchError as error:
        raise TritonTcError(str(error)) from error


def _decimal(value: object, name: str) -> Fraction:
    try:
        return parse_decimal(value, name)
    except MatmulModelError as error:
        raise TritonTcError(str(error)) from error


def _count(value: object, name: str, low: int, high: int) -> int:
    _require(
        type(value) is int and low <= value <= high,
        f"{name} must be an integer in [{low}, {high}]",
    )
    return value


@dataclass(frozen=True)
class Source:
    """One public page that declared inputs cite."""

    key: str
    title: str
    url: str
    read_on: str

    def __post_init__(self) -> None:
        _require(
            type(self.key) is str and _SLUG.fullmatch(self.key) is not None,
            "a source key must be a lowercase slug",
        )
        _text(self.title, f"{self.key} title", 160)
        _text(self.url, f"{self.key} url", 200)
        _require(self.url.startswith("https://"), f"{self.key} url must be https")
        _require(
            type(self.read_on) is str and _DATE.fullmatch(self.read_on) is not None,
            f"{self.key} read_on must be a YYYY-MM-DD date",
        )

    def to_mapping(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "read_on": self.read_on}


@dataclass(frozen=True)
class Declared:
    """One declared input: its value, status, source and the words it states."""

    value: object
    status: str
    source: str
    states: str

    def __post_init__(self) -> None:
        _require(self.status in STATUSES, f"status must be one of {STATUSES}")
        _require(
            (self.value is None) == (self.status == "not_stated"),
            "a declared input needs a value, and only a not_stated one has none",
        )
        _text(self.states, "states")

    def to_mapping(self) -> dict[str, object]:
        return {
            "value": self.value,
            "status": self.status,
            "source": self.source,
            "states": self.states,
        }


@dataclass(frozen=True)
class MathPath:
    """The per-SM rate and the listed peak of one math path on one device."""

    name: str
    fma_per_sm_per_clock: Declared
    listed_tflops: Declared
    with_sparsity: bool
    sparsity_factor: int

    def __post_init__(self) -> None:
        _require(self.name in PATHS, f"path must be one of {PATHS}")
        _count(self.fma_per_sm_per_clock.value, f"{self.name} rate", 1, 1 << 16)
        _decimal(self.listed_tflops.value, f"{self.name} listed_tflops")
        _require(type(self.with_sparsity) is bool, "with_sparsity must be a bool")
        _require(
            not (self.with_sparsity and self.name == "fp32"),
            "an FP32 FMA figure has no sparsity",
        )

    @property
    def rate(self) -> int:
        return int(self.fma_per_sm_per_clock.value)

    @property
    def dense_tflops(self) -> Fraction:
        listed = Fraction(str(self.listed_tflops.value))
        return listed / self.sparsity_factor if self.with_sparsity else listed

    def to_mapping(self) -> dict[str, object]:
        listed = self.listed_tflops.to_mapping()
        return {
            "fma_per_sm_per_clock": self.fma_per_sm_per_clock.to_mapping(),
            "listed_tflops": {
                "value": listed["value"],
                "with_sparsity": self.with_sparsity,
                "status": listed["status"],
                "source": listed["source"],
                "states": listed["states"],
            },
        }


@dataclass(frozen=True)
class Device:
    """One declared GPU of the model and its four math paths."""

    key: str
    name: str
    target: str
    sm_count: Declared
    boost_clock_mhz: Declared
    memory_bandwidth: Declared
    paths: tuple[MathPath, ...]

    def __post_init__(self) -> None:
        _require(
            type(self.key) is str and _SLUG.fullmatch(self.key) is not None,
            "a device key must be a lowercase slug",
        )
        _text(self.name, "device name", 80)
        _require(self.target in TARGETS, f"target must be one of {tuple(TARGETS)}")
        _count(self.sm_count.value, f"{self.key} sm_count", 1, 4096)
        if self.boost_clock_mhz.value is not None:
            _count(self.boost_clock_mhz.value, f"{self.key} boost clock", 1, 10_000)
        _decimal(self.memory_bandwidth.value, f"{self.key} bandwidth")
        _require(
            tuple(path.name for path in self.paths) == PATHS,
            f"a device must declare the paths {PATHS} in order",
        )
        for path in self.paths:
            check = self.reproduction(path.name)
            _require(
                check is None or check["within_rounding"],
                f"{self.key} {path.name}: the declared rate, SM count and clock do "
                "not reproduce the listed peak within its rounding",
            )

    @property
    def sms(self) -> int:
        return int(self.sm_count.value)

    @property
    def bandwidth_gbps(self) -> Fraction:
        return Fraction(str(self.memory_bandwidth.value))

    def path(self, name: str) -> MathPath:
        _require(name in PATHS, f"path must be one of {PATHS}")
        return self.paths[PATHS.index(name)]

    def implied_clock_mhz(self, name: str) -> Fraction:
        """Return the clock at which the per-SM rate gives the dense peak."""

        path = self.path(name)
        return path.dense_tflops * 10**6 / (2 * path.rate * self.sms)

    def reproduction(self, name: str) -> dict[str, object] | None:
        """Compare rate x SMs x 2 x clock with the listed peak, at its rounding."""

        if self.boost_clock_mhz.value is None:
            return None
        path = self.path(name)
        derived = 2 * path.rate * self.sms * int(self.boost_clock_mhz.value) * 10**6
        published = path.dense_tflops * 10**12
        places = max(0, -Decimal(str(path.listed_tflops.value)).as_tuple().exponent)
        half = Fraction(5 * 10 ** (11 - places))
        if path.with_sparsity:
            half /= path.sparsity_factor
        return {
            "derived_flops_per_second": derived,
            "published_flops_per_second": int(published),
            "half_unit_flops_per_second": int(half),
            "within_rounding": abs(derived - published) <= half,
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "target": self.target,
            "sm_count": self.sm_count.to_mapping(),
            "boost_clock_mhz": self.boost_clock_mhz.to_mapping(),
            "memory_bandwidth_gb_per_second": self.memory_bandwidth.to_mapping(),
            "paths": {path.name: path.to_mapping() for path in self.paths},
        }


@dataclass(frozen=True)
class TcModelConfig:
    """Every declared input of the speedup model and the sizes that it covers."""

    name: str
    statement: str
    sizes: tuple[int, ...]
    output_bytes: int
    sources: tuple[Source, ...]
    sparsity: Declared
    devices: tuple[Device, ...]

    def __post_init__(self) -> None:
        _require(
            type(self.name) is str and _SLUG.fullmatch(self.name) is not None,
            "config name must be a lowercase slug",
        )
        _text(self.statement, "statement")
        _require(
            type(self.sizes) is tuple
            and 1 <= len(self.sizes) <= 16
            and all(type(size) is int and 1 <= size <= MAX_SIZE for size in self.sizes)
            and list(self.sizes) == sorted(set(self.sizes)),
            f"sizes must be 1 to 16 increasing integers in [1, {MAX_SIZE}]",
        )
        _require(
            self.output_bytes == OUTPUT_BYTES and type(self.output_bytes) is int,
            "output_bytes must be 4: the pinned kernel stores float32 C",
        )
        keys = [source.key for source in self.sources]
        _require(len(set(keys)) == len(keys) and keys, "sources must not repeat")
        _count(self.sparsity.value, "sparsity factor", 2, 2)
        cited = [self.sparsity.source]
        for device in self.devices:
            cited += [
                device.sm_count.source,
                device.boost_clock_mhz.source,
                device.memory_bandwidth.source,
            ]
            for path in device.paths:
                cited += [path.fma_per_sm_per_clock.source, path.listed_tflops.source]
        missing = sorted(set(cited) - set(keys))
        _require(not missing, f"inputs cite sources that the table lacks: {missing}")
        _require(
            [device.target for device in self.devices] == list(TARGETS),
            f"devices must declare one GPU for each of {tuple(TARGETS)}, in order",
        )
        for device in self.devices:
            if device.key == DAY71_A100:
                _check_day71(device)

    def device(self, target: str) -> Device:
        for device in self.devices:
            if device.target == target:
                return device
        raise TritonTcError(f"the config declares no device for {target}")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "statement": self.statement,
            "sizes": list(self.sizes),
            "output_bytes": self.output_bytes,
            "sources": {source.key: source.to_mapping() for source in self.sources},
            "sparsity": {
                "factor": self.sparsity.value,
                "status": self.sparsity.status,
                "source": self.sparsity.source,
                "states": self.sparsity.states,
            },
            "devices": [device.to_mapping() for device in self.devices],
        }


def _check_day71(device: Device) -> None:
    """Hold the A100 record to the Day 71 pinned device of the same key."""

    pinned = PINNED_DEVICES[DAY71_A100]
    fp32 = device.path("fp32")
    _require(
        (
            device.sms,
            device.boost_clock_mhz.value,
            device.bandwidth_gbps,
            fp32.rate,
            Fraction(str(fp32.listed_tflops.value)),
        )
        == (
            pinned.sm_count,
            pinned.boost_clock_mhz,
            Fraction(pinned.dram_bandwidth_gb_per_second),
            pinned.architecture.fp32_lanes_per_sm,
            Fraction(pinned.datasheet_fp32_tflops),
        ),
        f"{DAY71_A100} must agree with the Day 71 pinned device record",
    )


def _declared(value: object, label: str) -> Declared:
    item = _object(value, _DECLARED_FIELDS, label)
    _require(type(item["source"]) is str, f"{label} source must be a source key")
    return Declared(item["value"], item["status"], item["source"], item["states"])


def _path(name: str, value: object, factor: int, label: str) -> MathPath:
    item = _object(value, _PATH_FIELDS, label)
    listed = _object(
        item["listed_tflops"], _DECLARED_FIELDS | {"with_sparsity"}, f"{label} peak"
    )
    return MathPath(
        name=name,
        fma_per_sm_per_clock=_declared(item["fma_per_sm_per_clock"], f"{label} rate"),
        listed_tflops=_declared(
            {key: listed[key] for key in _DECLARED_FIELDS}, f"{label} peak"
        ),
        with_sparsity=listed["with_sparsity"],
        sparsity_factor=factor,
    )


def parse_config(value: object) -> TcModelConfig:
    """Build the config from decoded JSON, refusing every unknown field."""

    config = _object(value, _CONFIG_FIELDS, "config")
    _require(
        type(config["schema_version"]) is int
        and config["schema_version"] == SCHEMA_VERSION,
        "unsupported config schema_version",
    )
    _require(type(config["sizes"]) is list, "sizes must be a JSON array")
    sources = config["sources"]
    _require(isinstance(sources, Mapping) and sources, "sources must be an object")
    sparsity = _object(
        config["sparsity"], {"factor", "status", "source", "states"}, "sparsity"
    )
    factor = sparsity["factor"]
    _count(factor, "sparsity factor", 2, 2)
    devices = config["devices"]
    _require(type(devices) is list, "devices must be a JSON array")
    parsed = []
    for item in devices:
        device = _object(item, _DEVICE_FIELDS, "device")
        paths = _object(device["paths"], set(PATHS), "paths")
        parsed.append(
            Device(
                key=device["key"],
                name=device["name"],
                target=device["target"],
                sm_count=_declared(device["sm_count"], "sm_count"),
                boost_clock_mhz=_declared(device["boost_clock_mhz"], "boost clock"),
                memory_bandwidth=_declared(
                    device["memory_bandwidth_gb_per_second"], "bandwidth"
                ),
                paths=tuple(
                    _path(name, paths[name], factor, f"{device['key']} {name}")
                    for name in PATHS
                ),
            )
        )
    return TcModelConfig(
        name=config["name"],
        statement=config["statement"],
        sizes=tuple(config["sizes"]),
        output_bytes=config["output_bytes"],
        sources=tuple(
            Source(key, **_object(item, _SOURCE_FIELDS, f"source {key}"))
            for key, item in sources.items()
        ),
        sparsity=Declared(
            sparsity["factor"],
            sparsity["status"],
            sparsity["source"],
            sparsity["states"],
        ),
        devices=tuple(parsed),
    )


def read_config(path: Path) -> tuple[bytes, TcModelConfig]:
    """Read one bounded, duplicate-free config file; return its bytes and model."""

    try:
        payload = read_bounded(path, limit=MAX_CONFIG_BYTES, label="config")
        decoded = loads_strict(payload, label="config")
    except KernelLaunchError as error:
        raise TritonTcError(str(error)) from error
    try:
        return payload, parse_config(decoded)
    except TypeError as error:
        raise TritonTcError(f"invalid config: {error}") from error


def load_config(path: Path) -> TcModelConfig:
    return read_config(path)[1]


def config_sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    return sha256(payload).hexdigest()
