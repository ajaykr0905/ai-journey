"""Declared SIMT architecture and product contracts for analytical CUDA planning.

Every number in this module is a declared input copied from a public NVIDIA
source named in the record's ``sources``. Nothing here detects, queries, or
measures a GPU; the records only describe what the sources state. H100 SXM5
is deliberately absent: Table 3 of NVIDIA's "NVIDIA Hopper Architecture
In-Depth" technical blog lists its boost clock as not finalized and the H100
product page lists no clock, so its derived FP32 peak cannot be checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from decimal import Decimal
import re
from types import MappingProxyType


class SimtError(ValueError):
    """Raised when a SIMT execution-model contract is invalid."""


_CAPABILITY = re.compile(r"[1-9][0-9]?\.[0-9]")
_KEY = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_TFLOPS = re.compile(r"(?:0|[1-9][0-9]{0,5})(?:\.[0-9]{1,3})?")
_FLOPS_PER_TFLOP = 10**12
_HZ_PER_MHZ = 10**6
_BYTES_PER_GB = 10**9
_KIB = 1024


def require_int(value: object, name: str, *, minimum: int, maximum: int) -> int:
    """Return an exact integer inside explicit bounds; booleans are rejected."""

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise SimtError(f"{name} must be between {minimum} and {maximum}")
    return value


def require_text(value: object, name: str, *, maximum: int = 240) -> str:
    """Return canonical printable ASCII text without surrounding whitespace."""

    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if (
        not value
        or len(value) > maximum
        or value != value.strip()
        or not value.isascii()
        or not value.isprintable()
    ):
        raise SimtError(f"{name} must be canonical printable ASCII text")
    return value


def require_sources(value: object, name: str) -> tuple[str, ...]:
    """Return a nonempty tuple of unique citation strings."""

    if type(value) is not tuple or not 1 <= len(value) <= 8:
        raise SimtError(f"{name} must be a tuple of one to eight citations")
    for item in value:
        require_text(item, name, maximum=400)
    if len(set(value)) != len(value):
        raise SimtError(f"{name} must not repeat a citation")
    return value


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _exact_fields(cls: type, mapping: object, label: str) -> dict[str, object]:
    if not isinstance(mapping, Mapping):
        raise SimtError(f"{label} must be an object")
    expected = {field.name for field in fields(cls)}
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(str(key) for key in set(mapping) - expected)
        raise SimtError(f"{label} fields mismatch: missing={missing}, extra={extra}")
    return dict(mapping)


def _tuple_of(value: object, label: str) -> tuple[object, ...]:
    if type(value) not in (list, tuple):
        raise SimtError(f"{label} must be an array")
    return tuple(value)


@dataclass(frozen=True)
class SimtArchitecture:
    """Per-compute-capability limits that govern launch geometry and residency."""

    compute_capability: str
    warp_size: int
    max_threads_per_block: int
    max_block_dim_x: int
    max_block_dim_y: int
    max_block_dim_z: int
    max_grid_dim_x: int
    max_grid_dim_y: int
    max_grid_dim_z: int
    max_threads_per_sm: int
    max_warps_per_sm: int
    max_blocks_per_sm: int
    registers_per_sm: int
    registers_per_block: int
    max_registers_per_thread: int
    register_allocation_unit: int
    register_allocation_granularity: str
    register_sub_partitions_per_sm: int
    warp_schedulers_per_sm: int
    fp32_lanes_per_sm: int
    shared_memory_per_sm_bytes: int
    shared_memory_carveouts_bytes: tuple[int, ...]
    shared_memory_per_block_bytes: int
    shared_memory_per_block_optin_bytes: int
    shared_memory_allocation_unit_bytes: int
    reserved_shared_memory_per_block_bytes: int
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.compute_capability) is not str
            or _CAPABILITY.fullmatch(self.compute_capability) is None
        ):
            raise SimtError("compute_capability must look like '8.0'")
        require_int(self.warp_size, "warp_size", minimum=1, maximum=64)
        for name, maximum in (
            ("max_threads_per_block", 4096),
            ("max_block_dim_x", 4096),
            ("max_block_dim_y", 4096),
            ("max_block_dim_z", 4096),
            ("max_grid_dim_x", 2**31 - 1),
            ("max_grid_dim_y", 2**31 - 1),
            ("max_grid_dim_z", 2**31 - 1),
            ("max_threads_per_sm", 8192),
            ("max_warps_per_sm", 256),
            ("max_blocks_per_sm", 256),
            ("registers_per_sm", 2**20),
            ("registers_per_block", 2**20),
            ("max_registers_per_thread", 1024),
            ("register_allocation_unit", 4096),
            ("register_sub_partitions_per_sm", 16),
            ("warp_schedulers_per_sm", 16),
            ("fp32_lanes_per_sm", 1024),
            ("shared_memory_per_sm_bytes", 4 * 2**20),
            ("shared_memory_per_block_bytes", 4 * 2**20),
            ("shared_memory_per_block_optin_bytes", 4 * 2**20),
            ("shared_memory_allocation_unit_bytes", 4096),
        ):
            require_int(getattr(self, name), name, minimum=1, maximum=maximum)
        require_int(
            self.reserved_shared_memory_per_block_bytes,
            "reserved_shared_memory_per_block_bytes",
            minimum=0,
            maximum=64 * _KIB,
        )
        for name in (
            "warp_size",
            "register_allocation_unit",
            "shared_memory_allocation_unit_bytes",
        ):
            if not _power_of_two(getattr(self, name)):
                raise SimtError(f"{name} must be a power of two")
        if self.max_threads_per_sm != self.max_warps_per_sm * self.warp_size:
            raise SimtError(
                "max_threads_per_sm must equal max_warps_per_sm * warp_size"
            )
        if (
            self.max_threads_per_block % self.warp_size
            or self.max_threads_per_block > self.max_threads_per_sm
        ):
            raise SimtError(
                "max_threads_per_block must be a warp multiple that fits one SM"
            )
        if any(
            dimension > self.max_threads_per_block
            for dimension in (
                self.max_block_dim_x,
                self.max_block_dim_y,
                self.max_block_dim_z,
            )
        ):
            raise SimtError(
                "block dimension limits cannot exceed max_threads_per_block"
            )
        if self.registers_per_block > self.registers_per_sm:
            raise SimtError("registers_per_block cannot exceed registers_per_sm")
        if self.registers_per_sm % (
            self.register_sub_partitions_per_sm * self.register_allocation_unit
        ):
            raise SimtError(
                "registers_per_sm must split into whole allocation units per "
                "sub-partition"
            )
        if self.register_allocation_granularity != "warp":
            raise SimtError("only warp register allocation granularity is modeled")
        self._validate_shared_memory()
        require_sources(self.sources, "architecture sources")

    def _validate_shared_memory(self) -> None:
        carveouts = self.shared_memory_carveouts_bytes
        if type(carveouts) is not tuple or not 1 <= len(carveouts) <= 32:
            raise SimtError("shared_memory_carveouts_bytes must be a nonempty tuple")
        for carveout in carveouts:
            require_int(
                carveout,
                "shared memory carveout",
                minimum=0,
                maximum=self.shared_memory_per_sm_bytes,
            )
            if carveout % self.shared_memory_allocation_unit_bytes:
                raise SimtError(
                    "shared memory carveouts must be whole allocation units"
                )
        if any(left >= right for left, right in zip(carveouts, carveouts[1:])):
            raise SimtError("shared memory carveouts must be strictly increasing")
        if carveouts[-1] != self.shared_memory_per_sm_bytes:
            raise SimtError(
                "the largest carveout must equal shared_memory_per_sm_bytes"
            )
        if (
            self.shared_memory_per_block_bytes
            > self.shared_memory_per_block_optin_bytes
        ):
            raise SimtError("default per-block shared memory cannot exceed opt-in")
        if (
            self.shared_memory_per_block_optin_bytes
            + self.reserved_shared_memory_per_block_bytes
            > self.shared_memory_per_sm_bytes
        ):
            raise SimtError("one opt-in block and its reservation must fit one SM")
        if (
            self.reserved_shared_memory_per_block_bytes
            % self.shared_memory_allocation_unit_bytes
        ):
            raise SimtError("reserved shared memory must be whole allocation units")

    @property
    def major(self) -> int:
        """Return the compute capability major version."""

        return int(self.compute_capability.split(".", 1)[0])

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-safe mapping with tuples as arrays."""

        mapping: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            mapping[field.name] = list(value) if type(value) is tuple else value
        return mapping

    @classmethod
    def from_mapping(cls, mapping: object) -> SimtArchitecture:
        """Parse an architecture without ignoring unknown or missing fields."""

        values = _exact_fields(cls, mapping, "architecture")
        for name in ("shared_memory_carveouts_bytes", "sources"):
            values[name] = _tuple_of(values[name], name)
        return cls(**values)


@dataclass(frozen=True)
class Fp32DatasheetCheck:
    """Exact comparison of a derived FP32 peak with a rounded datasheet figure."""

    published_tflops: str
    published_flops_per_second: int
    rounding_half_unit_flops_per_second: int
    derived_flops_per_second: int

    @property
    def absolute_difference_flops_per_second(self) -> int:
        return abs(self.derived_flops_per_second - self.published_flops_per_second)

    @property
    def within_published_rounding(self) -> bool:
        return (
            self.absolute_difference_flops_per_second
            <= self.rounding_half_unit_flops_per_second
        )


@dataclass(frozen=True)
class SimtDevice:
    """One declared product: an architecture plus SM count, clock, and DRAM."""

    key: str
    name: str
    architecture: SimtArchitecture
    sm_count: int
    boost_clock_mhz: int
    dram_bandwidth_gb_per_second: int
    datasheet_fp32_tflops: str
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.key) is not str or _KEY.fullmatch(self.key) is None:
            raise SimtError("device key must be a lowercase slug")
        require_text(self.name, "device name", maximum=80)
        if not isinstance(self.architecture, SimtArchitecture):
            raise TypeError("architecture must be SimtArchitecture")
        require_int(self.sm_count, "sm_count", minimum=1, maximum=4096)
        require_int(self.boost_clock_mhz, "boost_clock_mhz", minimum=1, maximum=10_000)
        require_int(
            self.dram_bandwidth_gb_per_second,
            "dram_bandwidth_gb_per_second",
            minimum=1,
            maximum=1_000_000,
        )
        if (
            type(self.datasheet_fp32_tflops) is not str
            or _TFLOPS.fullmatch(self.datasheet_fp32_tflops) is None
        ):
            raise SimtError("datasheet_fp32_tflops must be a canonical decimal string")
        require_sources(self.sources, "device sources")
        if not self.fp32_datasheet_check().within_published_rounding:
            raise SimtError(
                "declared SMs, FP32 lanes, and boost clock do not reproduce the "
                "datasheet FP32 peak within its published rounding"
            )

    @property
    def compute_capability(self) -> str:
        return self.architecture.compute_capability

    @property
    def boost_clock_hz(self) -> int:
        return self.boost_clock_mhz * _HZ_PER_MHZ

    @property
    def dram_bandwidth_bytes_per_second(self) -> int:
        """Return declared decimal-GB/s bandwidth as bytes per second."""

        return self.dram_bandwidth_gb_per_second * _BYTES_PER_GB

    @property
    def peak_fp32_flops_per_second(self) -> int:
        """Return SMs x FP32 lanes x 2 FLOPs per FMA x boost clock."""

        return (
            self.sm_count
            * self.architecture.fp32_lanes_per_sm
            * 2
            * self.boost_clock_hz
        )

    def fp32_datasheet_check(self) -> Fp32DatasheetCheck:
        """Compare the derived peak with the published figure at its precision."""

        published = Decimal(self.datasheet_fp32_tflops)
        decimals = max(0, -published.as_tuple().exponent)
        return Fp32DatasheetCheck(
            published_tflops=self.datasheet_fp32_tflops,
            published_flops_per_second=int(published * _FLOPS_PER_TFLOP),
            rounding_half_unit_flops_per_second=5 * 10 ** (11 - decimals),
            derived_flops_per_second=self.peak_fp32_flops_per_second,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-safe mapping that names the architecture by capability."""

        return {
            "key": self.key,
            "name": self.name,
            "compute_capability": self.compute_capability,
            "sm_count": self.sm_count,
            "boost_clock_mhz": self.boost_clock_mhz,
            "dram_bandwidth_gb_per_second": self.dram_bandwidth_gb_per_second,
            "datasheet_fp32_tflops": self.datasheet_fp32_tflops,
            "sources": list(self.sources),
        }

    @classmethod
    def from_mapping(
        cls, mapping: object, architectures: Mapping[str, SimtArchitecture]
    ) -> SimtDevice:
        """Parse a device and resolve its declared compute capability."""

        if not isinstance(mapping, Mapping):
            raise SimtError("device must be an object")
        expected = {
            "key",
            "name",
            "compute_capability",
            "sm_count",
            "boost_clock_mhz",
            "dram_bandwidth_gb_per_second",
            "datasheet_fp32_tflops",
            "sources",
        }
        if set(mapping) != expected:
            raise SimtError("device fields do not match the schema")
        capability = mapping["compute_capability"]
        if type(capability) is not str or capability not in architectures:
            raise SimtError("device compute_capability is not a declared architecture")
        return cls(
            key=mapping["key"],
            name=mapping["name"],
            architecture=architectures[capability],
            sm_count=mapping["sm_count"],
            boost_clock_mhz=mapping["boost_clock_mhz"],
            dram_bandwidth_gb_per_second=mapping["dram_bandwidth_gb_per_second"],
            datasheet_fp32_tflops=mapping["datasheet_fp32_tflops"],
            sources=_tuple_of(mapping["sources"], "device sources"),
        )


_GUIDE = "NVIDIA CUDA C++ Programming Guide v12.6 (archive 12.6.3)"
_OCCUPANCY_HEADER = (
    "NVIDIA CUDA Toolkit 12.6 cuda_occupancy.h (cuda_cudart 12.6.77): registers "
    "are allocated per warp in units of 256 across 4 sub-partitions per SM"
)
_TABLES = (
    f"{_GUIDE}, Table 21 Technical Specifications per Compute Capability, and "
    "Table 4 (64 FP32 add/multiply/multiply-add results per clock per SM for "
    "7.x and 8.0)"
)


def _architecture(
    capability: str,
    *,
    max_threads_per_sm: int,
    max_blocks_per_sm: int,
    carveouts_kib: tuple[int, ...],
    optin_kib: int,
    allocation_unit_bytes: int,
    reserved_bytes: int,
    section: str,
) -> SimtArchitecture:
    return SimtArchitecture(
        compute_capability=capability,
        warp_size=32,
        max_threads_per_block=1024,
        max_block_dim_x=1024,
        max_block_dim_y=1024,
        max_block_dim_z=64,
        max_grid_dim_x=2**31 - 1,
        max_grid_dim_y=65_535,
        max_grid_dim_z=65_535,
        max_threads_per_sm=max_threads_per_sm,
        max_warps_per_sm=max_threads_per_sm // 32,
        max_blocks_per_sm=max_blocks_per_sm,
        registers_per_sm=64 * _KIB,
        registers_per_block=64 * _KIB,
        max_registers_per_thread=255,
        register_allocation_unit=256,
        register_allocation_granularity="warp",
        register_sub_partitions_per_sm=4,
        warp_schedulers_per_sm=4,
        fp32_lanes_per_sm=64,
        shared_memory_per_sm_bytes=carveouts_kib[-1] * _KIB,
        shared_memory_carveouts_bytes=tuple(size * _KIB for size in carveouts_kib),
        shared_memory_per_block_bytes=48 * _KIB,
        shared_memory_per_block_optin_bytes=optin_kib * _KIB,
        shared_memory_allocation_unit_bytes=allocation_unit_bytes,
        reserved_shared_memory_per_block_bytes=reserved_bytes,
        sources=(
            _TABLES,
            section,
            f"{_OCCUPANCY_HEADER}; shared memory is allocated in "
            f"{allocation_unit_bytes}-byte units for compute capability {capability}",
        ),
    )


PINNED_ARCHITECTURES: Mapping[str, SimtArchitecture] = MappingProxyType(
    {
        "7.0": _architecture(
            "7.0",
            max_threads_per_sm=2048,
            max_blocks_per_sm=32,
            carveouts_kib=(0, 8, 16, 32, 64, 96),
            optin_kib=96,
            allocation_unit_bytes=256,
            reserved_bytes=0,
            section=(
                f"{_GUIDE}, section 16.6 Compute Capability 7.x: 64 FP32 cores and "
                "4 warp schedulers per SM; Volta shared memory capacity 0, 8, 16, "
                "32, 64 or 96 KB; one block may address the full 96 KB with "
                "dynamic opt-in above 48 KB"
            ),
        ),
        "7.5": _architecture(
            "7.5",
            max_threads_per_sm=1024,
            max_blocks_per_sm=16,
            carveouts_kib=(32, 64),
            optin_kib=64,
            allocation_unit_bytes=256,
            reserved_bytes=0,
            section=(
                f"{_GUIDE}, section 16.6 Compute Capability 7.x: 64 FP32 cores and "
                "4 warp schedulers per SM; Turing shared memory capacity 32 KB or "
                "64 KB; one block may address the full 64 KB with dynamic opt-in "
                "above 48 KB"
            ),
        ),
        "8.0": _architecture(
            "8.0",
            max_threads_per_sm=2048,
            max_blocks_per_sm=32,
            carveouts_kib=(0, 8, 16, 32, 64, 100, 132, 164),
            optin_kib=163,
            allocation_unit_bytes=128,
            reserved_bytes=1 * _KIB,
            section=(
                f"{_GUIDE}, section 16.7 Compute Capability 8.x: 64 FP32 cores and "
                "4 warp schedulers per SM for 8.0; shared memory capacity 0, 8, 16, "
                "32, 64, 100, 132 or 164 KB; up to 163 KB per block with 1 KB "
                "reserved for system use"
            ),
        ),
    }
)


PINNED_DEVICES: Mapping[str, SimtDevice] = MappingProxyType(
    {
        "tesla-v100-sxm2": SimtDevice(
            key="tesla-v100-sxm2",
            name="NVIDIA Tesla V100 SXM2",
            architecture=PINNED_ARCHITECTURES["7.0"],
            sm_count=80,
            boost_clock_mhz=1530,
            dram_bandwidth_gb_per_second=900,
            datasheet_fp32_tflops="15.7",
            sources=(
                "NVIDIA V100 Tensor Core GPU datasheet (JAN20): V100 SXM2 "
                "single-precision 15.7 TFLOPS; memory bandwidth 900 GB/sec",
                "NVIDIA Tesla V100 GPU Architecture whitepaper WP-08608-001_v1.1, "
                "Table 1: Tesla V100 80 SMs, 64 FP32 cores per SM, GPU boost clock "
                "1530 MHz",
            ),
        ),
        "tesla-t4": SimtDevice(
            key="tesla-t4",
            name="NVIDIA T4",
            architecture=PINNED_ARCHITECTURES["7.5"],
            sm_count=40,
            boost_clock_mhz=1590,
            dram_bandwidth_gb_per_second=300,
            datasheet_fp32_tflops="8.1",
            sources=(
                "NVIDIA T4 Tensor Core GPU datasheet (MAR19): single-precision 8.1 "
                "TFLOPS; 16 GB GDDR6 at 300 GB/sec",
                "NVIDIA Turing GPU Architecture whitepaper WP-09183-001_v01, "
                "Table 5: Tesla T4 40 SMs, 64 CUDA cores per SM, GPU boost clock "
                "1,590 MHz; it lists 320 GB/sec, but this record declares the "
                "datasheet figure",
            ),
        ),
        "a100-sxm4-80gb": SimtDevice(
            key="a100-sxm4-80gb",
            name="NVIDIA A100 SXM4 80GB",
            architecture=PINNED_ARCHITECTURES["8.0"],
            sm_count=108,
            boost_clock_mhz=1410,
            dram_bandwidth_gb_per_second=2039,
            datasheet_fp32_tflops="19.5",
            sources=(
                "NVIDIA A100 Tensor Core GPU datasheet (JUN21): FP32 19.5 TFLOPS; "
                "A100 80GB SXM GPU memory bandwidth 2,039 GB/s",
                "NVIDIA A100 Tensor Core GPU Architecture whitepaper V1.0, Table 4: "
                "A100 SXM4 108 SMs, 64 FP32 cores per SM, GPU boost clock 1410 MHz",
            ),
        ),
    }
)
