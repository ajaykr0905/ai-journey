"""Global-memory traffic of naive and shared-memory tiled matrix multiplication.

For ``C[M, N] = A[M, K] @ B[K, N]`` the counts are analytical:

- naive: every output element reads one row of A and one column of B directly
  from global memory, ``2 * M * N * K`` element loads with no cache reuse
  modeled;
- tiled: one T x T thread block computes one T x T tile of C in
  ``ceil(K / T)`` phases, and each phase stages one T x T tile of A and one of
  B in shared memory: ``2 * T * T * ceil(K / T)`` tile-slot loads per block
  over ``ceil(M / T) * ceil(N / T)`` blocks. Slots of partial edge tiles that
  fall outside A or B are padded slots, which a bounds-checked kernel
  zero-fills without a global load, so only useful loads are traffic;
- both kernels store each element of C once and perform ``2 * M * N * K``
  useful FLOPs.

Source: CUDA C++ Best Practices Guide v13.5, section 10.2.3.2 (shared memory
in matrix multiplication). With one thread per element of a C tile, T is at
most 32 because a block holds at most 1024 threads (CUDA C++ Programming
Guide v12.6, section 16.2). L1/L2 caching, register blocking, occupancy and
launch overhead are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory_access_pattern import MemoryAccessError
from .resource_accounting import bytes_per_element

MATMUL_DTYPES = ("float64", "float32", "float16", "bfloat16")
MAX_MATMUL_DIMENSION = 1 << 16
MAX_MATMUL_TILE = 32


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def matmul_element_bytes(dtype: object) -> int:
    """Resolve the width of one supported floating-point matmul element."""

    if not isinstance(dtype, str):
        raise TypeError("dtype must be a string")
    if dtype not in MATMUL_DTYPES:
        raise MemoryAccessError(f"dtype must be one of {', '.join(MATMUL_DTYPES)}")
    return bytes_per_element(dtype)


@dataclass(frozen=True)
class MatmulShape:
    """Dimensions of ``C[m, n] = A[m, k] @ B[k, n]``."""

    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        for name in ("m", "n", "k"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if not 1 <= value <= MAX_MATMUL_DIMENSION:
                raise MemoryAccessError(
                    f"{name} must be in [1, {MAX_MATMUL_DIMENSION}]"
                )

    @property
    def flops(self) -> int:
        """Count one multiply and one add per scalar product term."""

        return 2 * self.m * self.n * self.k

    def to_mapping(self) -> dict[str, int]:
        return {"m": self.m, "n": self.n, "k": self.k}


@dataclass(frozen=True)
class NaiveMatmulTraffic:
    """Every operand element is loaded from global memory for every use."""

    shape: MatmulShape
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not isinstance(self.shape, MatmulShape):
            raise TypeError("shape must be a MatmulShape")
        matmul_element_bytes(self.dtype)

    @property
    def element_bytes(self) -> int:
        return matmul_element_bytes(self.dtype)

    @property
    def load_elements(self) -> int:
        return 2 * self.shape.m * self.shape.n * self.shape.k

    @property
    def store_elements(self) -> int:
        return self.shape.m * self.shape.n

    @property
    def load_bytes(self) -> int:
        return self.load_elements * self.element_bytes

    @property
    def store_bytes(self) -> int:
        return self.store_elements * self.element_bytes

    @property
    def total_bytes(self) -> int:
        return self.load_bytes + self.store_bytes

    @property
    def flops(self) -> int:
        return self.shape.flops

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.total_bytes

    def to_mapping(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "element_bytes": self.element_bytes,
            "load_elements": self.load_elements,
            "store_elements": self.store_elements,
            "load_bytes": self.load_bytes,
            "store_bytes": self.store_bytes,
            "total_bytes": self.total_bytes,
            "flops": self.flops,
            "arithmetic_intensity": self.arithmetic_intensity,
        }


@dataclass(frozen=True)
class TiledMatmulTraffic:
    """Shared-memory tiling with one T x T thread block per C tile."""

    shape: MatmulShape
    tile: int
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not isinstance(self.shape, MatmulShape):
            raise TypeError("shape must be a MatmulShape")
        if type(self.tile) is not int:
            raise TypeError("tile must be an integer")
        if not 1 <= self.tile <= MAX_MATMUL_TILE:
            raise MemoryAccessError(f"tile must be in [1, {MAX_MATMUL_TILE}]")
        matmul_element_bytes(self.dtype)

    @property
    def element_bytes(self) -> int:
        return matmul_element_bytes(self.dtype)

    @property
    def block_rows(self) -> int:
        return _ceil_div(self.shape.m, self.tile)

    @property
    def block_columns(self) -> int:
        return _ceil_div(self.shape.n, self.tile)

    @property
    def blocks(self) -> int:
        return self.block_rows * self.block_columns

    @property
    def phases(self) -> int:
        return _ceil_div(self.shape.k, self.tile)

    @property
    def threads_per_block(self) -> int:
        return self.tile * self.tile

    @property
    def shared_memory_bytes_per_block(self) -> int:
        return 2 * self.tile * self.tile * self.element_bytes

    @property
    def tile_slot_loads(self) -> int:
        return 2 * self.tile * self.tile * self.phases * self.blocks

    @property
    def useful_a_loads(self) -> int:
        """Every block column reads all of A once."""

        return self.block_columns * self.shape.m * self.shape.k

    @property
    def useful_b_loads(self) -> int:
        """Every block row reads all of B once."""

        return self.block_rows * self.shape.k * self.shape.n

    @property
    def useful_loads(self) -> int:
        return self.useful_a_loads + self.useful_b_loads

    @property
    def padded_slots(self) -> int:
        return self.tile_slot_loads - self.useful_loads

    @property
    def load_elements(self) -> int:
        return self.useful_loads

    @property
    def store_elements(self) -> int:
        return self.shape.m * self.shape.n

    @property
    def load_bytes(self) -> int:
        return self.load_elements * self.element_bytes

    @property
    def store_bytes(self) -> int:
        return self.store_elements * self.element_bytes

    @property
    def total_bytes(self) -> int:
        return self.load_bytes + self.store_bytes

    @property
    def flops(self) -> int:
        return self.shape.flops

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.total_bytes

    @property
    def reuse_factor(self) -> float:
        """Return naive element loads divided by useful tiled loads."""

        return 2 * self.shape.m * self.shape.n * self.shape.k / self.useful_loads

    def to_mapping(self) -> dict[str, object]:
        return {
            "tile": self.tile,
            "dtype": self.dtype,
            "element_bytes": self.element_bytes,
            "block_rows": self.block_rows,
            "block_columns": self.block_columns,
            "blocks": self.blocks,
            "phases": self.phases,
            "threads_per_block": self.threads_per_block,
            "shared_memory_bytes_per_block": self.shared_memory_bytes_per_block,
            "tile_slot_loads": self.tile_slot_loads,
            "useful_a_loads": self.useful_a_loads,
            "useful_b_loads": self.useful_b_loads,
            "useful_loads": self.useful_loads,
            "padded_slots": self.padded_slots,
            "store_elements": self.store_elements,
            "load_bytes": self.load_bytes,
            "store_bytes": self.store_bytes,
            "total_bytes": self.total_bytes,
            "flops": self.flops,
            "arithmetic_intensity": self.arithmetic_intensity,
            "reuse_factor": self.reuse_factor,
        }
