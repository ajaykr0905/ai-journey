"""The registry of tiled matmul variants and each pipeline's verdict on its tile.

A variant is one kernel function: its name, its tile size T, its source text,
the SHA-256 digest of that text and a declared correctness mode. A run
compiles the variants that its config names into one library, the header
followed by their kernels in config order. Each digest covers one kernel's
text alone, and the library digest covers only the variants of the run, so a
later lab can register more variants without moving any digest of this one.

The correctness mode declares which arithmetic a variant's source performs.
It never decides a result:

- ``k_order_fma``: the source runs the ``fma_replay`` chain, K steps of
  ``sum = fma(a, b, sum)`` in increasing k from 0, so the GPU output may hold
  the replay's bits. A run measures whether it does.
- ``reordered_sum``: the source adds the products in some other order, so
  only the error bound is expected to hold.

Every run enforces the error bound and the untouched guard tail on every
variant, and records bit equality with the replay as a measurement.

The pipeline verdict comes from the limits that a compiled pipeline reports
about itself. A tile is refused when its pipeline allows fewer than T * T
threads in a threadgroup, or when the pipeline's static threadgroup memory
differs from the ``2 * T * T * 4`` bytes that the two tiles declare. A
refusal is data in the report, and a refused variant never launches.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

from .kernel_metal_geometry import PipelineLimits
from .matmul_metal_source import BUFFERS, SCALAR_BUFFERS
from .matmul_tiled_source import (
    HEADER,
    TILE_SIZES,
    MatmulTiledError,
    require_tile,
    threadgroup_bytes,
    tiled_kernel_name,
    tiled_kernel_source,
)

K_ORDER_FMA = "k_order_fma"
REORDERED_SUM = "reordered_sum"
CORRECTNESS_MODES = (K_ORDER_FMA, REORDERED_SUM)
FEWER_THREADS = "pipeline_allows_fewer_threads_than_the_tile"
OTHER_THREADGROUP_BYTES = "static_threadgroup_bytes_differ_from_the_tiles"
REFUSAL_REASONS = (FEWER_THREADS, OTHER_THREADGROUP_BYTES)
MAX_VARIANTS = 8
_NAME = re.compile(r"matmul_[a-z0-9_]{1,48}")


def text_sha256(text: str) -> str:
    """Hash the exact ASCII text that ``compile_shader`` receives."""

    return sha256(text.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class TiledVariant:
    """One kernel of the tiled library and what it declares about itself."""

    name: str
    tile: int
    source: str
    correctness_mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MatmulTiledError("a variant name must be matmul_ and a short slug")
        require_tile(self.tile)
        if (
            not isinstance(self.source, str)
            or not self.source.isascii()
            or f"kernel void {self.name}(" not in self.source
        ):
            raise MatmulTiledError(f"{self.name} needs ASCII source that defines it")
        if self.correctness_mode not in CORRECTNESS_MODES:
            raise MatmulTiledError(
                f"correctness_mode must be one of {', '.join(CORRECTNESS_MODES)}"
            )

    @property
    def group_size(self) -> tuple[int, int]:
        """The ``(x, y)`` threadgroup: one thread per element of a C tile."""

        return (self.tile, self.tile)

    @property
    def threads_per_group(self) -> int:
        """Threads in one threadgroup: T * T."""

        return self.tile * self.tile

    @property
    def threadgroup_bytes(self) -> int:
        """Static threadgroup memory the two tiles declare."""

        return threadgroup_bytes(self.tile)

    @property
    def source_sha256(self) -> str:
        """The digest of this kernel's text alone."""

        return text_sha256(self.source)

    @property
    def arg_casts(self) -> dict[int, str]:
        """Pass each scalar as 32 bits, the width of ``constant uint&``."""

        return {index: "int32" for index in SCALAR_BUFFERS}

    def to_mapping(self) -> dict[str, object]:
        """Return what a report records about the variant, without its text."""

        return {
            "name": self.name,
            "tile": self.tile,
            "group_size": list(self.group_size),
            "threads_per_group": self.threads_per_group,
            "threadgroup_bytes": self.threadgroup_bytes,
            "correctness_mode": self.correctness_mode,
            "buffers": list(BUFFERS),
            "scalar_buffers": list(SCALAR_BUFFERS),
            "source_sha256": self.source_sha256,
        }


def tiled_variant(tile: int) -> TiledVariant:
    """Return the Day 76 tiled kernel of one tile size."""

    return TiledVariant(
        name=tiled_kernel_name(tile),
        tile=tile,
        source=tiled_kernel_source(tile),
        correctness_mode=K_ORDER_FMA,
    )


VARIANTS: dict[str, TiledVariant] = {
    variant.name: variant for variant in map(tiled_variant, TILE_SIZES)
}


def variants_named(names: Iterable[object]) -> tuple[TiledVariant, ...]:
    """Return the registered variants of ``names``, in order, each at most once."""

    chosen = tuple(names)
    if not 1 <= len(chosen) <= MAX_VARIANTS:
        raise MatmulTiledError(f"a run names 1 to {MAX_VARIANTS} variants")
    if not all(isinstance(name, str) for name in chosen):
        raise MatmulTiledError("variant names must be strings")
    if len(set(chosen)) != len(chosen):
        raise MatmulTiledError("a run must not name a variant twice")
    for name in chosen:
        if name not in VARIANTS:
            raise MatmulTiledError(f"{name!r} is not a registered variant")
    return tuple(VARIANTS[name] for name in chosen)


def library_source(variants: tuple[TiledVariant, ...]) -> str:
    """Join the header and the kernels of one run into one library text."""

    if type(variants) is not tuple or not variants:
        raise MatmulTiledError("a library needs a tuple of variants")
    if not all(isinstance(variant, TiledVariant) for variant in variants):
        raise TypeError("a library holds TiledVariant values")
    return HEADER + "".join("\n" + variant.source for variant in variants)


@dataclass(frozen=True)
class VariantAdmission:
    """One pipeline's verdict on one variant's tile, with the numbers behind it."""

    variant: TiledVariant
    limits: PipelineLimits

    def __post_init__(self) -> None:
        if not isinstance(self.variant, TiledVariant):
            raise TypeError("variant must be a TiledVariant")
        if not isinstance(self.limits, PipelineLimits):
            raise TypeError("limits must be PipelineLimits")

    @property
    def reason(self) -> str | None:
        """The first rule the pipeline breaks, or ``None`` when it admits the tile."""

        if self.limits.max_threads_per_threadgroup < self.variant.threads_per_group:
            return FEWER_THREADS
        if (
            self.limits.static_threadgroup_memory_bytes
            != self.variant.threadgroup_bytes
        ):
            return OTHER_THREADGROUP_BYTES
        return None

    @property
    def admitted(self) -> bool:
        """Whether the variant may launch on this pipeline."""

        return self.reason is None

    def to_mapping(self) -> dict[str, object]:
        """Return the verdict as a report records it."""

        return {
            "name": self.variant.name,
            "admitted": self.admitted,
            "reason": self.reason,
            "required_threads": self.variant.threads_per_group,
            "allowed_threads": self.limits.max_threads_per_threadgroup,
            "declared_threadgroup_bytes": self.variant.threadgroup_bytes,
            "static_threadgroup_bytes": self.limits.static_threadgroup_memory_bytes,
        }
