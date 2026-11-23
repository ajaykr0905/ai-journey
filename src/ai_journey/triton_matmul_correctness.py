"""Correctness of the Triton matmul against float64, one run per config and family.

Each case draws two operand families from the seed:

- ``normal``: the Day 74 standard-normal float32 operands of case ``index``,
  from ``matmul_inputs(seed, CORRECTNESS_STREAM, index, shape)``. The order of
  the additions inside ``tl.dot`` is not fixed, so each output must keep its
  largest relative error within the Day 74 bound
  ``(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)``. That bound holds for
  every summation order (Higham, section 3.1), so it needs no bit pattern.
- ``exact``: integers from -8 to 8 as float32, from stream 2 of the seed. A
  product is at most 64 in size, so a partial sum of at most 8192 terms stays
  at most 2**19, and float32 holds each integer up to 2**24 exactly. Thus
  every order of the additions gives the float64 sum exactly, and the output
  must hold the bits of that sum. A verifier can rebuild such an output on any
  CPU.

Each output buffer is C followed by ``guard_elements`` slots, and every slot
starts as a quiet-NaN sentinel. The guard covers each index that a store
without its mask could reach in the padded tile grid. After a run, the tail
must keep its sentinels, and each element of C must be finite, so a tile that
no program stored also fails. A run that breaks a rule raises
``TritonMatmulError``, so a report never holds a failed run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .kernel_metal_geometry import ceil_div
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS, MAX_SEED
from .matmul_metal_reference import (
    CORRECTNESS_STREAM,
    MAX_CASE_INDEX,
    error_bound,
    float64_reference,
    matmul_inputs,
    matrix_sha256,
    max_relative_error,
)
from .memory_access_tiling import MatmulShape
from .triton_matmul_kernel import (
    TritonConfig,
    TritonMatmulError,
    _integer,
    k_blocks,
    programs,
    require_config,
    require_shape,
)

NORMAL = "normal"
EXACT = "exact"
FAMILIES = (NORMAL, EXACT)
EXACT_STREAM = 2
EXACT_LIMIT = 8
EXACT_INPUTS = (
    "numpy.random.default_rng([seed, 2, index]).integers(-8, 9) draws A[m, k] "
    "and then B[k, n] as int64, converted to float32"
)
NORMAL_INPUTS = (
    "numpy.random.default_rng([seed, 0, index]) draws standard-normal float32 "
    "A[m, k] and then B[k, n], as the Day 74 Metal lab draws case index"
)


def exact_inputs(
    seed: int, index: int, shape: MatmulShape
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Draw integer-valued float32 ``A[m, k]`` and then ``B[k, n]`` in [-8, 8]."""

    _integer(seed, "seed", 0, MAX_SEED)
    _integer(index, "case index", 0, MAX_CASE_INDEX)
    checked = require_shape(shape)
    generator = numpy.random.default_rng([seed, EXACT_STREAM, index])
    left = generator.integers(-EXACT_LIMIT, EXACT_LIMIT + 1, (checked.m, checked.k))
    right = generator.integers(-EXACT_LIMIT, EXACT_LIMIT + 1, (checked.k, checked.n))
    return left.astype(numpy.float32), right.astype(numpy.float32)


def family_inputs(
    family: str, seed: int, index: int, shape: MatmulShape
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return the operands of one family of one case."""

    if family == NORMAL:
        return matmul_inputs(seed, CORRECTNESS_STREAM, index, require_shape(shape))
    if family == EXACT:
        return exact_inputs(seed, index, shape)
    raise TritonMatmulError(f"family must be one of {FAMILIES}")


def guard_elements(shape: MatmulShape, config: TritonConfig) -> int:
    """Return the slots past C that a store without its mask could reach.

    The tile grid pads M to ``Mp = cdiv(M, BLOCK_M) * BLOCK_M`` and N to
    ``Np``. With row-major strides, the last padded element lies at
    ``(Mp - 1) * N + Np - 1``, so the buffer needs ``(Mp - 1) * N + Np``
    slots, and the guard is that count less ``M * N``.
    """

    checked, chosen = require_shape(shape), require_config(config)
    padded_m = ceil_div(checked.m, chosen.block_m) * chosen.block_m
    padded_n = ceil_div(checked.n, chosen.block_n) * chosen.block_n
    return (padded_m - 1) * checked.n + padded_n - checked.m * checked.n


def sentinel_buffer(count: int) -> numpy.ndarray:
    """Return ``count`` float32 quiet NaNs with the Day 73 sentinel bits."""

    return numpy.full(count, FLOAT32_SENTINEL_BITS, dtype=numpy.uint32).view(
        numpy.float32
    )


@dataclass(frozen=True)
class RunRecord:
    """One config and family of one case: its launch, digest and error."""

    config: str
    family: str
    programs: int
    k_blocks: int
    guard_elements: int
    buffer_sha256: str
    max_relative_error: float
    bound_ratio: float
    rounded_reference_differences: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "config": self.config,
            "family": self.family,
            "programs": self.programs,
            "k_blocks": self.k_blocks,
            "guard_elements": self.guard_elements,
            "buffer_sha256": self.buffer_sha256,
            "max_relative_error": self.max_relative_error,
            "bound_ratio": self.bound_ratio,
            "rounded_reference_differences": self.rounded_reference_differences,
        }


@dataclass(frozen=True)
class OperandRecord:
    """The digests of one family's operands and of its float64 reference."""

    family: str
    a_sha256: str
    b_sha256: str
    reference_sha256: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "family": self.family,
            "a_sha256": self.a_sha256,
            "b_sha256": self.b_sha256,
            "reference_sha256": self.reference_sha256,
        }


@dataclass(frozen=True)
class CaseRecord:
    """One shape, checked for every config in both families."""

    index: int
    shape: MatmulShape
    operands: tuple[OperandRecord, ...]
    runs: tuple[RunRecord, ...]

    @property
    def bound(self) -> float:
        return error_bound(self.shape.k)

    def to_mapping(self) -> dict[str, object]:
        return {
            "index": self.index,
            "shape": self.shape.to_mapping(),
            "error_bound": self.bound,
            "operands": [operand.to_mapping() for operand in self.operands],
            "runs": [run.to_mapping() for run in self.runs],
        }


def _output(result: object, length: int, label: str) -> numpy.ndarray:
    if (
        not isinstance(result, numpy.ndarray)
        or result.dtype != numpy.float32
        or result.shape != (length,)
    ):
        raise TritonMatmulError(f"{label} returned no float32 buffer of {length}")
    return numpy.array(result, copy=True)


def check_run(
    backend: object,
    *,
    family: str,
    left: numpy.ndarray,
    right: numpy.ndarray,
    reference: numpy.ndarray,
    magnitude: numpy.ndarray,
    shape: MatmulShape,
    config: TritonConfig,
    label: str,
) -> RunRecord:
    """Run one config on one family's operands and enforce every rule."""

    chosen = require_config(config)
    elements = shape.m * shape.n
    guard = guard_elements(shape, chosen)
    where = f"{label} {family} {chosen.name}"
    buffer = _output(
        backend.run(left, right, chosen, sentinel_buffer(elements + guard)),
        elements + guard,
        where,
    )
    if numpy.any(buffer[elements:].view(numpy.uint32) != FLOAT32_SENTINEL_BITS):
        raise TritonMatmulError(f"{where}: a store escaped its mask into the guard")
    output = buffer[:elements].reshape(shape.m, shape.n)
    if not numpy.isfinite(output).all():
        raise TritonMatmulError(
            f"{where}: an element of C is not finite, so no program stored it "
            "or the sum overflowed"
        )
    error = max_relative_error(output, reference, magnitude)
    bound = error_bound(shape.k)
    if not error <= bound:
        raise TritonMatmulError(
            f"{where}: the relative error {error!r} exceeds the bound {bound!r}"
        )
    rounded = reference.astype(numpy.float32)
    differences = int(
        numpy.count_nonzero(output.view(numpy.uint32) != rounded.view(numpy.uint32))
    )
    if family == EXACT and differences:
        raise TritonMatmulError(
            f"{where}: {differences} elements differ from the exact float64 sum"
        )
    return RunRecord(
        config=chosen.name,
        family=family,
        programs=programs(shape, chosen),
        k_blocks=k_blocks(shape, chosen),
        guard_elements=guard,
        buffer_sha256=matrix_sha256(buffer),
        max_relative_error=error,
        bound_ratio=error / bound,
        rounded_reference_differences=differences,
    )


def check_case(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: MatmulShape,
    configs: tuple[TritonConfig, ...],
) -> CaseRecord:
    """Run every config on both families of one case, config by config."""

    checked = require_shape(shape)
    if type(configs) is not tuple or not configs:
        raise TritonMatmulError("configs must be a non-empty tuple")
    label = f"case {index} ({checked.m} x {checked.n} x {checked.k})"
    operands, inputs = [], {}
    for family in FAMILIES:
        left, right = family_inputs(family, seed, index, checked)
        reference, magnitude = float64_reference(left, right)
        inputs[family] = (left, right, reference, magnitude)
        operands.append(
            OperandRecord(
                family,
                matrix_sha256(left),
                matrix_sha256(right),
                matrix_sha256(reference),
            )
        )
    runs = []
    for config in configs:
        for family in FAMILIES:
            left, right, reference, magnitude = inputs[family]
            runs.append(
                check_run(
                    backend,
                    family=family,
                    left=left,
                    right=right,
                    reference=reference,
                    magnitude=magnitude,
                    shape=checked,
                    config=config,
                    label=label,
                )
            )
    return CaseRecord(index, checked, tuple(operands), tuple(runs))
