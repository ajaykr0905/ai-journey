"""Correctness of the naive Metal matmul kernels against float64 and PyTorch.

Each case draws seeded operands, computes the float64 reference and the CPU
replay of the kernels' fma chain, and runs four implementations: both naive
kernels on the MPS device, and ``torch.matmul`` on the MPS device and on the
CPU. Each naive output buffer starts as sentinels and is longer than C by
the launch's padding threads. The tail must keep its sentinels, which shows
that the guard on both axes stopped every padding thread.

Two rules apply, and a case that breaks one stops the run, so a report never
holds a failed check:

- Each naive output must equal the CPU replay bit for bit. The two kernels
  thus hold the same bits: the mapping changes which thread computes an
  output, not the arithmetic.
- Every output, naive or PyTorch, must keep its largest relative error
  within ``error_bound(K)``.

Each record keeps the digest of its whole buffer, its largest relative error,
and the count of elements whose bits differ from the naive kernels, so a
report shows where the summation order of ``torch.matmul`` differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .kernel_metal_geometry import CheckedLaunch, require_integer
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS
from .matmul_metal_mapping import (
    MAPPINGS,
    MatmulMetalError,
    matmul_launch,
    matmul_shape,
)
from .matmul_metal_reference import (
    CORRECTNESS_STREAM,
    MAX_CASE_INDEX,
    error_bound,
    float64_reference,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
    max_relative_error,
)
from .matmul_metal_runtime import TORCH_MATMUL
from .matmul_metal_source import KERNEL_FOR_MAPPING
from .memory_access_tiling import MatmulShape

# (implementation, device) in the order every case records them.
IMPLEMENTATIONS = (
    *((KERNEL_FOR_MAPPING[mapping], "mps") for mapping in MAPPINGS),
    (TORCH_MATMUL, "mps"),
    (TORCH_MATMUL, "cpu"),
)


@dataclass(frozen=True)
class OutputRecord:
    """One implementation's output: where it ran, its digest and its error."""

    implementation: str
    device: str
    buffer_sha256: str
    guard_elements: int
    max_relative_error: float
    differing_elements: int

    def to_mapping(self, bound: float) -> dict[str, object]:
        """Return the record as a report stores it, with its share of the bound."""

        return {
            "implementation": self.implementation,
            "device": self.device,
            "buffer_sha256": self.buffer_sha256,
            "guard_elements": self.guard_elements,
            "max_relative_error": self.max_relative_error,
            "bound_ratio": self.max_relative_error / bound,
            "differing_elements": self.differing_elements,
        }


@dataclass(frozen=True)
class MatmulCase:
    """One seeded case, checked on every implementation."""

    index: int
    shape: MatmulShape
    launches: tuple[CheckedLaunch, ...]
    a_sha256: str
    b_sha256: str
    reference_sha256: str
    replay_sha256: str
    outputs: tuple[OutputRecord, ...]

    @property
    def bound(self) -> float:
        """The largest relative error this case's K allows."""

        return error_bound(self.shape.k)

    @property
    def max_relative_error(self) -> float:
        """The largest relative error of any implementation in this case."""

        return max(output.max_relative_error for output in self.outputs)

    def to_mapping(self) -> dict[str, object]:
        """Return the case as a report stores it."""

        return {
            "index": self.index,
            **self.shape.to_mapping(),
            "launches": {
                mapping: launch.to_mapping()
                for mapping, launch in zip(MAPPINGS, self.launches)
            },
            "a_sha256": self.a_sha256,
            "b_sha256": self.b_sha256,
            "reference_sha256": self.reference_sha256,
            "replay_sha256": self.replay_sha256,
            "bound": self.bound,
            "outputs": [output.to_mapping(self.bound) for output in self.outputs],
        }


def _float32(result: object, shape: tuple[int, ...], label: str) -> numpy.ndarray:
    if (
        not isinstance(result, numpy.ndarray)
        or result.dtype != numpy.float32
        or result.shape != shape
    ):
        raise MatmulMetalError(f"{label} returned no float32 output of shape {shape}")
    return numpy.ascontiguousarray(result).reshape(-1)


def _within_bound(error: float, bound: float, label: str) -> float:
    if not error <= bound:
        raise MatmulMetalError(
            f"{label} has a relative error of {error!r}; the bound is {bound!r}"
        )
    return error


def check_case(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: MatmulShape,
    group_size: tuple[int, int],
) -> MatmulCase:
    """Run one case on every implementation and enforce both rules."""

    require_integer(index, "case index", 0, MAX_CASE_INDEX)
    if not isinstance(shape, MatmulShape):
        raise TypeError("shape must be a MatmulShape")
    shape = matmul_shape(shape.m, shape.n, shape.k)
    left, right = matmul_inputs(seed, CORRECTNESS_STREAM, index, shape)
    reference, magnitude = float64_reference(left, right)
    replay = fma_replay(left, right)
    replay_bits = replay.reshape(-1).view(numpy.uint32)
    bound = error_bound(shape.k)
    label = f"case {index} ({shape.m} x {shape.n} x {shape.k})"
    elements = shape.m * shape.n
    launches, outputs = [], []
    for mapping in MAPPINGS:
        name = KERNEL_FOR_MAPPING[mapping]
        launch = CheckedLaunch(
            matmul_launch(mapping, shape, group_size), backend.limits[name]
        )
        guard = launch.geometry.padding_threads
        buffer = _float32(
            backend.run(name, left, right, launch),
            (elements + guard,),
            f"{label}: {name}",
        )
        if numpy.any(buffer[elements:].view(numpy.uint32) != FLOAT32_SENTINEL_BITS):
            raise MatmulMetalError(
                f"{label}: a padding thread of {name} wrote into the guard tail"
            )
        if not numpy.array_equal(buffer[:elements].view(numpy.uint32), replay_bits):
            raise MatmulMetalError(
                f"{label}: {name} differs from the CPU replay of its fma chain"
            )
        error = max_relative_error(
            buffer[:elements].reshape(shape.m, shape.n), reference, magnitude
        )
        launches.append(launch)
        outputs.append(
            OutputRecord(
                name,
                "mps",
                matrix_sha256(buffer),
                guard,
                _within_bound(error, bound, f"{label}: {name}"),
                0,
            )
        )
    for implementation, device in IMPLEMENTATIONS[len(MAPPINGS) :]:
        where = f"{label}: {implementation} on {device}"
        result = _float32(
            backend.torch_matmul(left, right, device), (shape.m, shape.n), where
        )
        error = max_relative_error(
            result.reshape(shape.m, shape.n), reference, magnitude
        )
        outputs.append(
            OutputRecord(
                implementation,
                device,
                matrix_sha256(result),
                0,
                _within_bound(error, bound, where),
                int(numpy.count_nonzero(result.view(numpy.uint32) != replay_bits)),
            )
        )
    return MatmulCase(
        index=index,
        shape=shape,
        launches=tuple(launches),
        a_sha256=matrix_sha256(left),
        b_sha256=matrix_sha256(right),
        reference_sha256=matrix_sha256(reference),
        replay_sha256=matrix_sha256(replay),
        outputs=tuple(outputs),
    )
