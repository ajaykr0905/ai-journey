"""Observe how this host splits each admitted threadgroup shape into SIMD groups.

The SIMD footprints of the admission records assume the linear layout: the
linear index is ``x + gx * y``, the SIMD group is ``linear // width`` and the
lane is ``linear % width``. The probe kernel records, for each thread of one
threadgroup, the SIMD group, lane, x, y and SIMD width that the GPU gave it.
The run compares that record with the model word for word and stops at any
difference, because each footprint in the report rests on the model. The
observed words then equal the model's, so the report keeps their SHA-256,
and a verifier rebuilds them from the shape and the width alone.

The probe shows which SIMD group and lane ran each thread. It does not show
when or where the GPU scheduled a SIMD group.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import prod

import numpy

from .kernel_metal_geometry import MAX_THREADGROUP_THREADS, require_integer
from .metal_launch_sweep_shapes import (
    LINEAR_LAYOUT,
    MetalLaunchSweepError,
    ShapeAdmission,
    SimdFootprint,
    group_shape,
    shape_label,
)
from .metal_launch_sweep_source import LAYOUT_FIELDS, SIMD_LAYOUT_PROBE


def linear_layout(group: tuple[int, ...], width: int) -> numpy.ndarray:
    """Return the words the probe writes under the linear layout, a row per thread."""

    shape = group_shape(group)
    lanes = require_integer(width, "simd width", 1, MAX_THREADGROUP_THREADS)
    linear = numpy.arange(prod(shape), dtype=numpy.uint32)
    columns = numpy.uint32(shape[0])
    words = (
        linear // numpy.uint32(lanes),
        linear % numpy.uint32(lanes),
        linear % columns,
        linear // columns,
        numpy.full(linear.size, lanes, dtype=numpy.uint32),
    )
    return numpy.stack(words, axis=1)


def words_sha256(words: numpy.ndarray) -> str:
    """Hash probe words as little-endian uint32 values in row order."""

    if not isinstance(words, numpy.ndarray) or words.dtype != numpy.uint32:
        raise MetalLaunchSweepError("probe words must be a uint32 array")
    return sha256(numpy.ascontiguousarray(words, dtype="<u4").tobytes()).hexdigest()


@dataclass(frozen=True)
class SimdLayout:
    """One observed threadgroup split that matched the linear layout."""

    group_size: tuple[int, ...]
    simd_width: int
    observed_sha256: str
    footprint: SimdFootprint

    def to_mapping(self) -> dict[str, object]:
        """Return the observation as a report records it."""

        return {
            "group_size": list(self.group_size),
            "layout": LINEAR_LAYOUT,
            "simd_width": self.simd_width,
            "observed_sha256": self.observed_sha256,
            **self.footprint.to_mapping(),
        }


def _first_difference(observed: numpy.ndarray, model: numpy.ndarray) -> str:
    thread = int(numpy.flatnonzero((observed != model).any(axis=1))[0])
    seen = dict(zip(LAYOUT_FIELDS, observed[thread].tolist()))
    expected = dict(zip(LAYOUT_FIELDS, model[thread].tolist()))
    return f"thread {thread} reported {seen}, the linear layout gives {expected}"


def observe_simd_layout(backend: object, admission: ShapeAdmission) -> SimdLayout:
    """Run the probe for one admitted shape and require the linear layout.

    The probe must see the execution width of the pipeline that admitted the
    shape, or its record could not back that pipeline's footprint.
    """

    if not isinstance(admission, ShapeAdmission) or not admission.admitted:
        raise MetalLaunchSweepError("only an admitted shape is probed")
    width = admission.limits.thread_execution_width
    probe_width = backend.limits[SIMD_LAYOUT_PROBE].thread_execution_width
    label = shape_label(admission.group_size)
    if probe_width != width:
        raise MetalLaunchSweepError(
            f"shape {label}: the probe pipeline has width {probe_width}, but the "
            f"pipeline that admitted it has width {width}"
        )
    model = linear_layout(admission.group_size, width)
    observed = backend.simd_layout(admission.group_size)
    if (
        not isinstance(observed, numpy.ndarray)
        or observed.dtype != numpy.uint32
        or observed.shape != model.shape
    ):
        raise MetalLaunchSweepError(
            f"shape {label}: the probe returned no {model.shape} uint32 record"
        )
    if not numpy.array_equal(observed, model):
        raise MetalLaunchSweepError(
            f"shape {label}: {_first_difference(observed, model)}"
        )
    return SimdLayout(
        group_size=admission.group_size,
        simd_width=width,
        observed_sha256=words_sha256(observed),
        footprint=admission.footprint,
    )
