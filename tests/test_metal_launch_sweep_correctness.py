import numpy
import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.kernel_metal_reference import array_sha256, float32_sentinels
from ai_journey.metal_launch_sweep_correctness import (
    LATTICE_POINTS,
    check_matmul_case,
    check_vector_sizes,
    lattice,
    matmul_accuracy,
    matmul_inputs,
    matmul_tolerance,
)
from ai_journey.metal_launch_sweep_shapes import MetalLaunchSweepError, admit_shapes
from ai_journey.metal_launch_sweep_source import MATMUL
from test_kernel_metal_runtime import requires_metal
from test_metal_launch_sweep_standin import StandInBackend, rounded_product

SWEPT = ((8, 8), (16, 16), (32, 32), (32, 8), (8, 32), (64, 4), (4, 64))
SIZES = ((8,), (16,), (32,), (48,), (256,))


def matmul_admissions(backend, shapes=SWEPT):
    limits = backend.limits[MATMUL]
    return admit_shapes(
        shapes, limits, requires_whole_simd_groups=backend.requires_whole_simd_groups
    )


def vector_admissions(backend, sizes=SIZES):
    return admit_shapes(
        sizes,
        backend.limits["vector_add"],
        requires_whole_simd_groups=backend.requires_whole_simd_groups,
    )


def test_inputs_are_seeded_per_case_and_the_bound_follows_k():
    a, b = matmul_inputs(74, 0, 37, 53, 29)

    assert (a.shape, b.shape, a.dtype, b.dtype) == (
        (37, 29),
        (29, 53),
        numpy.float32,
        numpy.float32,
    )
    again, _ = matmul_inputs(74, 0, 37, 53, 29)
    other, _ = matmul_inputs(74, 1, 37, 53, 29)
    assert numpy.array_equal(a, again) and not numpy.array_equal(a, other)
    gamma = 29 * 2.0**-24 / (1 - 29 * 2.0**-24) + 29 * 2.0**-53 / (1 - 29 * 2.0**-53)
    assert matmul_tolerance(29) == gamma
    with pytest.raises(TypeError, match="k must be an integer"):
        matmul_tolerance(2.0)


def test_the_lattice_spans_each_axis_from_the_first_to_the_last_position():
    assert LATTICE_POINTS == 5
    assert lattice(1) == (0,)
    assert lattice(2) == (0, 1)
    assert lattice(37) == (0, 9, 18, 27, 36)
    assert lattice(2048) == (0, 511, 1023, 1535, 2047)


def test_accuracy_keeps_the_ratio_and_the_lattice_values_of_a_good_result():
    a, b = matmul_inputs(74, 2, 37, 53, 29)
    c = rounded_product(a, b)

    accuracy = matmul_accuracy(a, b, c.reshape(-1), "case")

    assert accuracy.output_sha256 == array_sha256(c)
    assert 0 < accuracy.max_bound_ratio < 1
    assert len(accuracy.samples) == 25
    row, column, value = accuracy.samples[7]
    assert (row, column) == (9, 26) and value == float(c[9, 26])
    assert accuracy.to_mapping()["samples"][0] == [0, 0, float(c[0, 0])]


def test_accuracy_refuses_an_element_outside_the_bound_or_a_non_finite_one():
    a, b = matmul_inputs(74, 3, 16, 16, 64)
    c = rounded_product(a, b)
    reference = a.astype(numpy.float64) @ b.astype(numpy.float64)
    bound = matmul_tolerance(64) * (numpy.abs(a) @ numpy.abs(b)).astype(numpy.float64)

    far = c.copy()
    far[3, 5] = numpy.float32(reference[3, 5] + 4 * bound[3, 5])
    with pytest.raises(MetalLaunchSweepError, match="exceeds the float32 rounding"):
        matmul_accuracy(a, b, far, "case")
    broken = c.copy()
    broken[0, 0] = numpy.nan
    with pytest.raises(MetalLaunchSweepError, match="not finite"):
        matmul_accuracy(a, b, broken, "case")
    with pytest.raises(MetalLaunchSweepError, match="no 16x16 result"):
        matmul_accuracy(a, b, c[:8], "case")


def test_every_shape_runs_a_case_and_keeps_its_own_guard_count():
    backend = StandInBackend()

    case = check_matmul_case(
        backend,
        seed=74,
        index=0,
        shape=(37, 53, 29),
        admissions=matmul_admissions(backend),
    )

    mapping = case.to_mapping()
    assert [item["group_size"] for item in mapping["shapes"]] == [
        list(shape) for shape in SWEPT
    ]
    paddings = [launch.geometry.padding_threads for launch in backend.launches]
    assert [item["guard_elements"] for item in mapping["shapes"]] == paddings
    assert paddings[2] == 64 * 64 - 37 * 53
    a, b = matmul_inputs(74, 0, 37, 53, 29)
    assert (mapping["a_sha256"], mapping["b_sha256"]) == (
        array_sha256(a),
        array_sha256(b),
    )
    assert mapping["output_sha256"] == array_sha256(rounded_product(a, b))
    assert len(mapping["samples"]) == 25


class ShapeDependent(StandInBackend):
    def matmul(self, a, b, launch):
        buffer = super().matmul(a, b, launch)
        if launch.geometry.group_size == (4, 64):
            buffer[0] = numpy.nextafter(buffer[0], numpy.float32(numpy.inf))
        return buffer


class WritesTail(StandInBackend):
    def matmul(self, a, b, launch):
        buffer = super().matmul(a, b, launch)
        buffer[-1] = 0.0
        return buffer


class Wrong(StandInBackend):
    def matmul(self, a, b, launch):
        product = rounded_product(a, b).reshape(-1) + 1
        tail = float32_sentinels(launch.geometry.padding_threads)
        return numpy.concatenate([product, tail])


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (ShapeDependent(), "with group 4x64: the result differs between shapes"),
        (WritesTail(), "with group 8x8: a padding thread wrote the tail"),
        (Wrong(), "exceeds the float32 rounding bound"),
    ],
)
def test_a_case_stops_on_a_shape_dependent_wrong_or_unguarded_result(backend, message):
    with pytest.raises(MetalLaunchSweepError, match=message):
        check_matmul_case(
            backend,
            seed=74,
            index=0,
            shape=(37, 53, 29),
            admissions=matmul_admissions(backend),
        )


def test_the_day_73_vector_check_runs_once_per_admitted_group_size():
    backend = StandInBackend()
    strict = StandInBackend(requires_whole_simd_groups=True)

    cases = check_vector_sizes(
        backend, seed=74, length=65537, admissions=vector_admissions(backend)
    )
    kept = check_vector_sizes(
        strict, seed=74, length=65537, admissions=vector_admissions(strict)
    )

    assert [case.launch.geometry.group_size for case in cases] == list(SIZES)
    assert [case.index for case in cases] == [0, 1, 2, 3, 4]
    # Sizes 8, 16 and 48 are not whole SIMD groups; the others keep their place.
    assert [(case.index, case.launch.geometry.group_size) for case in kept] == [
        (2, (32,)),
        (4, (256,)),
    ]
    assert cases[3].to_mapping()["launch"]["idle_lanes_per_threadgroup"] == 16
    assert {case.to_mapping()["outputs"][0]["guard_elements"] for case in cases} == {
        case.launch.geometry.padding_threads for case in cases
    }


def test_a_sweep_without_an_admitted_shape_has_nothing_to_check():
    backend = StandInBackend(PipelineLimits(32, 32, 0))

    with pytest.raises(MetalLaunchSweepError, match="no shape of the sweep"):
        check_matmul_case(
            backend,
            seed=74,
            index=0,
            shape=(37, 53, 29),
            admissions=matmul_admissions(backend, ((8, 8), (16, 16))),
        )
    with pytest.raises(MetalLaunchSweepError, match="no shape of the sweep"):
        check_vector_sizes(
            backend,
            seed=74,
            length=1000,
            admissions=vector_admissions(backend, ((64,), (128,))),
        )


@requires_metal
def test_this_gpu_gives_one_bounded_result_for_every_swept_shape():
    from test_metal_launch_sweep_runtime import sweep_runtime

    runtime = sweep_runtime()

    case = check_matmul_case(
        runtime,
        seed=74,
        index=0,
        shape=(37, 53, 29),
        admissions=matmul_admissions(runtime),
    )
    cases = check_vector_sizes(
        runtime, seed=74, length=65537, admissions=vector_admissions(runtime)
    )

    assert len(case.guard_elements) == len(SWEPT)
    assert case.accuracy.max_bound_ratio < 1
    assert [case.index for case in cases] == [0, 1, 2, 3, 4]
