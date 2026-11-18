import numpy
import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.metal_launch_sweep_layout import (
    linear_layout,
    observe_simd_layout,
    words_sha256,
)
from ai_journey.metal_launch_sweep_shapes import (
    MetalLaunchSweepError,
    ShapeAdmission,
    admit_shapes,
)
from ai_journey.metal_launch_sweep_source import SIMD_LAYOUT_PROBE

HOST = PipelineLimits(1024, 32, 0)


class Probe:
    """A stand-in GPU whose probe returns a chosen record for each shape."""

    def __init__(self, layout=linear_layout, width=32):
        self.limits = {SIMD_LAYOUT_PROBE: PipelineLimits(1024, width, 0)}
        self._layout = layout
        self.shapes = []

    def simd_layout(self, group):
        self.shapes.append(group)
        return self._layout(group, 32)


def test_the_linear_layout_numbers_threads_x_fastest_in_simd_groups():
    words = linear_layout((4, 64), 32)

    assert (words.shape, words.dtype) == ((256, 5), numpy.uint32)
    # Thread 33 is the second lane of the second SIMD group: x 1, row 8.
    assert words[0].tolist() == [0, 0, 0, 0, 32]
    assert words[33].tolist() == [1, 1, 1, 8, 32]
    assert words[255].tolist() == [7, 31, 3, 63, 32]
    assert linear_layout((48,), 32)[47].tolist() == [1, 15, 47, 0, 32]
    assert words_sha256(words) == words_sha256(words.astype(">u4").astype("u4"))


def test_a_record_that_matches_the_model_is_kept_by_its_digest_and_footprint():
    probe = Probe()
    admission = ShapeAdmission((4, 64), HOST, False)

    layout = observe_simd_layout(probe, admission)

    assert probe.shapes == [(4, 64)]
    assert layout.to_mapping() == {
        "group_size": [4, 64],
        "layout": "linear_x_fastest",
        "simd_width": 32,
        "observed_sha256": words_sha256(linear_layout((4, 64), 32)),
        "simd_groups": 8,
        "idle_lanes": 0,
        "rows_per_simd_group": 8,
        "columns_per_simd_group": 4,
    }


def y_fastest(group, width):
    words = linear_layout(group, width)
    if len(group) == 2:
        gx, gy = group
        linear = numpy.arange(gx * gy, dtype=numpy.uint32)
        words[:, 2], words[:, 3] = linear // gy, linear % gy
    return words


def test_a_different_split_on_the_gpu_stops_the_run_with_the_first_thread():
    with pytest.raises(MetalLaunchSweepError, match="shape 4x64: thread 1 reported"):
        observe_simd_layout(Probe(y_fastest), ShapeAdmission((4, 64), HOST, False))


def test_the_probe_must_see_the_width_of_the_admitting_pipeline():
    narrow = PipelineLimits(1024, 16, 0)

    with pytest.raises(MetalLaunchSweepError, match="probe pipeline has width 32"):
        observe_simd_layout(Probe(), ShapeAdmission((16, 16), narrow, False))


def test_only_admitted_shapes_with_a_full_record_are_probed():
    (rejected,) = admit_shapes(
        ((32, 32),), PipelineLimits(512, 32, 0), requires_whole_simd_groups=False
    )
    with pytest.raises(MetalLaunchSweepError, match="only an admitted shape"):
        observe_simd_layout(Probe(), rejected)

    def short(group, width):
        return linear_layout(group, width)[:-1]

    def signed(group, width):
        return linear_layout(group, width).astype(numpy.int32)

    for layout in (short, signed):
        with pytest.raises(MetalLaunchSweepError, match="no \\(64, 5\\) uint32"):
            observe_simd_layout(Probe(layout), ShapeAdmission((8, 8), HOST, False))
    with pytest.raises(MetalLaunchSweepError, match="must be a uint32 array"):
        words_sha256(numpy.zeros(5, dtype=numpy.int64))
