import itertools

import pytest

from ai_journey.kernel_launch_buffer import values_sha256
from ai_journey.kernel_launch_config import Dim3, KernelLaunchError, launch_2d
from ai_journey.kernel_launch_kernels import (
    GRAYSCALE_WEIGHTS_PER_10000,
    luma_u8,
    run_rgb_to_grayscale,
    run_vector_add,
)
from ai_journey.kernel_launch_reference import (
    compare_exact,
    reference_rgb_to_grayscale,
    reference_vector_add,
    rgb_image,
    vector_add_inputs,
)
from ai_journey.kernel_launch_simulator import LaunchOrder
from test_kernel_launch_reference import tie_triples

ORDERS = [LaunchOrder.canonical(), *(LaunchOrder.permuted(seed) for seed in (1, 2))]


def test_luma_integer_rule_matches_the_exact_reference_rule():
    assert GRAYSCALE_WEIGHTS_PER_10000 == (2989, 5870, 1140)
    assert luma_u8(0, 0, 250) == 29
    assert luma_u8(0, 36, 12) == 23
    # (2989 + 5870 + 1140) * 255 + 5000 = 2,554,745: the clamp never binds.
    assert luma_u8(255, 255, 255) == 255
    samples = tie_triples() + list(
        itertools.product((0, 1, 127, 128, 254, 255), repeat=3)
    )
    flat = [channel for triple in samples for channel in triple]
    assert [luma_u8(*triple) for triple in samples] == list(
        reference_rgb_to_grayscale(flat, len(samples), 1)
    )


def test_guarded_vector_add_rounds_like_float32_and_idles_the_tail():
    run = run_vector_add(
        [1.0, 1.0 + 2**-23, 0.5, 3.0, -1.0],
        [2**-24, 2**-24, 0.25, 4.0, 1.0],
        threads_per_block=4,
    )

    assert run.output == (1.0, 1.0 + 2**-22, 0.75, 7.0, 0.0)
    assert run.output_sha256 == values_sha256("float32", run.output)
    assert run.result.config.grid == Dim3(2)
    assert (run.result.threads_run, run.useful_threads, run.idle_threads) == (8, 5, 3)
    assert run.result.log.counts() == {
        "reads": 10,
        "writes": 5,
        "out_of_bounds_reads": 0,
        "out_of_bounds_writes": 0,
        "write_conflicts": 0,
        "read_write_conflicts": 0,
    }


def test_guarded_grayscale_checks_both_axes_of_a_partial_block():
    image = [0, 0, 250, 0, 36, 12, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0]
    run = run_rgb_to_grayscale(image, width=3, height=2, block=(2, 2))
    tiny = run_rgb_to_grayscale(rgb_image(5, 5, 3), width=5, height=3, block=(16, 16))

    assert run.output == (29, 23, 255, 0, 76, 150)
    assert run.result.config == launch_2d(3, 2, 2, 2)
    assert (run.result.threads_run, run.useful_threads, run.idle_threads) == (8, 6, 2)
    assert (tiny.result.threads_run, tiny.useful_threads) == (256, 15)
    assert tiny.result.log.memory_safe and tiny.result.log.race_free


@pytest.mark.parametrize("n", [1000, 1024, 1, 257])
def test_vector_add_matches_the_reference_bit_for_bit_in_every_order(n):
    a, b = vector_add_inputs(73, n)
    expected = reference_vector_add(a, b)
    digests = set()
    for order in ORDERS:
        run = run_vector_add(a, b, threads_per_block=256, order=order)
        assert compare_exact("float32", expected, run.output).equal
        assert run.result.log.memory_safe and run.result.log.race_free
        assert run.useful_threads == n
        assert run.idle_threads == run.result.config.launched_threads - n
        digests.add(run.output_sha256)
    assert digests == {values_sha256("float32", expected)}


@pytest.mark.parametrize(("width", "height"), [(40, 30), (48, 32), (17, 1), (1, 33)])
def test_grayscale_matches_the_reference_in_every_order(width, height):
    image = rgb_image(73, width, height)
    expected = reference_rgb_to_grayscale(image, width, height)
    digests = set()
    for order in ORDERS:
        run = run_rgb_to_grayscale(
            image, width=width, height=height, block=(16, 16), order=order
        )
        assert compare_exact("uint8", expected, run.output).equal
        assert run.result.log.memory_safe and run.result.log.race_free
        assert run.useful_threads == width * height
        assert run.result.log.counts()["reads"] == 3 * width * height
        digests.add(run.output_sha256)
    assert digests == {values_sha256("uint8", expected)}


def test_run_helpers_reject_mismatched_shapes():
    with pytest.raises(KernelLaunchError, match="one length"):
        run_vector_add([0.5], [0.5, 0.5], threads_per_block=32)
    with pytest.raises(KernelLaunchError, match="3 \\* width \\* height"):
        run_rgb_to_grayscale([0] * 5, width=1, height=2, block=(16, 16))
    with pytest.raises(KernelLaunchError, match="block_x, block_y"):
        run_rgb_to_grayscale([0] * 3, width=1, height=1, block=[16, 16])
