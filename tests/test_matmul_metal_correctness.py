import numpy
import pytest

from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.kernel_metal_reference import float32_sentinels
from ai_journey.matmul_metal_correctness import IMPLEMENTATIONS, check_case
from ai_journey.matmul_metal_mapping import MatmulMetalError, matmul_shape
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    error_bound,
    float64_reference,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
)
from test_kernel_metal_runtime import requires_metal
from test_matmul_metal_recorded import RecordedBackend
from test_matmul_metal_runtime import matmul_runtime

GROUP = (32, 8)


def run_case(backend, index=1, shape=(17, 33, 9)):
    return check_case(
        backend,
        seed=74,
        index=index,
        shape=matmul_shape(*shape),
        group_size=GROUP,
    )


def test_a_ragged_case_records_every_implementation_and_its_digests():
    case = run_case(RecordedBackend())
    mapping = case.to_mapping()
    shape = matmul_shape(17, 33, 9)
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 1, shape)
    replay = fma_replay(left, right)
    reference, _ = float64_reference(left, right)

    assert [(o["implementation"], o["device"]) for o in mapping["outputs"]] == [
        ("matmul_x_columns", "mps"),
        ("matmul_x_rows", "mps"),
        ("torch_matmul", "mps"),
        ("torch_matmul", "cpu"),
    ]
    assert list(IMPLEMENTATIONS) == [
        (o["implementation"], o["device"]) for o in mapping["outputs"]
    ]
    assert (mapping["m"], mapping["n"], mapping["k"]) == (17, 33, 9)
    assert mapping["a_sha256"] == matrix_sha256(left)
    assert mapping["b_sha256"] == matrix_sha256(right)
    assert mapping["reference_sha256"] == matrix_sha256(reference)
    assert mapping["replay_sha256"] == matrix_sha256(replay)
    assert mapping["bound"] == error_bound(9) == case.bound
    pads = {"x_columns": 64 * 24 - 17 * 33, "x_rows": 32 * 40 - 17 * 33}
    for (mapping_name, launch), output in zip(
        mapping["launches"].items(), mapping["outputs"]
    ):
        assert launch["padding_threads"] == pads[mapping_name]
        assert output["guard_elements"] == pads[mapping_name]
        tail = float32_sentinels(pads[mapping_name])
        rebuilt = numpy.concatenate([replay.reshape(-1), tail])
        assert output["buffer_sha256"] == matrix_sha256(rebuilt)
        assert output["differing_elements"] == 0
    for output in mapping["outputs"]:
        assert 0 < output["max_relative_error"] <= mapping["bound"]
        assert output["bound_ratio"] == output["max_relative_error"] / mapping["bound"]
    assert case.max_relative_error == max(
        o["max_relative_error"] for o in mapping["outputs"]
    )


class OneUlpOff(RecordedBackend):
    def run(self, name, left, right, launch):
        buffer = super().run(name, left, right, launch)
        if name == "matmul_x_rows":
            buffer[5] = numpy.nextafter(buffer[5], numpy.float32(numpy.inf))
        return buffer


class WritesTheTail(RecordedBackend):
    def run(self, name, left, right, launch):
        buffer = super().run(name, left, right, launch)
        buffer[-1] = 0.0
        return buffer


class TorchOffByMuch(RecordedBackend):
    def torch_matmul(self, left, right, device):
        product = left @ right
        if device == "cpu":
            product[2, 3] += 1.0
        return product


class TorchTransposed(RecordedBackend):
    def torch_matmul(self, left, right, device):
        return (left @ right).T.copy()


class TorchOneUlpFurther(RecordedBackend):
    """torch.matmul returns the replay with its worst element one ulp further out.

    That output keeps within the bound for K > 1, has other bits than the
    replay in exactly one element, and has a larger error than the kernels.
    """

    def torch_matmul(self, left, right, device):
        product = fma_replay(left, right)
        reference, magnitude = float64_reference(left, right)
        error = numpy.abs(product - reference) / magnitude
        worst = numpy.unravel_index(numpy.argmax(error), error.shape)
        outward = numpy.sign(product[worst] - reference[worst]) * numpy.inf
        product[worst] = numpy.nextafter(product[worst], numpy.float32(outward))
        return product


def test_torch_bits_that_differ_from_the_replay_are_counted():
    case = run_case(TorchOneUlpFurther())
    outputs = case.to_mapping()["outputs"]

    assert [output["differing_elements"] for output in outputs] == [0, 0, 1, 1]
    assert outputs[2]["buffer_sha256"] != case.replay_sha256
    assert outputs[2]["max_relative_error"] > outputs[0]["max_relative_error"]
    assert outputs[2]["max_relative_error"] <= case.bound


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (OneUlpOff(), "matmul_x_rows differs from the CPU replay of its fma chain"),
        (WritesTheTail(), "a padding thread of matmul_x_columns wrote into the guard"),
        (TorchOffByMuch(), "torch_matmul on cpu has a relative error of"),
        (TorchTransposed(), r"no float32 output of shape \(17, 33\)"),
    ],
)
def test_a_broken_rule_stops_the_case_with_its_reason(backend, message):
    with pytest.raises(MatmulMetalError, match=message):
        run_case(backend)


def test_cases_are_bounded_and_shapes_checked_first():
    with pytest.raises(KernelMetalError, match="case index must be in"):
        run_case(RecordedBackend(), index=64)
    with pytest.raises(TypeError, match="MatmulShape"):
        check_case(
            RecordedBackend(), seed=74, index=0, shape=(4, 4, 4), group_size=GROUP
        )


@requires_metal
@pytest.mark.parametrize("shape", [(1, 1, 1), (33, 17, 65), (65, 63, 1000)])
def test_this_gpu_meets_both_rules_on_ragged_and_non_square_cases(shape):
    case = run_case(matmul_runtime(), index=2, shape=shape)
    mapping = case.to_mapping()

    naive = mapping["outputs"][:2]
    assert {output["differing_elements"] for output in naive} == {0}
    assert naive[0]["max_relative_error"] == naive[1]["max_relative_error"]
    for output in mapping["outputs"]:
        assert output["max_relative_error"] <= mapping["bound"]
