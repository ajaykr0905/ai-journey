import itertools
import json
import math
import platform

import numpy
import pytest
import torch

from ai_journey.memory_access_baseline import (
    MatmulCase,
    MatmulChecker,
    cpu_environment,
    inputs_sha256,
    matmul_inputs,
    measure_matmul,
    rounding_tolerance,
)
from ai_journey.memory_access_pattern import MemoryAccessError
from ai_journey.memory_access_tiling import MatmulShape


def stepping_clock(step: int = 250):
    return itertools.count(1_000, step).__next__


def test_seeded_inputs_are_float32_and_fingerprinted_by_content():
    shape = MatmulShape(6, 5, 4)
    left, right = matmul_inputs(shape, 74)
    again = matmul_inputs(shape, 74)

    assert left.dtype == right.dtype == numpy.float32
    assert (left.shape, right.shape) == ((6, 4), (4, 5))
    assert inputs_sha256(left, right) == inputs_sha256(*again)
    assert inputs_sha256(left, right) != inputs_sha256(*matmul_inputs(shape, 75))
    with pytest.raises(TypeError, match="float32"):
        inputs_sha256(left.astype(numpy.float64), right)


def test_rounding_tolerance_is_the_float32_plus_float64_gamma_bound():
    inner = 512
    expected = inner * 2.0**-24 / (1 - inner * 2.0**-24) + inner * 2.0**-53 / (
        1 - inner * 2.0**-53
    )

    assert rounding_tolerance(inner) == expected
    assert rounding_tolerance(1) < rounding_tolerance(1024)
    with pytest.raises(MemoryAccessError, match="inner"):
        rounding_tolerance(0)


def test_checker_accepts_numpy_and_torch_float32_results():
    left, right = matmul_inputs(MatmulShape(16, 12, 64), 3)
    checker = MatmulChecker(left, right)

    checker(left @ right)
    checker(torch.matmul(torch.from_numpy(left), torch.from_numpy(right)))
    check = checker.result()

    assert check.checked_results == 2
    assert 0 <= check.max_bound_ratio < 1
    assert check.max_abs_error > 0
    assert check.tolerance == rounding_tolerance(64)
    assert check.to_mapping()["reference"] == "float64"


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda value: value + numpy.float32(1e-2), "rounding bound"),
        (lambda value: value.astype(numpy.float64), "float32 array"),
        (lambda value: value[:, :-1], "wrong shape"),
        (lambda value: numpy.full_like(value, numpy.nan), "not finite"),
        (lambda value: value.tolist(), "float32 array"),
        (
            lambda value: torch.from_numpy(value).to(torch.float64),
            "CPU float32 tensor",
        ),
    ],
)
def test_checker_rejects_wrong_or_imprecise_results(corrupt, message):
    left, right = matmul_inputs(MatmulShape(8, 8, 32), 5)
    checker = MatmulChecker(left, right)

    with pytest.raises(MemoryAccessError, match=message):
        checker(corrupt(left @ right))
    with pytest.raises(MemoryAccessError, match="no matmul result"):
        checker.result()


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_measurement_uses_the_injected_clock_and_checks_every_result(backend):
    case = MatmulCase(backend, MatmulShape(24, 16, 32))
    measurement = measure_matmul(
        case, warmup=2, repeats=5, seed=74, clock=stepping_clock(250)
    )

    assert measurement.samples_ns == (250,) * 5
    assert measurement.timing.median_ns == 250
    assert measurement.timing.iqr_ns == 0
    assert measurement.flops == 2 * 24 * 16 * 32
    assert measurement.median_gflops_per_second == measurement.flops / 250
    assert measurement.best_gflops_per_second == measurement.flops / 250
    assert measurement.correctness.checked_results == 7
    assert measurement.correctness.max_bound_ratio < 1
    mapping = measurement.to_mapping()
    assert mapping["device"] == "cpu"
    assert (mapping["backend"], mapping["dtype"]) == (backend, "float32")
    assert (mapping["m"], mapping["n"], mapping["k"]) == (24, 16, 32)
    assert mapping["samples_ns"] == [250] * 5
    assert mapping["input_sha256"] == inputs_sha256(*matmul_inputs(case.shape, 74))


def test_one_real_cpu_timing_is_positive_and_finite():
    measurement = measure_matmul(
        MatmulCase("numpy", MatmulShape(32, 32, 32)), warmup=1, repeats=3, seed=74
    )

    assert all(sample > 0 for sample in measurement.samples_ns)
    assert math.isfinite(measurement.median_gflops_per_second)
    assert measurement.median_gflops_per_second > 0


def test_environment_record_names_runtime_versions_without_host_identity():
    environment = cpu_environment()

    assert set(environment) == {
        "python_version",
        "python_implementation",
        "numpy_version",
        "torch_version",
        "platform_system",
        "platform_machine",
        "cpu_count",
        "torch_threads",
        "torch_float32_matmul_precision",
    }
    assert environment["numpy_version"] == numpy.__version__
    assert environment["torch_threads"] == torch.get_num_threads()
    serialized = json.dumps(environment)
    hostname = platform.node()
    if len(hostname) >= 4:
        assert hostname not in serialized


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda: MatmulCase("cupy", MatmulShape(1, 1, 1)),
            MemoryAccessError,
            "backend",
        ),
        (
            lambda: MatmulCase("numpy", MatmulShape(1025, 1, 1)),
            MemoryAccessError,
            "exceed",
        ),
        (lambda: MatmulCase("torch", (1, 1, 1)), TypeError, "MatmulShape"),
        (lambda: matmul_inputs(MatmulShape(1, 1, 1), -1), MemoryAccessError, "seed"),
        (lambda: matmul_inputs(MatmulShape(1, 1, 1), 1.0), TypeError, "seed"),
        (
            lambda: measure_matmul("case", warmup=0, repeats=1, seed=0),
            TypeError,
            "MatmulCase",
        ),
        (
            lambda: measure_matmul(
                MatmulCase("numpy", MatmulShape(2, 2, 2)), warmup=0, repeats=0, seed=0
            ),
            MemoryAccessError,
            "repeats",
        ),
    ],
)
def test_baseline_contracts_are_bounded(call, error, message):
    with pytest.raises(error, match=message):
        call()
