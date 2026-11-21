"""Tests of the correctness harness with a NumPy stand-in for the interpreter.

``StandInBackend`` runs the kernel's arithmetic as the Triton 3.9.0
interpreter runs it: each program loads zero-filled tiles, and ``tl.dot`` is
``numpy.matmul(a, b, dtype=float32) + acc``. Its faults break one rule each,
and each fault must stop the run. A fake ``triton`` module tests the runtime
probe and the launcher on a host without triton.
"""

import os
import sys
import types

import numpy
import pytest

from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    error_bound,
    float64_reference,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
)
from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.triton_matmul_correctness import (
    EXACT,
    FAMILIES,
    NORMAL,
    check_case,
    exact_inputs,
    family_inputs,
    guard_elements,
    sentinel_buffer,
)
from ai_journey.triton_matmul_kernel import (
    AUTOTUNE_CONFIGS,
    CONFIGS_BY_NAME,
    TritonConfig,
    TritonMatmulError,
    k_blocks,
    program_tile,
    programs,
    tile_grid,
)
from ai_journey.triton_matmul_runtime import (
    INTERPRET_VARIABLE,
    InterpreterBackend,
    TritonMatmulUnavailableError,
)

SMALL = CONFIGS_BY_NAME["m64_n32_k32_g8_w2_s5"]
RAGGED = (MatmulShape(1, 1, 1), MatmulShape(17, 33, 9), MatmulShape(31, 47, 129))
STAND_IN_ENVIRONMENT = {
    "python": "3.11.0",
    "implementation": "CPython",
    "system": "Linux",
    "machine": "x86_64",
    "numpy": numpy.__version__,
    "torch": "none",
    "triton": "none",
    "triton_interpret": "1",
}


class StandInBackend:
    """The interpreter's arithmetic in NumPy, with one optional fault."""

    name = "injected"

    def __init__(self, fault=None):
        self.fault = fault
        self.environment = dict(STAND_IN_ENVIRONMENT)
        self.source_sha256 = None
        self.calls = 0

    def run(self, left, right, config, buffer):
        self.calls += 1
        m, k = left.shape
        n = right.shape[1]
        out = numpy.array(buffer, copy=True)
        rows, columns = tile_grid(MatmulShape(m, n, k), config)
        bm, bn, bk = config.block_m, config.block_n, config.block_k
        skip_last = self.fault == "skip_last_program"
        for pid in range(rows * columns - int(skip_last)):
            row, column = program_tile(pid, rows, columns, config.group_m)
            r0, c0 = row * bm, column * bn
            acc = numpy.zeros((bm, bn), dtype=numpy.float32)
            stop = k - (k % bk or bk) if self.fault == "skip_k_tail" and k > bk else k
            for k0 in range(0, stop, bk):
                a = numpy.zeros((bm, bk), dtype=numpy.float32)
                b = numpy.zeros((bk, bn), dtype=numpy.float32)
                block_a = left[r0 : r0 + bm, k0 : k0 + bk]
                block_b = right[k0 : k0 + bk, c0 : c0 + bn]
                a[: block_a.shape[0], : block_a.shape[1]] = block_a
                b[: block_b.shape[0], : block_b.shape[1]] = block_b
                acc = numpy.matmul(a, b, dtype=numpy.float32) + acc
            if self.fault == "no_row_mask":
                # Store every padded row: the rows past M fall in the guard.
                for i in range(bm):
                    stop_n = min(bn, n - c0)
                    start = (r0 + i) * n + c0
                    out[start : start + stop_n] = acc[i, :stop_n]
                continue
            view = out[: m * n].reshape(m, n)
            rr, cc = min(bm, m - r0), min(bn, n - c0)
            view[r0 : r0 + rr, c0 : c0 + cc] = acc[:rr, :cc]
        if self.fault == "one_ulp" and numpy.all(left == numpy.round(left)):
            out[0] = numpy.nextafter(out[0], numpy.float32(numpy.inf))
        if self.fault == "float64":
            return out.astype(numpy.float64)
        if self.fault == "short":
            return out[:-1]
        return out


def run_case(backend, shape, configs=(SMALL,), index=1, seed=74):
    return check_case(backend, seed=seed, index=index, shape=shape, configs=configs)


def test_the_normal_family_is_the_operands_of_the_day_76_cuda_contract():
    shape = MatmulShape(17, 33, 9)
    left, right = family_inputs(NORMAL, 74, 1, shape)
    day_74 = matmul_inputs(74, CORRECTNESS_STREAM, 1, shape)

    assert matrix_sha256(left) == matrix_sha256(day_74[0])
    assert matrix_sha256(right) == matrix_sha256(day_74[1])
    # The replay digest of case 1 in docs/gpu-tiled-cuda-contract.md.
    assert matrix_sha256(fma_replay(left, right)) == (
        "3b282a4121200a322b2e56328041b32cf50a6ffd2cefa1437e35ef6e8944795e"
    )


def test_the_exact_family_is_small_integers_whose_sums_are_exact():
    shape = MatmulShape(64, 48, 4096)
    left, right = exact_inputs(74, 6, shape)
    reference, magnitude = float64_reference(left, right)

    assert left.dtype == right.dtype == numpy.float32
    assert left.shape == (64, 4096) and right.shape == (4096, 48)
    for operand in (left, right):
        assert numpy.array_equal(operand, numpy.round(operand))
        assert operand.min() == -8 and operand.max() == 8
    assert magnitude.max() <= 64 * 4096 < 2**24
    assert numpy.array_equal(reference.astype(numpy.float32), reference)
    assert matrix_sha256(exact_inputs(74, 6, shape)[0]) == matrix_sha256(left)
    assert matrix_sha256(exact_inputs(75, 6, shape)[0]) != matrix_sha256(left)
    with pytest.raises(TritonMatmulError, match="family"):
        family_inputs("uniform", 74, 0, shape)
    with pytest.raises(TritonMatmulError, match="case index"):
        exact_inputs(74, 64, shape)


@pytest.mark.parametrize("config", AUTOTUNE_CONFIGS, ids=lambda item: item.name)
def test_the_guard_covers_every_index_that_an_unmasked_store_reaches(config):
    for shape in (*RAGGED, MatmulShape(1000, 999, 100), MatmulShape(3, 1000, 2048)):
        rows, columns = tile_grid(shape, config)
        last_row = rows * config.block_m - 1
        last_column = columns * config.block_n - 1
        reach = last_row * shape.n + last_column + 1

        assert guard_elements(shape, config) == reach - shape.m * shape.n
    assert guard_elements(MatmulShape(256, 256, 64), config) == 0


def test_the_sentinel_buffer_holds_the_day_73_quiet_nan_bits():
    buffer = sentinel_buffer(5)

    assert buffer.dtype == numpy.float32 and buffer.shape == (5,)
    assert set(buffer.view(numpy.uint32).tolist()) == {0x7FC00000}


def test_the_stand_in_passes_every_rule_for_every_config_and_family():
    backend = StandInBackend()

    for index, shape in enumerate(RAGGED):
        case = run_case(backend, shape, AUTOTUNE_CONFIGS, index=index)
        mapping = case.to_mapping()

        assert mapping["error_bound"] == error_bound(shape.k)
        assert [item["family"] for item in mapping["operands"]] == list(FAMILIES)
        assert [(run.config, run.family) for run in case.runs] == [
            (config.name, family) for config in AUTOTUNE_CONFIGS for family in FAMILIES
        ]
        for run, config in zip(case.runs[::2], AUTOTUNE_CONFIGS):
            assert run.programs == programs(shape, config)
            assert run.k_blocks == k_blocks(shape, config)
            assert run.guard_elements == guard_elements(shape, config)
        for run in case.runs:
            assert 0 <= run.bound_ratio <= 1
            assert run.bound_ratio == run.max_relative_error / case.bound
        exact = [run for run in case.runs if run.family == EXACT]
        assert {run.max_relative_error for run in exact} == {0.0}
        assert {run.rounded_reference_differences for run in exact} == {0}
    assert backend.calls == len(RAGGED) * len(AUTOTUNE_CONFIGS) * len(FAMILIES)


def test_a_record_names_the_buffer_digest_with_its_guard_tail():
    shape = MatmulShape(17, 33, 9)
    case = run_case(StandInBackend(), shape)
    exact_run = case.runs[1]
    left, right = exact_inputs(74, 1, shape)
    reference, _ = float64_reference(left, right)
    tail = sentinel_buffer(guard_elements(shape, SMALL))

    assert exact_run.family == EXACT
    assert exact_run.buffer_sha256 == matrix_sha256(
        numpy.concatenate([reference.astype(numpy.float32).reshape(-1), tail])
    )
    assert case.operands[1].reference_sha256 == matrix_sha256(reference)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("no_row_mask", "escaped its mask into the guard"),
        ("skip_last_program", "not finite"),
        ("skip_k_tail", "exceeds the bound"),
        ("one_ulp", "differ from the exact float64 sum"),
        ("float64", "no float32 buffer"),
        ("short", "no float32 buffer"),
    ],
)
def test_each_broken_rule_stops_the_run(fault, message):
    with pytest.raises(TritonMatmulError, match=message):
        run_case(StandInBackend(fault), MatmulShape(31, 47, 129))


def test_check_case_refuses_an_empty_config_list():
    with pytest.raises(TritonMatmulError, match="non-empty tuple"):
        run_case(StandInBackend(), RAGGED[0], configs=())
    with pytest.raises(TypeError, match="TritonConfig"):
        run_case(StandInBackend(), RAGGED[0], configs=(SMALL.to_mapping(),))


def keep_environment(monkeypatch):
    """Let a test unset the variable; teardown puts back its first value."""

    monkeypatch.setenv(INTERPRET_VARIABLE, "placeholder")
    monkeypatch.delenv(INTERPRET_VARIABLE)


def fake_triton(monkeypatch, jit):
    language = types.ModuleType("triton.language")
    language.constexpr = object
    triton = types.ModuleType("triton")
    triton.language = language
    triton.jit = jit
    triton.cdiv = lambda numerator, denominator: -(-numerator // denominator)
    triton.__version__ = "0.0.fake"
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)


class InterpretedFunction:
    """A stand-in with the name and the launch syntax of the interpreter type."""

    launches = []

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            InterpretedFunction.launches.append((grid, args, kwargs))

        return launch


def interpreted_when_enabled(fn):
    enabled = os.environ.get(INTERPRET_VARIABLE) == "1"
    return InterpretedFunction(fn) if enabled else fn


def test_a_host_without_triton_or_torch_is_named_and_left_unchanged(monkeypatch):
    keep_environment(monkeypatch)
    monkeypatch.setitem(sys.modules, "triton", None)

    with pytest.raises(TritonMatmulUnavailableError) as stopped:
        InterpreterBackend()
    assert stopped.value.reason == "triton_not_importable"
    assert INTERPRET_VARIABLE not in os.environ

    fake_triton(monkeypatch, interpreted_when_enabled)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(TritonMatmulUnavailableError) as stopped:
        InterpreterBackend()
    assert stopped.value.reason == "torch_not_importable"
    assert str(stopped.value).endswith("(torch_not_importable)")
    assert INTERPRET_VARIABLE not in os.environ


def test_the_backend_refuses_a_jit_that_is_not_the_interpreter(monkeypatch):
    keep_environment(monkeypatch)
    fake_triton(monkeypatch, interpreted_when_enabled)
    monkeypatch.setenv(INTERPRET_VARIABLE, "0")

    with pytest.raises(TritonMatmulUnavailableError) as stopped:
        InterpreterBackend()
    assert stopped.value.reason == "interpreter_disabled"
    assert "TRITON_INTERPRET='0' gives function" in str(stopped.value)
    assert os.environ[INTERPRET_VARIABLE] == "0"


def test_the_backend_sets_the_interpreter_and_launches_the_text(monkeypatch):
    keep_environment(monkeypatch)
    fake_triton(monkeypatch, interpreted_when_enabled)
    InterpretedFunction.launches.clear()

    backend = InterpreterBackend()
    left = numpy.ones((17, 9), dtype=numpy.float32)
    right = numpy.ones((9, 33), dtype=numpy.float32)
    buffer = sentinel_buffer(17 * 33 + guard_elements(MatmulShape(17, 33, 9), SMALL))

    assert os.environ[INTERPRET_VARIABLE] == "1"
    assert backend.environment["triton"] == "0.0.fake"
    assert backend.environment["triton_interpret"] == "1"
    assert backend.run(left, right, SMALL, buffer) is buffer
    ((grid, args, kwargs),) = InterpretedFunction.launches
    assert grid == (1 * 2,)
    assert [type(item).__name__ for item in args[:3]] == ["Tensor"] * 3
    assert args[3:] == (17, 33, 9, 9, 1, 33, 1, 33, 1)
    assert kwargs == {
        "BLOCK_M": 64,
        "BLOCK_N": 32,
        "BLOCK_K": 32,
        "GROUP_M": 8,
        "num_warps": 2,
        "num_stages": 5,
    }


def crashing_jit(fn):
    class InterpretedFunction:
        """An interpreted stand-in whose launch fails, as InterpreterError does."""

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise RuntimeError("index out of range")

            return launch

    return InterpretedFunction()


def test_a_crash_inside_the_interpreter_is_named_as_a_lab_error(monkeypatch):
    keep_environment(monkeypatch)
    fake_triton(monkeypatch, crashing_jit)
    backend = InterpreterBackend()
    buffer = sentinel_buffer(4)
    left = numpy.ones((1, 1), dtype=numpy.float32)

    with pytest.raises(TritonMatmulError, match="interpreter failed on m64_n32"):
        backend.run(left, left, SMALL, buffer)


def test_the_unavailable_error_accepts_only_a_known_reason():
    with pytest.raises(ValueError, match="reason"):
        TritonMatmulUnavailableError("no_gpu", "nothing")
    assert TritonConfig(16, 16, 16, 1, 1, 1).name == "m16_n16_k16_g1_w1_s1"
