"""Tests that run the pinned text in the Triton CPU interpreter.

They need triton and torch, so they skip on a host without them. Triton has
no macOS wheel. A linux/aarch64 python:3.11-slim container with triton 3.9.0
and CPU torch 2.15.0 runs them, with TRITON_INTERPRET=1. The interpreter runs
on the CPU, so no test here measures or claims GPU behavior.
"""

import inspect
import unittest

import numpy
import pytest

# pytest and the unittest discovery of the CI job ``test`` both skip a module
# that raises unittest.SkipTest. A pytest skip fails that discovery.
try:
    import triton

    # A jit function finds triton.language in its module globals.
    import triton.language as tl
except ImportError:
    raise unittest.SkipTest("this host has no triton wheel") from None
try:
    import torch
except ImportError:
    raise unittest.SkipTest("the interpreter takes torch tensors") from None

from ai_journey import triton_matmul_runtime  # noqa: E402
from ai_journey.memory_access_tiling import MatmulShape  # noqa: E402
from ai_journey.triton_matmul_correctness import (  # noqa: E402
    EXACT,
    FAMILIES,
    check_case,
    family_inputs,
    guard_elements,
    sentinel_buffer,
)
from ai_journey.triton_matmul_kernel import (  # noqa: E402
    AUTOTUNE_CONFIGS,
    AUTOTUNE_KEY,
    CONFIGS_BY_NAME,
    TRITON_SOURCE,
    TritonMatmulError,
    source_sha256,
)
from ai_journey.triton_matmul_runtime import (  # noqa: E402
    InterpreterBackend,
    source_filename,
)
from test_triton_matmul_correctness import StandInBackend  # noqa: E402

SHAPES = (
    MatmulShape(1, 1, 1),
    MatmulShape(17, 33, 9),
    MatmulShape(32, 32, 16),
    MatmulShape(31, 47, 129),
)


@pytest.fixture(scope="module")
def backend():
    return InterpreterBackend()


def test_the_backend_runs_the_pinned_text_as_interpreted_functions(backend):
    source = inspect.getsource(backend.kernel.fn)

    assert type(backend.kernel).__name__ == "InterpretedFunction"
    assert backend.source_sha256 == source_sha256()
    assert backend.environment["triton"] == triton.__version__
    assert backend.environment["triton_interpret"] == "1"
    assert backend.environment["system"] == "Linux"
    assert source.startswith("@triton.jit\ndef matmul_kernel(\n")
    assert source in TRITON_SOURCE
    # The interpreter rewrote the kernel from the text that linecache holds.
    filename = backend.kernel.rewrite().__code__.co_filename
    assert filename == source_filename(TRITON_SOURCE)


@pytest.mark.parametrize("index", range(len(SHAPES)))
def test_every_declared_config_holds_both_families_on_a_ragged_case(backend, index):
    shape = SHAPES[index]

    case = check_case(
        backend, seed=74, index=index, shape=shape, configs=AUTOTUNE_CONFIGS
    )

    assert len(case.runs) == len(AUTOTUNE_CONFIGS) * len(FAMILIES)
    for run in case.runs:
        assert 0 <= run.bound_ratio <= 1
        if run.family == EXACT:
            assert run.max_relative_error == 0.0
            assert run.rounded_reference_differences == 0


@pytest.mark.parametrize("config", AUTOTUNE_CONFIGS[::2], ids=lambda item: item.name)
def test_the_numpy_stand_in_rebuilds_the_interpreter_bits(backend, config):
    stand_in = StandInBackend()

    for index, shape in enumerate(
        (MatmulShape(31, 47, 129), MatmulShape(129, 65, 300))
    ):
        for family in FAMILIES:
            left, right = family_inputs(family, 74, index, shape)
            count = shape.m * shape.n + guard_elements(shape, config)
            real = backend.run(left, right, config, sentinel_buffer(count))
            emulated = stand_in.run(left, right, config, sentinel_buffer(count))

            assert numpy.array_equal(
                real.view(numpy.uint32), emulated.view(numpy.uint32)
            )


def unmasked_text():
    pinned = "    tl.store(c_ptrs, acc, mask=c_mask)\n"
    assert TRITON_SOURCE.count(pinned) == 1
    return TRITON_SOURCE.replace(pinned, "    tl.store(c_ptrs, acc)\n")


def test_a_text_without_its_store_mask_writes_into_the_guard(monkeypatch):
    unmasked = unmasked_text()
    monkeypatch.setattr(triton_matmul_runtime, "TRITON_SOURCE", unmasked)

    broken = InterpreterBackend()
    assert broken.source_sha256 == source_sha256(unmasked) != source_sha256()
    with pytest.raises(TritonMatmulError, match="escaped its mask into the guard"):
        check_case(
            broken,
            seed=74,
            index=3,
            shape=MatmulShape(31, 47, 129),
            configs=(CONFIGS_BY_NAME["m64_n32_k32_g8_w2_s5"],),
        )


def test_a_later_text_never_replaces_the_source_of_an_earlier_kernel(monkeypatch):
    # The interpreter reads the kernel source at the first launch, so a
    # backend compiled before another text must still run its own text.
    pinned = InterpreterBackend()
    monkeypatch.setattr(triton_matmul_runtime, "TRITON_SOURCE", unmasked_text())
    InterpreterBackend()

    case = check_case(
        pinned,
        seed=74,
        index=3,
        shape=MatmulShape(31, 47, 129),
        configs=(CONFIGS_BY_NAME["m64_n32_k32_g8_w2_s5"],),
    )
    assert [run.rounded_reference_differences for run in case.runs][1] == 0


def test_the_declared_configs_wrap_the_kernel_and_tune_nothing(backend):
    configs = [
        triton.Config(
            config.meta_parameters(),
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        for config in AUTOTUNE_CONFIGS
    ]

    tuned = triton.autotune(configs=configs, key=list(AUTOTUNE_KEY))(backend.kernel)

    assert type(tuned).__name__ == "Autotuner"
    assert tuned.keys == ["M", "N", "K"]
    assert [
        (item.kwargs, item.num_warps, item.num_stages) for item in tuned.configs
    ] == [
        (config.meta_parameters(), config.num_warps, config.num_stages)
        for config in AUTOTUNE_CONFIGS
    ]
    assert tuned.cache == {}


@triton.jit
def copy_tile(ptr, M, N, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.make_tensor_descriptor(
        ptr, shape=[M, N], strides=[STRIDE, 1], block_shape=[BLOCK, BLOCK]
    )
    tile.store([0, 0], tile.load([0, 0]) + 1)


def test_block_pointers_are_gone_and_a_descriptor_refuses_a_ragged_stride():
    with pytest.raises(NotImplementedError, match="tensor descriptor API"):
        tl.make_block_ptr(
            None, (1, 1), (1, 1), (0, 0), (16, 16), (1, 0), _semantic=object()
        )

    aligned = torch.zeros(17 * 32, dtype=torch.float32)
    copy_tile[(1,)](aligned, 17, 32, STRIDE=32, BLOCK=16)
    assert float(aligned[0]) == 1.0
    # A row of N = 33 float32 values spans 132 bytes, not a multiple of 16.
    with pytest.raises(Exception, match="16-byte aligned.*33 \\* 4 = 132 bytes"):
        copy_tile[(1,)](torch.zeros(17 * 33), 17, 33, STRIDE=33, BLOCK=16)
