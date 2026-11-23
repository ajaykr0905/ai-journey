"""Tests that run the pinned flash attention text in the Triton CPU interpreter.

They need triton and torch, so they skip on a host without them. Triton has
no macOS wheel, so they skip on the development Mac and run in the Linux CI
job with triton 3.9.0, CPU torch and ``TRITON_INTERPRET=1``. The interpreter
runs on the CPU, so no test here measures or claims GPU behavior.
"""

import inspect
import unittest

import pytest

# pytest and the unittest discovery of the CI job ``test`` both skip a module
# that raises unittest.SkipTest. A pytest skip fails that discovery.
try:
    import triton
except ImportError:
    raise unittest.SkipTest("this host has no triton wheel") from None
try:
    import torch  # noqa: F401
except ImportError:
    raise unittest.SkipTest("the interpreter takes torch tensors") from None

from ai_journey import triton_flash_runtime  # noqa: E402
from ai_journey.triton_flash_config import PINNED_PLAN  # noqa: E402
from ai_journey.triton_flash_correctness import check_case  # noqa: E402
from ai_journey.triton_flash_kernel import (  # noqa: E402
    CONFIGS_BY_NAME,
    FLASH_CONFIGS,
    TRITON_SOURCE,
    TritonFlashError,
    source_sha256,
)
from ai_journey.triton_flash_reference import FAMILIES, AttentionCase  # noqa: E402
from ai_journey.triton_flash_runtime import (  # noqa: E402
    InterpreterBackend,
    source_filename,
)

SMALL = CONFIGS_BY_NAME["m16_n16_w1_s1"]


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
    assert source.startswith("@triton.jit\ndef flash_attention_forward(\n")
    assert source in TRITON_SOURCE
    filename = backend.kernel.rewrite().__code__.co_filename
    assert filename == source_filename(TRITON_SOURCE)


@pytest.mark.parametrize("index", range(len(PINNED_PLAN.cases)))
def test_every_declared_config_holds_both_families_on_each_pinned_case(backend, index):
    case = PINNED_PLAN.cases[index]

    record = check_case(backend, seed=81, index=index, case=case, configs=FLASH_CONFIGS)

    assert len(record.runs) == len(FLASH_CONFIGS) * len(FAMILIES)
    for run in record.runs:
        assert 0 <= run.output_bound_ratio <= 1
        assert 0 <= run.lse_bound_ratio <= 1


def changed_text(old, new):
    assert TRITON_SOURCE.count(old) == 1
    return TRITON_SOURCE.replace(old, new)


def test_a_text_without_its_store_mask_writes_into_the_guard(monkeypatch):
    unmasked = changed_text(
        "tl.store(o_ptrs, acc, mask=row_mask[:, None])", "tl.store(o_ptrs, acc)"
    )
    monkeypatch.setattr(triton_flash_runtime, "TRITON_SOURCE", unmasked)

    broken = InterpreterBackend()
    assert broken.source_sha256 == source_sha256(unmasked) != source_sha256()
    with pytest.raises(TritonFlashError, match="escaped its mask into the guard"):
        check_case(
            broken,
            seed=81,
            index=1,
            case=AttentionCase(2, 17, 16, True),
            configs=(SMALL,),
        )


def test_a_text_without_the_rescale_breaks_the_bound(monkeypatch):
    unscaled = changed_text(
        "alpha = tl.exp(m_i - m_new)", "alpha = tl.exp(m_new - m_new)"
    )
    monkeypatch.setattr(triton_flash_runtime, "TRITON_SOURCE", unscaled)

    with pytest.raises(TritonFlashError, match="times its bound"):
        check_case(
            InterpreterBackend(),
            seed=81,
            index=3,
            case=AttentionCase(1, 129, 64, True),
            configs=(SMALL,),
        )


def test_a_later_text_never_replaces_the_source_of_an_earlier_kernel(monkeypatch):
    pinned = InterpreterBackend()
    monkeypatch.setattr(
        triton_flash_runtime,
        "TRITON_SOURCE",
        changed_text("alpha = tl.exp(m_i - m_new)", "alpha = tl.exp(m_new - m_new)"),
    )
    InterpreterBackend()

    record = check_case(
        pinned,
        seed=81,
        index=3,
        case=AttentionCase(1, 129, 64, True),
        configs=(SMALL,),
    )
    assert all(run.output_bound_ratio <= 1 for run in record.runs)


def test_the_declared_configs_wrap_the_kernel_and_tune_nothing(backend):
    configs = [
        triton.Config(
            config.meta_parameters(),
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        for config in FLASH_CONFIGS
    ]

    tuned = triton.autotune(configs=configs, key=["N", "HEAD_DIM"])(backend.kernel)

    assert type(tuned).__name__ == "Autotuner"
    assert [item.kwargs for item in tuned.configs] == [
        config.meta_parameters() for config in FLASH_CONFIGS
    ]
    assert tuned.cache == {}
