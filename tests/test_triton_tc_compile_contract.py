"""Compile the four variants with the real triton and check the class rule.

This module needs triton, so it raises ``unittest.SkipTest`` at import on a
host without it: pytest and the unittest discovery of the CI job ``test``
both count that as a skip. Triton has no macOS wheel, so the module skips on
the development Mac. A Linux CI job with triton 3.9.0 runs it for real. That
job did not run when this text was written. Triton compiles for the declared
targets sm_80 and sm_90; no GPU runs, and no PTX or cubin is executed.

The rule states what Triton 3.9 does, so one test requires the declared
triton 3.9.0 release; another release fails it instead of skipping.

When ``TRITON_TC_DUMP_DIR`` names a folder, the module writes there every
TTGIR and PTX text, the dump index and ``summary.json`` with the rule rows,
so the CI job can keep the compiler output as an artifact. It writes the
texts before it applies the rule, so a failed rule still leaves them.
"""

import os
import unittest
from pathlib import Path

# pytest and the unittest discovery of the CI job ``test`` both skip a module
# that raises unittest.SkipTest. A pytest skip fails that discovery.
try:
    import triton
except ImportError:
    raise unittest.SkipTest("this host has no triton wheel") from None

import pytest  # noqa: E402

from ai_journey.kernel_launch_io import canonical_json_bytes, write_atomic  # noqa: E402
from ai_journey.triton_tc_compile import (  # noqa: E402
    COMPILE_ORDER,
    SUMMARY_NAME,
    CompilerBackend,
    compile_all,
    load_dump,
    write_dump,
)
from ai_journey.triton_tc_kernel import (  # noqa: E402
    PINNED_SHA256,
    TC_CONFIG,
    VARIANTS,
    VARIANTS_BY_NAME,
    TritonTcError,
)
from ai_journey.triton_matmul_runtime import PYPI_VERSION  # noqa: E402
from ai_journey.triton_tc_rule import check_compile, rule_summary  # noqa: E402

DUMP_VARIABLE = "TRITON_TC_DUMP_DIR"
TENSOR_CORE = [variant.name for variant in VARIANTS if variant.tensor_cores]


@pytest.fixture(scope="module")
def compiled():
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("TRITON_INTERPRET", raising=False)
        backend = CompilerBackend()
        results = compile_all(backend)
    folder = os.environ.get(DUMP_VARIABLE)
    if folder:
        write_dump(Path(folder), results, backend.environment)
    return backend, results


def checked_row(item):
    """Apply the rule; keep an analysis error as a refused row of the summary."""

    try:
        return check_compile(item)
    except TritonTcError as error:
        return {
            "variant": item.variant,
            "target": item.target,
            "observed": None,
            "accepted": False,
            "note": None,
            "checks": {},
            "error": str(error),
        }


@pytest.fixture(scope="module")
def rows(compiled):
    backend, results = compiled
    checked = [checked_row(item) for item in results]
    folder = os.environ.get(DUMP_VARIABLE)
    if folder:
        summary = {
            "kernel_sha256": PINNED_SHA256,
            "config": TC_CONFIG.to_mapping(),
            "environment": backend.environment,
            "rule": rule_summary(checked),
            "rows": checked,
            "gpu_executed": False,
            "ptx_executed": False,
        }
        write_atomic(Path(folder) / SUMMARY_NAME, canonical_json_bytes(summary) + b"\n")
    return {(row["variant"], row["target"]): row for row in checked}


def row_of(rows, variant, target):
    row = rows[(variant, target)]
    assert "error" not in row, row.get("error")
    return row


def test_the_compiler_is_the_declared_triton_release():
    """The rule states what Triton 3.9 does, so another release fails here.

    A skip would hide a CI job that installed another triton, for example the
    one that a CUDA torch wheel pulls in, so the version must match.
    """

    assert PYPI_VERSION == "3.9.0"
    assert triton.__version__ == PYPI_VERSION


def test_the_real_triton_compiles_every_variant_for_both_targets(compiled):
    backend, results = compiled

    assert backend.environment["triton"] == triton.__version__
    assert backend.source_sha256 == PINNED_SHA256
    assert [item.key for item in results] == list(COMPILE_ORDER)
    for item in results:
        assert (item.num_warps, item.num_stages) == (4, 3), item.key
        assert item.shared_bytes > 0, item.key


def test_fp32_ieee_uses_scalar_fma_and_no_mma_on_both_targets(rows):
    for target in ("sm_80", "sm_90"):
        row = row_of(rows, "fp32-ieee", target)
        assert row["observed"] == "scalar_fma", row["ptx"]["classes"]
        assert row["ptx"]["classes"]["mma_sync"] == 0
        assert row["ptx"]["classes"]["wgmma_mma_async"] == 0
        assert row["ptx"]["classes"]["fma_rn_f32"] > 0
        assert row["accepted"], row["checks"]


def test_the_tensor_core_variants_use_mma_sync_on_sm_80(rows):
    for name in TENSOR_CORE:
        row = row_of(rows, name, "sm_80")
        assert row["observed"] == "mma_sync", (name, row["ptx"]["classes"])
        forms = row["ptx"]["mma_forms"]
        assert {form["a_type"] for form in forms} == {
            VARIANTS_BY_NAME[name].mma_operand_type
        }
        assert row["accepted"], (name, row["checks"])


def test_the_tensor_core_variants_use_a_tensor_core_class_on_sm_90(rows):
    """wgmma is expected; mma.sync is accepted and recorded, not a failure."""

    for name in TENSOR_CORE:
        row = row_of(rows, name, "sm_90")
        assert row["observed"] in ("wgmma", "mma_sync"), (name, row["observed"])
        assert (row["note"] is None) == (row["observed"] == "wgmma"), name
        assert row["ptx"]["target"].startswith("sm_90"), name
        assert row["accepted"], (name, row["checks"])


def test_every_compile_passes_every_check_of_the_rule(rows):
    failed = {
        key: row.get("error")
        or [name for name, passed in row["checks"].items() if not passed]
        for key, row in rows.items()
        if not row["accepted"]
    }
    assert failed == {}
    assert rule_summary(list(rows.values()))["all_accepted"] is True


def test_the_compiler_output_round_trips_through_a_dump_folder(compiled, tmp_path):
    backend, results = compiled
    index = write_dump(tmp_path, results, backend.environment)

    read_index, loaded = load_dump(tmp_path)
    assert loaded == results
    assert read_index == index
