import copy
import functools

import numpy
import pytest

# The report records whether torch was imported before its warning capture.
# Import it here, so the answer is the same on macOS and on the Linux CI job.
import torch  # noqa: F401

from ai_journey.memory_access_report import seal_report
from ai_journey.metal_flash_plan import plan_sha256
from ai_journey.metal_flash_report import GPU_ONLY, MEASUREMENT_BOUNDARY
from ai_journey.metal_flash_verify import MetalFlashVerificationError, verify_report
from test_metal_flash_emulator import FlashEmulator
from test_metal_flash_report import (
    recorded_flash_ceiling,
    small_config,
    synthetic_report,
)


@functools.cache
def _base(directory):
    return synthetic_report(directory)


@pytest.fixture
def report(tmp_path_factory):
    return copy.deepcopy(_base(tmp_path_factory.getbasetemp()))


def resealed(report):
    report = copy.deepcopy(report)
    report.pop("report_sha256")
    return seal_report(report)


def edit(report, path, value):
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return report


def test_a_synthetic_report_verifies_with_its_config_and_ceiling(report, tmp_path):
    summary = verify_report(
        report, config=small_config(), ceiling=recorded_flash_ceiling(tmp_path)
    )

    assert summary["measured"] is False
    assert summary["plan_sha256"] == plan_sha256()
    assert summary["correctness_cases"] == 2
    assert summary["identical_repeats"] == {"n0": True, "f1": True}


def test_a_change_without_a_new_seal_is_rejected(report):
    report["summary"]["warnings"] = 5

    with pytest.raises(MetalFlashVerificationError, match="digest"):
        verify_report(report)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("measured",), True, "measurement"),
        (("evidence_boundary",), MEASUREMENT_BOUNDARY, "boundary"),
        (("evidence_boundary", "ptx_executed"), True, "boundary"),
        (("plan", "steps", 2, "predicted"), ["faster"], "declared plan"),
        (("correctness", "cases", 0, "q_sha256"), "0" * 64, "seed"),
        (
            ("correctness", "cases", 1, "modes", "full", "reference_sha256"),
            "0" * 64,
            "seed",
        ),
        (
            ("correctness", "cases", 1, "modes", "causal", "unscaled_exp_overflows"),
            False,
            "seed",
        ),
        (
            ("benchmark", "points", 0, "sides", "f1", "gflops_per_second"),
            1.0,
            "recompute",
        ),
        (("benchmark", "points", 1, "sides", "n0", "samples_ns"), [1, 2], "samples_ns"),
        (("benchmark", "points", 0, "pairs"), 1, "recompute"),
        (
            ("benchmark", "points", 0, "memory", "n0", "construction_bytes", "scratch"),
            0,
            "recompute",
        ),
        (
            (
                "benchmark",
                "points",
                0,
                "memory",
                "torch_sdpa",
                "peak_driver_over_before_bytes",
            ),
            1,
            "recompute",
        ),
        (("benchmark", "points", 1, "memory", "f1", "readings", "polls"), 0, "form"),
        (("benchmark", "points", 0, "checks", "f1", "max_bound_ratio"), 1.5, "bound"),
        (("speedup", "points", 0, "steps", 1, "verdict"), "faster", "speedup"),
        (("speedup", "hypotheses", 0, "held_at"), [0, 1], "speedup"),
        (("compute_ceiling", "fma_gflops_per_second"), 99999.0, "supplied ceiling"),
        (
            ("shaders", "kernels", 6, "limits", "static_threadgroup_memory_bytes"),
            8192,
            "shader section",
        ),
        (("memory_method", "cannot_show"), "nothing", "memory method"),
        (("verification", "gpu_only"), [], "scope"),
        (("summary", "max_bound_ratio", "f1"), 0.0, "summary"),
        (("run", "elapsed_ns"), 1, "budget"),
    ],
)
def test_a_changed_and_resealed_field_is_rejected(report, tmp_path, path, value, match):
    changed = resealed(edit(report, path, value))

    with pytest.raises(MetalFlashVerificationError, match=match):
        verify_report(changed, ceiling=recorded_flash_ceiling(tmp_path))


def test_an_output_that_claims_the_rounded_reference_must_rebuild_it(report):
    entry = report["correctness"]["cases"][0]["modes"]["full"]["outputs"]["n0"]
    entry["differing_from_rounded_reference"] = 0

    with pytest.raises(MetalFlashVerificationError, match="rounded reference"):
        verify_report(resealed(report))


def test_the_report_must_come_from_the_supplied_config(report):
    other = small_config(benchmark={"check_rows": 9})

    with pytest.raises(MetalFlashVerificationError, match="supplied config"):
        verify_report(report, config=other)


def test_a_gpu_only_reading_changed_and_resealed_still_verifies(report):
    """The CPU cannot recompute what only the GPU run saw, so it checks form only."""

    entry = report["benchmark"]["points"][0]["memory"]["f1"]["readings"]
    entry["polls"] = 7
    report["correctness"]["cases"][0]["modes"]["full"]["outputs"]["f1"][
        "identical_repeats"
    ] = False
    changed = resealed(report)
    changed["summary"]["identical_repeats"]["f1"] = False
    changed = resealed(changed)

    assert verify_report(changed)["identical_repeats"]["f1"] is False
    assert any("identical_repeats" in text for text in GPU_ONLY)


def test_a_report_from_a_corrupted_kernel_never_exists(tmp_path):
    def nudge(side, output):
        output = output.copy()
        output[0] = numpy.float32(output[0] + 0.5)
        return output

    with pytest.raises(ValueError):
        synthetic_report(tmp_path, backend=FlashEmulator(corrupt=nudge))


@pytest.mark.parametrize("value", [None, [], "report", {"kind": "x"}])
def test_a_malformed_report_is_rejected(value):
    with pytest.raises(MetalFlashVerificationError):
        verify_report(value)
