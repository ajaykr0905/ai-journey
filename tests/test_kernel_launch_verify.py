import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ai_journey import kernel_launch_report, kernel_launch_verify
from ai_journey.kernel_launch_config import KernelLaunchError
from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.kernel_launch_plan import KernelLaunchPlan, load_plan
from ai_journey.kernel_launch_report import DIGEST_FIELD, build_report, seal_report
from ai_journey.kernel_launch_verify import (
    KernelLaunchVerificationError,
    load_report,
    verify_report,
)
from test_kernel_launch_plan import plan_mapping

ROOT = Path(__file__).resolve().parents[1]
PLAN = KernelLaunchPlan.from_mapping(plan_mapping())


@pytest.fixture(scope="module")
def sealed():
    return seal_report(build_report(PLAN))


def resealed(report, change):
    tampered = copy.deepcopy(report)
    del tampered[DIGEST_FIELD]
    change(tampered)
    return seal_report(tampered)


def set_path(*path):
    """Return a change that stores the last argument at the other keys."""

    *keys, value = path

    def change(report):
        target = report
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return change


def test_a_fresh_report_verifies_and_binds_to_its_plan(sealed):
    assert verify_report(sealed) == PLAN
    assert verify_report(sealed, plan=PLAN) == PLAN
    other = KernelLaunchPlan.from_mapping({**plan_mapping(), "name": "other-plan"})
    with pytest.raises(KernelLaunchVerificationError, match="not built from this plan"):
        verify_report(sealed, plan=other)
    with pytest.raises(TypeError, match="KernelLaunchPlan"):
        verify_report(sealed, plan=plan_mapping())


def test_the_pinned_report_reproduces_from_the_pinned_plan():
    plan = load_plan(ROOT / "config/day-73-kernel-launch.json")

    assert verify_report(seal_report(build_report(plan)), plan=plan) == plan


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (set_path("summary", "out_of_bounds_writes_caught", 0), "summary.out_of_bounds_writes_caught"),
        (set_path("summary", "kernels", 2.0), "summary.kernels"),
        (set_path("summary", "kernels", True), "summary.kernels"),
        (set_path("vector_add", "runs", 1, "output_sha256", "0" * 64), r"vector_add.runs\[1\].output_sha256"),
        (set_path("vector_add", "runs", 0, "useful_threads", 128), r"vector_add.runs\[0\].useful_threads"),
        (set_path("vector_add", "launch", "idle_threads", 0), "vector_add.launch.idle_threads"),
        (set_path("rgb_to_grayscale", "order_independent", False), "rgb_to_grayscale.order_independent"),
        (set_path("rgb_to_grayscale", "runs", 1, "order_sha256", "0" * 64), r"rgb_to_grayscale.runs\[1\].order_sha256"),
        (set_path("unguarded_demos", 0, "out_of_bounds_writes", 0), r"unguarded_demos\[0\].out_of_bounds_writes"),
        (set_path("unguarded_demos", 1, "first_out_of_bounds_writes", []), r"unguarded_demos\[1\].first_out_of_bounds_writes"),
        (set_path("unguarded_demos", 1, "conflicts_sha256", "0" * 64), r"unguarded_demos\[1\].conflicts_sha256"),
        (set_path("limits", "max_threads_per_block", 2048), "limits.max_threads_per_block"),
        (set_path("grayscale_rule", "weights_per_10000", [2990, 5870, 1140]), r"grayscale_rule.weights_per_10000\[0\]"),
        (set_path("evidence_boundary", "gpu_executed", True), "evidence_boundary.gpu_executed"),
        (set_path("cuda_source", "compiled", True), "cuda_source.compiled"),
        (set_path("plan", "vector_add", "n", 101), "vector_add.extent"),
        (set_path("plan", "vector_add", "n", True), "embedded plan is invalid"),
        (set_path("plan", "unguarded_demos", []), "embedded plan is invalid"),
        (set_path("kind", "other"), "report kind"),
        (set_path("schema_version", 2), "schema_version"),
        (set_path("extra", 1), "fields do not match the schema"),
    ],
)  # fmt: skip
def test_resealed_tampering_is_caught_at_the_first_wrong_field(sealed, change, message):
    with pytest.raises(KernelLaunchVerificationError, match=message):
        verify_report(resealed(sealed, change))


def test_edited_cuda_text_fails_its_structural_check_even_when_resealed(sealed):
    def unguard(report):
        cuda = report["cuda_source"]
        cuda["text"] = cuda["text"].replace("if (i < n) {", "{", 1)

    with pytest.raises(
        KernelLaunchVerificationError,
        match="cuda_source.text structure.checks.vector_add_guard",
    ):
        verify_report(resealed(sealed, unguard))


def test_an_unsealed_edit_fails_the_digest_before_anything_else(sealed):
    tampered = copy.deepcopy(sealed)
    tampered["summary"]["runs"] = 99

    with pytest.raises(KernelLaunchVerificationError, match="digest does not match"):
        verify_report(tampered)
    with pytest.raises(KernelLaunchVerificationError, match="fields do not match"):
        verify_report([sealed])


def test_a_broken_simulator_cannot_produce_a_verifiable_report(monkeypatch):
    real = kernel_launch_report.run_rgb_to_grayscale

    def off_by_one(rgb, **options):
        run = real(rgb, **options)
        if options["order"].kind == "permuted":
            return replace(
                run,
                output_sha256=kernel_launch_report.values_sha256(
                    "uint8", tuple(min(255, value + 1) for value in run.output)
                ),
            )
        return run

    monkeypatch.setattr(kernel_launch_report, "run_rgb_to_grayscale", off_by_one)
    report = seal_report(build_report(PLAN))

    # Fields compare in sorted order, so order_independent fails before runs.
    with pytest.raises(
        KernelLaunchVerificationError, match="rgb_to_grayscale.order_independent"
    ):
        verify_report(report)


def test_the_report_must_also_equal_a_fresh_build(sealed, monkeypatch):
    real = kernel_launch_verify.build_report

    def drifted(plan):
        report = real(plan)
        report["cuda_source"]["text"] += "\n"
        return report

    monkeypatch.setattr(kernel_launch_verify, "build_report", drifted)
    with pytest.raises(
        KernelLaunchVerificationError, match="differs from a fresh build"
    ):
        verify_report(sealed)


def test_loading_separates_unreadable_files_from_rejected_reports(sealed, tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(sealed) + b"\n")
    assert load_report(path, plan=PLAN) == sealed

    with pytest.raises(KernelLaunchError) as unreadable:
        load_report(tmp_path / "missing.json")
    assert not isinstance(unreadable.value, KernelLaunchVerificationError)

    for payload, message in (
        (json.dumps(sealed, indent=2).encode("ascii") + b"\n", "not canonical JSON"),
        (canonical_json_bytes(sealed), "not canonical JSON"),
        (b'{"kind":"a","kind":"b"}', "duplicate JSON key: kind"),
        (b"[]\n", "must be a JSON object"),
    ):
        path.write_bytes(payload)
        with pytest.raises(KernelLaunchVerificationError, match=message):
            load_report(path)
