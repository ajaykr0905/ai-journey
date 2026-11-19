import ast
import copy
import json
from pathlib import Path

import pytest

from ai_journey.gpu_tiled_config import PINNED_PLAN, GpuTiledPlan
from ai_journey.gpu_tiled_cuda import GpuTiledError
from ai_journey.gpu_tiled_report import (
    DIGEST_FIELD,
    EVIDENCE_BOUNDARY,
    build_report,
    report_digest,
    seal_report,
)
from ai_journey.gpu_tiled_verify import (
    GpuTiledVerificationError,
    load_report,
    verify_report,
)
from ai_journey.kernel_launch_io import canonical_json_bytes
from test_gpu_tiled_config import small_mapping

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA256 = "2419f0bfedcffd5eaa363e07640023e4e81a4bcfc752a415fd8a30158f06ffae"
SOURCE_SHA256 = {
    8: "3b72d60eb486fb5d1671efcf5428d3b6d93f4c61d0e6ec1b4c97815be31884fa",
    16: "a789a301b41dde5d8449860297a0c5340fabfbc9c8c39209fa15791b038fc867",
    32: "178e6d6ec2976d70ef02e0f9cc38eedbdacdd863588b3e441fecc393826b7d6e",
    "cublas": "9a86b34a84f3550412c2971a11b3972f6d1595b67193812afdb9583e1c36f889",
}
SMALL = GpuTiledPlan.from_mapping(small_mapping())


@pytest.fixture(scope="module")
def pinned():
    report = seal_report(build_report(PINNED_PLAN))
    assert verify_report(report, plan=PINNED_PLAN) == PINNED_PLAN
    return report


@pytest.fixture(scope="module")
def small():
    report = seal_report(build_report(SMALL))
    assert verify_report(report, plan=SMALL) == SMALL
    return report


def recorded_day_74_cases():
    """Read the recorded Apple GPU report as text; importing it probes Metal."""

    tree = ast.parse((ROOT / "tests/test_matmul_metal_recorded.py").read_text())
    (value,) = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", None) == "RECORDED_REPORT_JSON"
    ]
    return json.loads(ast.literal_eval(value))["correctness"]["cases"]


def resealed(report, change):
    tampered = copy.deepcopy(report)
    change(tampered)
    del tampered[DIGEST_FIELD]
    return seal_report(tampered)


def test_the_pinned_report_reproduces_its_digests_and_counts(pinned):
    assert pinned[DIGEST_FIELD] == REPORT_SHA256 == report_digest(pinned)
    assert [item["sha256"] for item in pinned["sources"]["tiled"]] == [
        SOURCE_SHA256[tile] for tile in (8, 16, 32)
    ]
    assert pinned["sources"]["cublas"]["sha256"] == SOURCE_SHA256["cublas"]
    assert pinned["summary"] == {
        "cases": 9,
        "tiles": 3,
        "block_runs": 27,
        "block_runs_matching_replay": 27,
        "block_runs_matching_traffic": 27,
        "race_free_block_runs": 27,
        "stored_once_block_runs": 27,
        "cuda_checks_passed": 297,
        "cuda_checks": 297,
        "cublas_checks_passed": 63,
        "cublas_checks": 63,
        "occupancy_estimates": 9,
    }
    assert pinned["evidence_boundary"] == dict(EVIDENCE_BOUNDARY)
    assert len(canonical_json_bytes(pinned)) < 64 * 1024


def test_every_tile_holds_its_case_replay_and_the_bound_holds(pinned):
    for case in pinned["cases"]:
        assert [entry["run"]["output_sha256"] for entry in case["tiles"]] == [
            case["replay_sha256"]
        ] * 3
        assert all(entry["matches_replay"] for entry in case["tiles"])
        assert 0 < case["replay_relative_error"] <= case["error_bound"]
    # The largest case: every block column reads A once, every block row B.
    largest = pinned["cases"][8]["tiles"][0]["run"]["global_loads"]
    assert (largest["a"], largest["b"]) == (125 * 1000 * 100, 125 * 100 * 999)


def test_the_replays_name_the_bits_that_the_apple_gpu_held_on_day_74(pinned):
    recorded = {case["index"]: case for case in recorded_day_74_cases()}

    # The recorded run used seed 74 too, with the same first two shapes.
    for index in (0, 1):
        case = pinned["cases"][index]
        assert (recorded[index]["m"], recorded[index]["n"], recorded[index]["k"]) == (
            case["shape"]["m"],
            case["shape"]["n"],
            case["shape"]["k"],
        )
        assert case["replay_sha256"] == recorded[index]["replay_sha256"]
        assert case["inputs_sha256"]["a"] == recorded[index]["a_sha256"]


def test_the_occupancy_section_lists_each_tile_and_register_count(pinned):
    table = {
        (item["tile"], item["kernel"]["registers_per_thread"]): (
            item["active_warps_per_sm"],
            item["occupancy"]["decimal"],
            item["shared_memory"]["allocated_bytes_per_block"],
        )
        for item in pinned["occupancy"]["estimates"]
    }

    assert table == {
        (8, 32): (64, "1.000000", 1536),
        (8, 40): (48, "0.750000", 1536),
        (8, 64): (32, "0.500000", 1536),
        (16, 32): (64, "1.000000", 3072),
        (16, 40): (48, "0.750000", 3072),
        (16, 64): (32, "0.500000", 3072),
        (32, 32): (64, "1.000000", 9216),
        (32, 40): (32, "0.500000", 9216),
        (32, 64): (32, "0.500000", 9216),
    }
    assert pinned["occupancy"]["architecture"]["launch_limits_agree"] is True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["evidence_boundary"].update(gpu_executed=True), "evidence_boundary"),
        (lambda r: r["evidence_boundary"].update(tf32_assumption_checked=True), "evidence_boundary"),
        (lambda r: r["sources"]["tiled"][0].update(text="__global__ void k() {}\n"), "sources"),
        (lambda r: r["sources"]["cublas"].update(called=True), "sources"),
        (lambda r: r["rule"].update(fma="sum += a * b"), "rule"),
        (lambda r: r["summary"].update(block_runs_matching_replay=5), "summary"),
        (lambda r: r["occupancy"]["estimates"][0].update(active_warps_per_sm=63), "occupancy"),
        (lambda r: r["cases"][0].update(replay_sha256="0" * 64), r"cases\[0\]"),
        (lambda r: r["cases"][1]["tiles"][0]["run"].update(output_sha256="0" * 64), r"cases\[1\]"),
        (lambda r: r["cases"][1]["tiles"][1]["run"]["global_loads"].update(a=1), r"cases\[1\]"),
        (lambda r: r["cases"][2]["tiles"][0]["run"]["hazards"].update(read_after_write=1), r"cases\[2\]"),
        (lambda r: r["cases"][2]["cublas_check"]["checks"].update(cublas_math_mode=False), r"cases\[2\]"),
        (lambda r: r["cases"][2].update(error_bound=1.0), r"cases\[2\]"),
        (lambda r: r["cases"].pop(), "cases rows"),
    ],
)  # fmt: skip
def test_a_resealed_change_to_any_section_is_named(small, change, message):
    with pytest.raises(GpuTiledVerificationError, match=message):
        verify_report(resealed(small, change), plan=SMALL)


def test_the_seal_the_schema_and_the_plan_binding_are_checked(small):
    unsealed = copy.deepcopy(small)
    unsealed["summary"]["cases"] = 4
    with pytest.raises(GpuTiledVerificationError, match="digest does not match"):
        verify_report(unsealed)
    with pytest.raises(GpuTiledVerificationError, match="fields do not match"):
        verify_report({**small, "extra": 1})
    with pytest.raises(GpuTiledVerificationError, match="not built from this plan"):
        verify_report(small, plan=PINNED_PLAN)
    renamed = resealed(small, lambda r: r["plan"].update(name="Other Plan"))
    with pytest.raises(GpuTiledVerificationError, match="embedded plan is invalid"):
        verify_report(renamed)
    kind = resealed(small, lambda r: r.update(kind="matmul_model"))
    with pytest.raises(GpuTiledVerificationError, match="kind"):
        verify_report(kind)
    with pytest.raises(GpuTiledError, match="already sealed"):
        seal_report(small)
    with pytest.raises(TypeError, match="GpuTiledPlan"):
        build_report(PINNED_PLAN.to_mapping())


def test_a_report_file_must_hold_canonical_bytes(small, tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(small) + b"\n")
    assert load_report(path, plan=SMALL) == small

    path.write_bytes(json.dumps(small, indent=1).encode("ascii"))
    with pytest.raises(GpuTiledVerificationError, match="not canonical"):
        load_report(path)
    path.write_bytes(b"[1]\n")
    with pytest.raises(GpuTiledVerificationError, match="JSON object"):
        load_report(path)
    path.write_bytes(b'{"kind":1,"kind":2}\n')
    with pytest.raises(GpuTiledVerificationError, match="duplicate"):
        load_report(path)
