import copy
import json
import math
import random
import struct
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from ai_journey import matmul_model_report, matmul_model_verify
from ai_journey.kernel_launch_config import KernelLaunchError
from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.matmul_model_access import OperandTraffic
from ai_journey.matmul_model_plan import MatmulModelPlan, load_plan
from ai_journey.matmul_model_reference import round_to_float32
from ai_journey.matmul_model_report import DIGEST_FIELD, build_report, seal_report
from ai_journey.matmul_model_verify import (
    MatmulModelVerificationError,
    load_report,
    verify_report,
)
from test_matmul_model_plan import plan_mapping

ROOT = Path(__file__).resolve().parents[1]
PLAN = MatmulModelPlan.from_mapping(plan_mapping())


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
    other = MatmulModelPlan.from_mapping({**plan_mapping(), "name": "other-plan"})
    with pytest.raises(MatmulModelVerificationError, match="not built from this plan"):
        verify_report(sealed, plan=other)
    with pytest.raises(TypeError, match="MatmulModelPlan"):
        verify_report(sealed, plan=plan_mapping())


def test_the_pinned_report_reproduces_from_the_pinned_plan():
    plan = load_plan(ROOT / "config/day-74-matmul-model.json")

    assert verify_report(seal_report(build_report(plan)), plan=plan) == plan


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (set_path("summary", "runs", 9), "summary.runs"),
        (set_path("summary", "cases", 2.0), "summary.cases"),
        (set_path("summary", "cases", True), "summary.cases"),
        (set_path("cases", 0, "reference_sha256", "0" * 64), r"cases\[0\].reference_sha256"),
        (set_path("cases", 1, "inputs_sha256", "a", "0" * 64), r"cases\[1\].inputs_sha256.a"),
        (set_path("cases", 0, "order_rounding", "differing", 0), r"cases\[0\].order_rounding.differing"),
        (set_path("cases", 0, "mappings", 1, "runs", 1, "order_sha256", "0" * 64), r"cases\[0\].mappings\[1\].runs\[1\].order_sha256"),
        (set_path("cases", 0, "mappings", 0, "launch", "idle_threads", 0), r"cases\[0\].mappings\[0\].launch.idle_threads"),
        (set_path("cases", 0, "mappings", 0, "access", "operands", "a", "sectors", 1), r"cases\[0\].mappings\[0\].access.operands.a.sectors"),
        (set_path("cases", 0, "mappings", 1, "access", "operands", "b", "classes", "strided", 1), r"cases\[0\].mappings\[1\].access.operands.b.classes.strided"),
        (set_path("cases", 0, "mappings", 0, "probes", 1, "steps", 2, "b", "lane_addresses", [0] * 32), r"cases\[0\].mappings\[0\].probes\[1\].steps\[2\].b.lane_addresses\[0\]"),
        (set_path("cases", 0, "mappings", 0, "probes", 0, "trace_matches_model", False), r"cases\[0\].mappings\[0\].probes\[0\].trace_matches_model"),
        (set_path("cases", 1, "mappings", 1, "cross_check", "passed", False), r"cases\[1\].mappings\[1\].cross_check.passed"),
        (set_path("accounting", 2, "roofline", 0, "no_reuse", "bound", "compute"), r"accounting\[2\].roofline\[0\].no_reuse.bound"),
        (set_path("accounting", 0, "no_reuse", "intensity", "decimal", "0.25"), r"accounting\[0\].no_reuse.intensity.decimal"),
        (set_path("accounting", 1, "cross_checks", "roofline_matmul_workload", False), r"accounting\[1\].cross_checks.roofline_matmul_workload"),
        (set_path("occupancy", "estimates", 1, "active_blocks_per_sm", 8), r"occupancy.estimates\[1\].active_blocks_per_sm"),
        (set_path("occupancy", "architecture", "max_warps_per_sm", 48), "occupancy.architecture.max_warps_per_sm"),
        (set_path("limits", "max_threads_per_block", 2048), "limits.max_threads_per_block"),
        (set_path("rule", "summation", "any order"), "rule.summation"),
        (set_path("geometry", "cache_modeled", True), "geometry.cache_modeled"),
        (set_path("evidence_boundary", "gpu_executed", True), "evidence_boundary.gpu_executed"),
        (set_path("evidence_boundary", "ceilings_measured", True), "evidence_boundary.ceilings_measured"),
        (set_path("cuda_source", "compiled", True), "cuda_source.compiled"),
        (set_path("plan", "cases", 0, "k", 4), r"cases\[0\].inputs_sha256.a"),
        (set_path("plan", "block", [32, 3]), "structure for ragged.checks.matmul_grid"),
        (set_path("plan", "cases", []), "embedded plan is invalid"),
        (set_path("plan", "registers_per_thread", [True]), "embedded plan is invalid"),
        (set_path("kind", "other"), "report kind"),
        (set_path("schema_version", 2), "schema_version"),
        (set_path("extra", 1), "fields do not match the schema"),
    ],
)  # fmt: skip
def test_resealed_tampering_is_caught_at_the_first_wrong_field(sealed, change, message):
    with pytest.raises(MatmulModelVerificationError, match=message):
        verify_report(resealed(sealed, change))


def test_edited_cuda_text_fails_its_structural_check_even_when_resealed(sealed):
    # The store moves past the guard, so threads outside C would write.
    def unguard(report):
        cuda = report["cuda_source"]
        cuda["text"] = cuda["text"].replace(
            "        c[row * n + col] = sum;\n    }\n}",
            "    }\n    c[row * n + col] = sum;\n}",
            1,
        )

    tampered = resealed(sealed, unguard)
    assert tampered["cuda_source"]["text"] != sealed["cuda_source"]["text"]
    with pytest.raises(
        MatmulModelVerificationError,
        match="cuda_source.text structure for ragged.checks.matmul_guard",
    ):
        verify_report(tampered)


def test_an_unsealed_edit_fails_the_digest_before_anything_else(sealed):
    tampered = copy.deepcopy(sealed)
    tampered["summary"]["runs"] = 99

    with pytest.raises(MatmulModelVerificationError, match="digest does not match"):
        verify_report(tampered)
    with pytest.raises(MatmulModelVerificationError, match="fields do not match"):
        verify_report([sealed])


def test_a_drifted_access_model_fails_the_independent_recount(monkeypatch):
    # The fresh build would repeat the drift, so only the verifier's own
    # address arithmetic can catch it.
    real = matmul_model_report.analyze_launch

    def drifted(launch):
        access = real(launch)
        operands = dict(access.operands)
        operands["b"] = replace(operands["b"], lines=operands["b"].lines - 1)
        assert type(operands["b"]) is OperandTraffic
        return replace(access, operands=tuple(operands.items()))

    monkeypatch.setattr(matmul_model_report, "analyze_launch", drifted)
    with pytest.raises(
        MatmulModelVerificationError,
        match=r"cases\[0\].mappings\[0\].access.operands.b.lines",
    ):
        verify_report(seal_report(build_report(PLAN)))


def test_a_broken_simulator_cannot_produce_a_verifiable_report(monkeypatch):
    real = matmul_model_report.run_naive_matmul

    def off_by_one_order(a, b, shape, **options):
        run = real(a, b, shape, **options)
        if options["order"].kind == "permuted":
            return replace(run, output_sha256="0" * 64)
        return run

    monkeypatch.setattr(matmul_model_report, "run_naive_matmul", off_by_one_order)
    report = seal_report(build_report(PLAN))

    # Fields compare in sorted order, so order_independent fails before runs.
    with pytest.raises(
        MatmulModelVerificationError, match=r"mappings\[0\].order_independent"
    ):
        verify_report(report)


def test_the_report_must_also_equal_a_fresh_build(sealed, monkeypatch):
    real = matmul_model_verify.build_report

    def drifted(plan):
        report = real(plan)
        report["cuda_source"]["text"] += "\n"
        return report

    monkeypatch.setattr(matmul_model_verify, "build_report", drifted)
    with pytest.raises(
        MatmulModelVerificationError, match="differs from a fresh build"
    ):
        verify_report(sealed)


def test_loading_separates_unreadable_files_from_rejected_reports(sealed, tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(sealed) + b"\n")
    assert load_report(path, plan=PLAN) == sealed

    with pytest.raises(KernelLaunchError) as unreadable:
        load_report(tmp_path / "missing.json")
    assert not isinstance(unreadable.value, MatmulModelVerificationError)

    for payload, message in (
        (json.dumps(sealed, indent=2).encode("ascii") + b"\n", "not canonical JSON"),
        (canonical_json_bytes(sealed), "not canonical JSON"),
        (b'{"kind":"a","kind":"b"}', "duplicate JSON key: kind"),
        (b"[]\n", "must be a JSON object"),
    ):
        path.write_bytes(payload)
        with pytest.raises(MatmulModelVerificationError, match=message):
            load_report(path)


def test_an_overlong_cuda_literal_is_a_rejection_not_a_crash(sealed, tmp_path):
    def overflow(report):
        cuda = report["cuda_source"]
        cuda["text"] = cuda["text"].replace(
            "dim3 block(32, 2);", f"dim3 block({'9' * 5_000}, 2);"
        )

    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(resealed(sealed, overflow)) + b"\n")

    with pytest.raises(MatmulModelVerificationError, match="checks.matmul_grid"):
        load_report(path)


def float32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


@pytest.mark.parametrize(
    ("significand", "exponent"),
    [
        (2**25 - 1, 0),
        (-(2**25 - 1), -40),
        (2**24 + 1, 3),
        (2**24 + 3, -3),
        (2**48 - 1, -60),
        (3, -2),
        (-1, 0),
    ],
)
def test_the_verifier_rounds_and_encodes_edge_values_like_ieee(significand, exponent):
    # 2**25 - 1 rounds up with a carry to a 25-bit 2**24. No pinned input
    # reaches a carry, so these values pin the encoder on that path.
    rounded = matmul_model_verify._round24(significand, exponent)

    assert matmul_model_verify._bits(rounded) == float32_bits(
        math.ldexp(significand, exponent)
    )
    assert math.ldexp(*rounded) == round_to_float32(
        Fraction(significand) * Fraction(2) ** exponent
    )


def test_the_verifier_encoding_matches_struct_on_seeded_values():
    # Each significand has at most 48 bits, so ldexp is exact in double and
    # struct rounds it to float32 once.
    draw = random.Random(74)
    for _ in range(2_000):
        significand = draw.randrange(1, 2**48) * draw.choice((1, -1))
        exponent = draw.randrange(-70, 40)
        rounded = matmul_model_verify._round24(significand, exponent)
        assert matmul_model_verify._bits(rounded) == float32_bits(
            math.ldexp(significand, exponent)
        )
    assert matmul_model_verify._bits((0, 0)) == 0
    with pytest.raises(MatmulModelVerificationError, match="normal float32 range"):
        matmul_model_verify._bits((1, 200))
    with pytest.raises(MatmulModelVerificationError, match="normal float32 range"):
        matmul_model_verify._bits((1, -200))
