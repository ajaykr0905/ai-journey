import copy
import json
import dataclasses
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.matmul_bottleneck_config import (
    MatmulBottleneckVerificationError,
    config_from_mapping,
    load_config,
)
from ai_journey.matmul_bottleneck_report import build_report
from ai_journey.matmul_bottleneck_verify import verify_report
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import DIGEST_FIELD, report_digest
from test_matmul_bottleneck_recorded import RECORDED_REPORT_JSON
from test_matmul_bottleneck_report import measured_timing, synthetic_report
from test_matmul_bottleneck_sources import recorded_sources, synthetic_timing

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config/day-75-matmul-bottleneck.json")
SYNTHETIC = synthetic_report()


def resealed(change, report=SYNTHETIC):
    tampered = copy.deepcopy(report)
    del tampered[DIGEST_FIELD]
    change(tampered)
    tampered[DIGEST_FIELD] = report_digest(tampered)
    return tampered


def test_a_synthetic_report_verifies_and_binds_to_its_config():
    summary = verify_report(SYNTHETIC, config=CONFIG)

    assert summary == {
        "kind": "matmul_bottleneck",
        "report_sha256": SYNTHETIC[DIGEST_FIELD],
        "config_sha256": CONFIG.fingerprint(),
        "measured": False,
        "scope": "synthetic_backend_or_clock",
        "label": "synthetic timing, not a measurement",
        "points": 2,
        "top_bottleneck": {
            "matmul_x_columns": "indeterminate",
            "matmul_x_rows": "indeterminate",
        },
        "sources_checked": False,
    }


def test_a_measured_report_verifies_as_measured():
    report = build_report(recorded_sources(), measured_timing(), CONFIG)

    assert verify_report(report)["measured"] is True


def test_another_config_does_not_bind():
    other = config_from_mapping(
        {**CONFIG.to_mapping(), "name": "another-bottleneck-config"}
    )

    with pytest.raises(MatmulBottleneckVerificationError, match="supplied config"):
        verify_report(SYNTHETIC, config=other)


def test_an_edit_without_a_new_seal_breaks_the_digest():
    edited = copy.deepcopy(SYNTHETIC)
    edited["points"][0]["kernels"]["matmul_x_rows"]["decision"]["bottleneck"] = (
        "memory-bound"
    )

    with pytest.raises(MatmulBottleneckVerificationError, match="digest"):
        verify_report(edited)


def kernel(report, index=0, name="matmul_x_rows"):
    return report["points"][index]["kernels"][name]


def raise_dispatch_threshold(report):
    # The digest follows, so only the decisions can catch the change.
    report["config"]["rule"]["dispatch_overhead_share_min"] = 0.6
    report["config_sha256"] = sha256(canonical_json_bytes(report["config"])).hexdigest()


TAMPERS = {
    "renamed decision": (
        lambda r: kernel(r)["decision"].update(bottleneck="memory-bound"),
        "point 0 does not recompute",
    ),
    "changed one-call time": (
        lambda r: kernel(r)["inputs"].update(median_host_ns=9_000_000),
        "point 0 does not recompute",
    ),
    "changed median": (
        lambda r: r["points"][1]["median_sample_ns"].update(matmul_x_rows=10),
        "point 1 does not recompute",
    ),
    "changed reason": (
        lambda r: kernel(r)["decision"].update(reason="near_fma_ceiling"),
        "point 0 does not recompute",
    ),
    "changed verdict": (
        lambda r: r["points"][0]["mapping"].update(
            verdict="no_difference_in_model_or_timing"
        ),
        "point 0 does not recompute",
    ),
    "changed sectors": (
        lambda r: r["points"][0]["mapping"]["warps"]["x_rows"].update(sectors=1),
        "point 0 does not recompute",
    ),
    "changed threadgroups": (
        lambda r: r["points"][0]["threadgroups"].update(matmul_x_rows=[1, 1]),
        "point 0 does not recompute",
    ),
    "changed ceiling": (
        lambda r: r["ceilings"].update(fma_gflops_per_second=9000.0),
        "point 0 does not recompute",
    ),
    "threshold and digest changed together": (
        raise_dispatch_threshold,
        "point 1 does not recompute",
    ),
    "config without its digest": (
        lambda r: r["config"]["rule"].update(near_roof_share_min=0.9),
        "config digest does not match",
    ),
    "invalid embedded config": (
        lambda r: r["config"]["rule"].update(near_roof_share_min=2.0),
        "embedded config is invalid",
    ),
    "summary": (
        lambda r: r["summary"]["top_bottleneck"]["matmul_x_rows"].update(
            top="compute-bound"
        ),
        "summary does not reconcile",
    ),
    "counter claim": (
        lambda r: r["evidence_boundary"].update(hardware_counters_used=True),
        "evidence labels or boundary were weakened",
    ),
    "claimed measurement": (
        lambda r: r.update(measured=True),
        "only when the gap and the timing input are",
    ),
    "measured timing label on synthetic report": (
        lambda r: r["sources"]["timing"].update(
            measured=True, label="host-specific Apple GPU (Metal) measurement"
        ),
        "only when the gap and the timing input are",
    ),
    "wide SIMD group": (
        lambda r: r["geometry"]["simd_widths"].update(matmul_x_rows=64),
        "SIMD width of 32",
    ),
    "definitions": (
        lambda r: r["definitions"].update(overhead_share="anything"),
        "definitions were changed",
    ),
    "reasons": (
        lambda r: r["reasons"]["below_both_roofs"].update(bottleneck="memory-bound"),
        "reasons were changed",
    ),
    "unused timing count": (
        lambda r: r["sources"]["timing"].update(entries_used=3),
        "one entry per naive kernel",
    ),
    "faster than the fastest call": (
        lambda r: r["sources"]["timing"]["fastest_calls"]["matmul_x_rows"].update(
            median_host_ns=2_000_000
        ),
        "faster than the fastest call",
    ),
    "changed fastest call": (
        lambda r: r["sources"]["timing"]["fastest_calls"]["matmul_x_rows"].update(
            median_host_ns=129_000
        ),
        "point 0 does not recompute",
    ),
    "malformed point": (
        lambda r: r["points"].__setitem__(0, []),
        "point 0 must be a JSON object",
    ),
    "missing kernel": (
        lambda r: r["points"][0]["kernels"].pop("matmul_x_rows"),
        "point 0 must hold both naive kernels",
    ),
}


@pytest.mark.parametrize("name", TAMPERS)
def test_every_resealed_tamper_is_caught_by_recomputation(name):
    change, message = TAMPERS[name]

    with pytest.raises(MatmulBottleneckVerificationError, match=message):
        verify_report(resealed(change))


def test_a_measured_report_must_come_from_macos():
    report = build_report(recorded_sources(), measured_timing(), CONFIG)
    linux = resealed(lambda r: r["environment"].update(platform_system="Linux"), report)

    with pytest.raises(MatmulBottleneckVerificationError, match="from macOS"):
        verify_report(linux)


def test_a_rebuilt_report_must_match_byte_for_byte():
    # A measured timing input has another digest, so the sources differ.
    other_timing = build_report(recorded_sources(), measured_timing(), CONFIG)
    # A looser threshold keeps the sources but changes the bytes.
    relaxed = dataclasses.replace(
        CONFIG, rule=dataclasses.replace(CONFIG.rule, near_roof_share_min=0.69)
    )
    other_config = synthetic_report_with(relaxed)

    assert verify_report(SYNTHETIC, rebuilt=SYNTHETIC)["sources_checked"] is True
    with pytest.raises(MatmulBottleneckVerificationError, match="supplied sources"):
        verify_report(SYNTHETIC, rebuilt=other_timing)
    with pytest.raises(MatmulBottleneckVerificationError, match="rebuilt from"):
        verify_report(SYNTHETIC, rebuilt=other_config)


def synthetic_report_with(config):
    return build_report(recorded_sources(), synthetic_timing(), config)


@pytest.mark.parametrize(
    "change",
    [{"median_gpu_ns": 1}, {"median_dispatch_ns": 999_999_999}],
    ids=["gpu time", "dispatch time"],
)
def test_a_fastest_call_at_a_report_shape_must_be_that_points_timing(change):
    # The recorded fastest calls sit at 128 x 128 x 128, which is point 0.
    report = json.loads(RECORDED_REPORT_JSON)
    fastest = report["sources"]["timing"]["fastest_calls"]["matmul_x_columns"]
    assert (fastest["m"], fastest["n"], fastest["k"]) == (128, 128, 128)
    tampered = resealed(
        lambda r: r["sources"]["timing"]["fastest_calls"]["matmul_x_columns"].update(
            change
        ),
        report,
    )

    assert verify_report(report)["points"] == 9
    with pytest.raises(MatmulBottleneckVerificationError, match="fastest call"):
        verify_report(tampered)
