import copy
import json

import numpy
import pytest

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import seal_report
from ai_journey.metal_liger_io import load_report, read_ceiling, save_report
from ai_journey.kernel_metal_report import MEASUREMENT_LABEL, MEASUREMENT_SCOPE
from ai_journey.memory_access_benchmark import TIMER
from ai_journey.metal_liger_report import GPU_ONLY, MEASUREMENT_BOUNDARY
from ai_journey.metal_liger_source import MetalLigerError
from ai_journey.metal_liger_verify import MetalLigerVerificationError, verify_report
from test_metal_fused_ceiling import recorded_ceiling, write_recorded
from test_metal_liger_report import small_config, synthetic_report


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return synthetic_report(tmp_path_factory.mktemp("ceiling"))


def test_a_sealed_report_verifies_against_its_config_and_ceiling(report, tmp_path):
    summary = verify_report(
        report, config=small_config(), ceiling=recorded_ceiling(tmp_path)
    )

    assert summary["measured"] is False
    assert summary["label"] == "synthetic timing, not a measurement"
    assert summary["report_sha256"] == report["report_sha256"]
    assert summary["benchmark_points"] == 3 and summary["correctness_cases"] == 4
    assert len(summary["points"]) == 3


def test_the_verifier_needs_neither_torch_nor_the_run_reference(report, monkeypatch):
    import sys

    import ai_journey.metal_liger_verify as verifier

    monkeypatch.setitem(sys.modules, "torch", None)
    borrowed = {"liger_inputs", "liger_reference", "exp64", "log64", "silu64"}
    assert not borrowed & set(vars(verifier))
    assert verify_report(copy.deepcopy(report))["measured"] is False


def test_the_verifier_ignores_a_libm_that_rounds_exp_and_log_otherwise(
    report, monkeypatch
):
    real_exp, real_log = numpy.exp, numpy.log

    def nudge(function):
        def other(values, *args, **kwargs):
            result = function(values, *args, **kwargs)
            if isinstance(result, numpy.ndarray) and result.dtype == numpy.float64:
                result = numpy.nextafter(result, numpy.inf)
            return result

        return other

    monkeypatch.setattr(numpy, "exp", nudge(real_exp))
    monkeypatch.setattr(numpy, "log", nudge(real_log))
    assert verify_report(copy.deepcopy(report))["measured"] is False


def _reseal(report, change):
    tampered = copy.deepcopy(report)
    change(tampered)
    tampered.pop("report_sha256")
    return seal_report(tampered)


def _promote(report):
    """Relabel a synthetic report as a measurement, with every label to match."""

    report.update(
        measured=True,
        timer=TIMER,
        backend="torch.mps.compile_shader",
        measurement_scope=MEASUREMENT_SCOPE,
        label=MEASUREMENT_LABEL,
        evidence_boundary=dict(MEASUREMENT_BOUNDARY),
    )
    report["summary"]["label"] = MEASUREMENT_LABEL
    # A measurement must come from macOS. A version that no ceiling records keeps
    # this case on the same-environment check on every host.
    report["environment"].update(platform_system="Darwin", macos_version="0.0")


def _first_kernel_side(report):
    point = report["benchmark"]["points"][0]
    return point["sides"][point["side_order"][0]]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(measured=True), "make a measurement"),
        (_promote, "same environment"),
        (
            lambda r: r["evidence_boundary"].update(cuda_kernel_executed=True),
            "weakened",
        ),
        (
            lambda r: r["evidence_boundary"].update(liger_kernel_code_executed=True),
            "weakened",
        ),
        (lambda r: r["plan"]["steps"][0].update(predicted=["slower"]), "declared plan"),
        (lambda r: r["config"].update(seed=81), "config digest"),
        (
            lambda r: r["correctness"]["cases"][0].update(main_sha256="0" * 64),
            "rebuild from the seed",
        ),
        (
            lambda r: r["correctness"]["cases"][2]["output"].update(
                max_bound_ratio=1.5
            ),
            "breaks its bound",
        ),
        (
            lambda r: r["correctness"]["cases"][0].update(bound_sha256="1" * 64),
            "rebuild from the seed",
        ),
        (lambda r: _first_kernel_side(r)["samples_ns"].__setitem__(0, 1), "recompute"),
        (
            lambda r: _first_kernel_side(r).update(effective_gb_per_second=999.0),
            "recompute",
        ),
        (
            lambda r: r["speedup"]["points"][0]["to_torch"].update(verdict="faster"),
            "speedup rows",
        ),
        (lambda r: r["speedup"]["points"][0].update(held=True), "speedup rows"),
        (
            lambda r: r["speedup"].update(ceiling_gb_per_second=500.0),
            "speedup rows",
        ),
        (lambda r: r["ceiling"].update(gb_per_second=500.0), "speedup rows"),
        (
            lambda r: r["benchmark"]["traffic_model"].update(rmsnorm="none"),
            "benchmark section",
        ),
        (
            lambda r: r["benchmark"]["points"][2].update(unwritten_softmax_bytes=0),
            "recompute",
        ),
        (
            lambda r: r["math_probes"]["log"].update(max_relative_error=1e-3),
            "declared rule",
        ),
        (lambda r: r["math_probes"]["rsqrt"].update(holds=False), "declared rule"),
        (
            lambda r: r["shaders"]["kernels"][0]["limits"].update(
                static_threadgroup_memory_bytes=0
            ),
            "shader section does not recompute",
        ),
        (lambda r: r["verification"]["gpu_only"].pop(), "verification scope"),
        (lambda r: r["summary"].update(warnings=3), "summary"),
        (lambda r: r["run"].update(elapsed_ns=1), "within the budget"),
        (lambda r: r.pop("summary"), "schema"),
    ],
)
def test_a_tampered_and_resealed_report_is_rejected(report, change, message):
    with pytest.raises(MetalLigerVerificationError, match=message):
        verify_report(_reseal(report, change))


def test_a_change_without_a_new_seal_breaks_the_digest(report):
    tampered = copy.deepcopy(report)
    tampered["summary"]["warnings"] = 1

    with pytest.raises(MetalLigerVerificationError, match="digest"):
        verify_report(tampered)


def test_the_report_names_what_only_the_gpu_saw(report):
    assert report["verification"]["gpu_only"] == list(GPU_ONLY)
    assert any("samples_ns" in entry for entry in GPU_ONLY)
    assert any("identical_repeats" in entry for entry in GPU_ONLY)


def test_io_publishes_only_canonical_verified_reports(report, tmp_path):
    path = tmp_path / "report.json"
    config = small_config()
    ceilings = write_recorded(tmp_path)

    summary = save_report(path, report, config=config, protected=(ceilings,))
    loaded, again = load_report(path, config=config, ceiling=read_ceiling(ceilings))
    assert loaded == report and again == summary
    assert path.read_bytes() == canonical_json_bytes(report) + b"\n"

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=1), encoding="utf-8")
    with pytest.raises(MetalLigerVerificationError, match="canonical"):
        load_report(pretty)
    with pytest.raises(MetalLigerError, match="ceiling report was refused"):
        read_ceiling(tmp_path / "missing.json")
    with pytest.raises(MetalLigerError):
        save_report(ceilings, report, protected=(ceilings,))
    other = small_config(repeats=6)
    with pytest.raises(MetalLigerVerificationError, match="supplied config"):
        load_report(path, config=other)
