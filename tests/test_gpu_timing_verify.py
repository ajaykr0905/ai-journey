import copy
import functools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.gpu_timing_config import config_from_mapping
from ai_journey.gpu_timing_report import measure_report
from ai_journey.gpu_timing_verify import GpuTimingVerificationError, verify_report
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import DIGEST_FIELD, seal_report
from test_gpu_timing_config import small
from test_gpu_timing_report import stand_in_report
from test_gpu_timing_runtime import timing_runtime
from test_kernel_metal_runtime import requires_metal

ROOT = Path(__file__).resolve().parents[1]
X_ROWS = "matmul_x_rows"


@functools.cache
def _sealed():
    return stand_in_report()


def report():
    return copy.deepcopy(_sealed())


def reseal(payload):
    unsigned = {key: value for key, value in payload.items() if key != DIGEST_FIELD}
    return seal_report(unsigned)


def side(payload, index=1, name=X_ROWS):
    return payload["benchmark"]["points"][index]["sides"][name]


def test_a_stand_in_report_verifies_as_synthetic():
    summary = verify_report(report(), config=config_from_mapping(small()))

    assert summary["measured"] is False
    assert summary["scope"] == "synthetic_backend_or_clock"
    assert summary["sweep_points"] == 3
    assert summary["timed_samples"] == 10 * 3 * 3
    assert summary["load_average_before"] == [2.5, 3.0, 3.25]


def test_a_report_from_another_config_is_refused():
    with pytest.raises(GpuTimingVerificationError, match="given config"):
        verify_report(report(), config=config_from_mapping(small(seed=1)))


def test_an_edit_without_a_new_seal_is_refused():
    payload = report()
    payload["summary"]["sweep_points"] = 4

    with pytest.raises(GpuTimingVerificationError, match="digest"):
        verify_report(payload)


def edit_sample(payload):
    side(payload)["clocks"]["host_synchronized"]["samples_ns"][0] += 1


def edit_event_ms(payload):
    side(payload)["clocks"]["mps_event"]["elapsed_ms"][0] *= 2


def edit_event_ns(payload):
    side(payload)["clocks"]["mps_event"]["samples_ns"][0] += 1


def edit_domain(payload):
    side(payload)["clocks"]["mps_event"]["domain"] = "host_perf_counter_ns"


def drop_raw_ms(payload):
    del side(payload)["clocks"]["mps_event"]["elapsed_ms"]


def edit_repeats(payload):
    side(payload)["repeats"] = 4


def edit_fit(payload):
    fit = payload["analysis"]["fits"][X_ROWS]["mps_event"]
    fit["intercept_ns"] = fit["intercept_ns"] * (1 + 2**-40)


def edit_split(payload):
    entry = payload["analysis"]["points"][0]["sides"][X_ROWS]["clocks"]
    entry["mps_event"]["regime"] = "size_dependent_dominant"


def hide_cross_domain(payload):
    entry = payload["analysis"]["points"][0]["sides"][X_ROWS]["host_minus_event"]
    entry["cross_domain"] = False


def edit_fraction(payload):
    entry = payload["analysis"]["points"][-1]["sides"][X_ROWS]
    entry["event_fraction_of_fma_ceiling"] *= 10


def edit_summary(payload):
    payload["summary"]["sides"][X_ROWS]["first_size_dominant_index"] = None


def edit_launch(payload):
    payload["benchmark"]["points"][1]["launches"]["x_rows"]["groups"][0] += 1


def edit_disagreement(payload):
    payload["benchmark"]["points"][2]["torch_disagreement"] = 1.0


def claim_measured(payload):
    payload["measured"] = True


def claim_real_backend_and_timer(payload):
    # A real backend and the host timer make a measurement, never synthetic.
    payload["backend"] = "torch.mps.compile_shader"
    payload["timer"] = "time.perf_counter_ns"


def edit_boundary(payload):
    payload["evidence_boundary"]["nvidia_gpu_used"] = True


def edit_load(payload):
    payload["host_load"]["before"] = [1, 2, 3]


def edit_environment(payload):
    payload["environment"]["platform_machine"] = "host name with spaces"


def edit_empty_source(payload):
    payload["shaders"]["empty_source"] += "\n"


def edit_matmul_digest(payload):
    payload["shaders"]["matmul_source_sha256"] = "0" * 64


def edit_clock_rule(payload):
    payload["clocks"]["definitions"]["host_dispatch"] = "with a synchronize"


def edit_planned(payload):
    payload["planned_work"]["launches"] += 1


def edit_timed(payload):
    payload["run"]["timed_host_ns"] += 1


def shrink_elapsed(payload):
    payload["run"]["elapsed_ns"] = 1


def edit_integer_type(payload):
    side(payload)["clocks"]["host_dispatch"]["timing"]["count"] = 3.0


@pytest.mark.parametrize(
    "edit",
    [
        edit_sample,
        edit_event_ms,
        edit_event_ns,
        edit_domain,
        drop_raw_ms,
        edit_repeats,
        edit_fit,
        edit_split,
        hide_cross_domain,
        edit_fraction,
        edit_summary,
        edit_launch,
        edit_disagreement,
        claim_measured,
        claim_real_backend_and_timer,
        edit_boundary,
        edit_load,
        edit_environment,
        edit_empty_source,
        edit_matmul_digest,
        edit_clock_rule,
        edit_planned,
        edit_timed,
        shrink_elapsed,
        edit_integer_type,
    ],
)
def test_a_resealed_edit_fails_recomputation(edit):
    payload = report()
    edit(payload)

    with pytest.raises(GpuTimingVerificationError):
        verify_report(reseal(payload))


def test_a_dispatch_median_above_the_synchronized_median_is_refused():
    payload = report()
    clocks = side(payload)["clocks"]
    synchronized, dispatch = clocks["host_synchronized"], clocks["host_dispatch"]
    synchronized["samples_ns"], dispatch["samples_ns"] = (
        dispatch["samples_ns"],
        synchronized["samples_ns"],
    )
    synchronized["timing"], dispatch["timing"] = (
        dispatch["timing"],
        synchronized["timing"],
    )

    with pytest.raises(GpuTimingVerificationError, match="dispatch median exceeds"):
        verify_report(reseal(payload))


def test_a_missing_or_extra_field_is_refused():
    payload = report()
    payload["extra"] = 1
    with pytest.raises(GpuTimingVerificationError, match="schema"):
        verify_report(reseal(payload))
    with pytest.raises(GpuTimingVerificationError, match="JSON object"):
        verify_report([])


def test_the_verifier_never_imports_torch():
    script = (
        "import sys; sys.modules['torch'] = None\n"
        "import ai_journey.gpu_timing_verify\n"
        "print('torch' in sys.modules and sys.modules['torch'] is not None)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


@requires_metal
def test_a_short_run_on_this_gpu_verifies_as_a_measurement():
    config = config_from_mapping(small())

    sealed = measure_report(config, backend=timing_runtime())
    summary = verify_report(sealed, config=config)

    assert summary["measured"] is True
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"


def test_key_order_does_not_matter_because_files_hold_sorted_keys():
    payload = json.loads(canonical_json_bytes(report()))

    assert verify_report(payload)["measured"] is False
