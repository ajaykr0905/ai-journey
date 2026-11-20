import collections

import numpy
import pytest

from ai_journey.gpu_scan_config import config_from_mapping
from ai_journey.gpu_scan_plan import VENDOR
from ai_journey.gpu_scan_report import measure_report
from ai_journey.gpu_scan_source import S0_HILLIS_STEELE, S1_BLELLOCH, S2_SIMD
from ai_journey.gpu_scan_verify import GpuScanVerificationError, verify_report
from ai_journey.memory_access_report import DIGEST_FIELD, seal_report
from test_gpu_scan_config import make_config_mapping
from test_gpu_scan_correctness import StandInBackend
from test_gpu_scan_report import LOAD, scripted_clock, sealed, stand_in_report


def reseal(report):
    """Seal a changed report again, so only recomputation can catch the change."""

    return seal_report(
        {key: value for key, value in report.items() if key != DIGEST_FIELD}
    )


def outputs(report, case):
    return {
        output["side"]: output
        for output in report["correctness"]["cases"][case]["outputs"]
    }


def set_measured(report):
    report["measured"] = True


def set_boundary(report):
    report["evidence_boundary"]["cuda_kernel_executed"] = True


def set_seed(report):
    report["config"]["seed"] = 79


def set_hypothesis(report):
    report["plan"]["hypotheses"][2]["hypothesis"] += " Probably."


def set_token(report):
    report["environment"]["numpy_version"] = "1.26 4"


def set_load(report):
    report["host_load"]["before"] = [1.0, 2.0]


def set_warnings(report):
    report["warnings"]["count"] = 3


def set_memory(report):
    report["shaders"]["kernels"][2]["limits"]["static_threadgroup_memory_bytes"] = 256


def set_refusal(report):
    report["refusals"] = [dict(report["shaders"]["kernels"][2]["admission"])]


def set_calls(report):
    report["planned_work"]["calls"] += 1


def set_input(report):
    report["correctness"]["cases"][0]["input_sha256"] = "0" * 64


def set_other_bits(report):
    entry = outputs(report, 0)[S0_HILLIS_STEELE]
    entry["replay_bits"], entry["replay_differences"] = False, 3


def set_error(report):
    outputs(report, 2)[S1_BLELLOCH]["max_relative_error"] *= 2


def set_large_error(report):
    entry = outputs(report, 2)[S2_SIMD]
    entry["max_relative_error"] = entry["bound"] * 1.5
    entry["bound_ratio"] = 1.5


def set_deterministic(report):
    outputs(report, 2)[S2_SIMD]["deterministic"] = False


def set_second_run(report):
    outputs(report, 0)[S0_HILLIS_STEELE]["run_sha256"][1] = "1" * 64


def set_integer_run(report):
    outputs(report, 1)[S1_BLELLOCH]["run_sha256"][1] = "2" * 64


def set_vendor_exact(report):
    outputs(report, 1)[VENDOR]["exact_bits"] = False


def set_sample(report):
    report["benchmark"]["points"][0]["sides"][S2_SIMD]["samples_ns"][0] += 1


def set_check_ratio(report):
    report["benchmark"]["points"][1]["checks"][S2_SIMD]["bound_ratio"] = 0.5


def set_same_bits(report):
    report["benchmark"]["points"][1]["checks"][S1_BLELLOCH]["same_bits_as_s0"] = True


def set_copy(report):
    report["benchmark"]["points"][0]["checks"]["stream_copy"] = {
        "same_bits_as_input": False
    }


def set_verdict(report):
    row = report["speedup"]["points"][0]["rows"][1]
    row["to_base"]["verdict"] = "faster"


def set_held(report):
    report["speedup"]["hypotheses"][0]["held_at"] = [0]


def set_elapsed(report):
    report["run"]["elapsed_ns"] = report["run"]["timed_ns"] - 1


def set_summary(report):
    report["summary"]["warnings"] = 1


def set_field(report):
    report["notes"] = "faster than expected"


def set_schema(report):
    report["schema_version"] = 2


def set_cases(report):
    report["correctness"]["cases"] = report["correctness"]["cases"][:2]


def set_config_type(report):
    report["config"]["benchmark"]["repeats"] = "3"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (set_measured, "only the host timer"),
        (set_boundary, "labels or boundary were weakened"),
        (set_seed, "config digest does not match"),
        (set_hypothesis, "does not seal the declared plan"),
        (set_token, "version or platform token"),
        (set_load, "host_load must hold"),
        (set_warnings, "warning counts do not reconcile"),
        (set_memory, "shader section does not recompute"),
        (set_refusal, "refusals do not match"),
        (set_calls, "planned work does not match"),
        (set_input, "does not rebuild from the seed"),
        (set_other_bits, "records other bits, so it needs another digest"),
        (set_error, "does not rebuild from its case"),
        (set_large_error, "within its bound"),
        (set_deterministic, "does not rebuild from its case"),
        (set_second_run, "does not rebuild from its case"),
        (set_integer_run, "exact integer sums with its tail in every run"),
        (set_vendor_exact, "exact bits of torch.cumsum do not match"),
        (set_sample, "does not recompute from its samples"),
        (set_check_ratio, "check does not recompute"),
        (set_same_bits, "cannot hold the S0 bits"),
        (set_copy, "the copy must hold its input"),
        (set_verdict, "do not recompute from the samples"),
        (set_held, "do not recompute from the samples"),
        (set_elapsed, "does not cover its samples"),
        (set_summary, "summary does not reconcile"),
        (set_field, "fields do not match the schema"),
        (set_schema, "unsupported report schema"),
        (set_cases, "correctness cases do not cover the config"),
        (set_config_type, "report failed verification: invalid gpu-scan config"),
    ],
)
def test_a_changed_report_fails_even_when_sealed_again(change, message):
    report = sealed()
    change(report)

    with pytest.raises(GpuScanVerificationError, match=message):
        verify_report(reseal(report))


def test_a_changed_report_without_a_new_seal_fails_on_its_digest():
    report = sealed()
    report["label"] = "host-specific Apple GPU (Metal) measurement"

    with pytest.raises(GpuScanVerificationError, match="digest does not match"):
        verify_report(report)
    with pytest.raises(GpuScanVerificationError, match="must be a JSON object"):
        verify_report([report])


def test_a_report_binds_to_the_config_it_came_from():
    report = sealed()
    mapping = make_config_mapping()

    summary = verify_report(report, config=config_from_mapping(mapping))
    assert summary["config_sha256"] == report["config_sha256"]
    mapping["seed"] = 79
    with pytest.raises(GpuScanVerificationError, match="supplied config"):
        verify_report(report, config=config_from_mapping(mapping))
    with pytest.raises(TypeError, match="GpuScanConfig"):
        verify_report(report, config=mapping)


def test_a_kernel_with_other_bits_verifies_with_its_own_digest_and_error():
    report = stand_in_report(nudge={S1_BLELLOCH})

    summary = verify_report(report)
    entry = outputs(report, 2)[S1_BLELLOCH]
    assert entry["replay_bits"] is False and entry["replay_differences"] > 0
    assert summary["cases_with_other_bits"][S1_BLELLOCH] == [0, 2]


def record_error(report, case, name, error):
    """Record another largest error for one kernel and reconcile the summary."""

    entry = outputs(report, case)[name]
    entry["max_relative_error"] = error
    entry["bound_ratio"] = error / entry["bound"]
    report["summary"]["max_bound_ratio"] = max(
        output["bound_ratio"]
        for record in report["correctness"]["cases"]
        for output in record["outputs"]
        if output["side"] != VENDOR
    )


def hide_the_first_run_error(report, case, name):
    """Let a later run differ and halve the error, with each derived field.

    The first run keeps the replay bits, so only the error of that run,
    computed again from those bits, can show the change.
    """

    entry = outputs(report, case)[name]
    entry["run_sha256"][-1] = "3" * 64
    entry["deterministic"] = False
    nondeterministic = report["summary"]["nondeterministic_cases"]
    nondeterministic[name] = sorted({*nondeterministic[name], case})
    record_error(report, case, name, entry["max_relative_error"] / 2)


@pytest.mark.parametrize("name", [S0_HILLIS_STEELE, S1_BLELLOCH])
def test_a_changed_later_run_cannot_hide_the_error_of_the_replay_run(name):
    report = sealed()
    entry = outputs(report, 0)[name]
    assert entry["replay_bits"] is True and entry["max_relative_error"] > 0.0

    hide_the_first_run_error(report, 0, name)

    with pytest.raises(GpuScanVerificationError, match="less error than its first"):
        verify_report(reseal(report))


def test_the_tamper_of_s0_case_3_of_the_pinned_cases_is_refused():
    # Seven pinned cases in the pinned order and three runs: case 3
    # holds 1000 normal values, and S0 holds the replay bits in each run.
    mapping = make_config_mapping()
    mapping["correctness"] = {
        "cases": [
            [1, "standard_normal"],
            [31, "standard_normal"],
            [32, "small_integers"],
            [1000, "standard_normal"],
            [1024, "standard_normal"],
            [1025, "small_integers"],
            [1048577, "standard_normal"],
        ],
        "runs": 3,
    }
    report = stand_in_report(mapping)
    entry = outputs(report, 3)[S0_HILLIS_STEELE]
    assert entry["deterministic"] is True and entry["replay_bits"] is True
    assert entry["max_relative_error"] > 0.0
    verify_report(report)

    hide_the_first_run_error(report, 3, S0_HILLIS_STEELE)

    assert report["summary"]["nondeterministic_cases"][S0_HILLIS_STEELE] == [3]
    with pytest.raises(GpuScanVerificationError, match="less error than its first"):
        verify_report(reseal(report))


class LaterRunsDiffer(StandInBackend):
    """A stand-in whose later runs of S0 and S1 give other bits, as a GPU can.

    The first run of each case keeps the replay bits. On a case that is not
    all integers, each later run moves the first element two float32 steps
    away from 0, so that run has a larger error than the first run.
    """

    def __init__(self):
        super().__init__()
        self.seen = collections.Counter()

    def scan(self, name, values):
        buffer = super().scan(name, values)
        self.seen[name, values.size] += 1
        whole = numpy.array_equal(values, numpy.round(values))
        if name in (S0_HILLIS_STEELE, S1_BLELLOCH) and not whole:
            if self.seen[name, values.size] > 1:
                away = numpy.float32(numpy.copysign(numpy.inf, buffer[0]))
                buffer[0] = numpy.nextafter(numpy.nextafter(buffer[0], away), away)
        return buffer


def test_a_report_whose_later_runs_gave_other_bits_still_verifies():
    report = measure_report(
        config_from_mapping(make_config_mapping()),
        backend=LaterRunsDiffer(),
        clock=scripted_clock(),
        load_average=lambda: LOAD,
    )
    same = sealed()

    summary = verify_report(report)
    for name in (S0_HILLIS_STEELE, S1_BLELLOCH):
        assert summary["nondeterministic_cases"][name] == [0, 2]
        for case in (0, 2):
            entry, first = outputs(report, case)[name], outputs(same, case)[name]
            assert entry["replay_bits"] is True and entry["deterministic"] is False
            assert entry["run_sha256"][0] == first["run_sha256"][0]
            assert entry["max_relative_error"] > first["max_relative_error"]
    first = outputs(same, 0)[S0_HILLIS_STEELE]["max_relative_error"]
    record_error(report, 0, S0_HILLIS_STEELE, first / 2)
    with pytest.raises(GpuScanVerificationError, match="less error than its first"):
        verify_report(reseal(report))
