import json

import pytest

from ai_journey.gpu_timing_clocks import DISPATCH, EVENT, SYNCHRONIZED
from ai_journey.matmul_bottleneck_cli import EXIT_INVALID, EXIT_OK, EXIT_REJECTED, main
from ai_journey.matmul_bottleneck_config import (
    MatmulBottleneckError,
    MatmulBottleneckVerificationError,
)
from ai_journey.matmul_bottleneck_gpu_timing import (
    load_timing_source,
    timing_from_gpu_timing_report,
)
from ai_journey.matmul_bottleneck_io import InputPaths, load_report
from ai_journey.matmul_bottleneck_report import TIMING_SOURCE, build_report
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_runtime import SIDES
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import DIGEST_FIELD, seal_report
from test_gpu_timing_recorded import recorded as recorded_gpu_timing
from test_gpu_timing_report import stand_in_report
from test_matmul_bottleneck_cli import CONFIG, classify, input_arguments
from test_matmul_bottleneck_io import write_inputs
from test_matmul_bottleneck_recorded import CONFIG as BOTTLENECK_CONFIG
from test_matmul_bottleneck_sources import (
    recorded_reports,
    recorded_sources,
    synthetic_timing,
    write_reports,
)

# The recorded Day 74 gap holds 256 x 256 x 256 and 129 x 65 x 1000, so a
# sweep with both shapes covers every gap point.
GAP_SWEEP = [[1, 1, 1], [129, 65, 1000], [256, 256, 256]]


def gap_timing_report(**changes):
    """A synthetic gpu-timing report that names the Day 74 fixture's host.

    The stand-in reads the environment of the test host, so the test pins it
    to that of the recorded Day 74 reports and seals the report again. It
    stays synthetic, as its backend and timer say.
    """

    report = stand_in_report(sweep__shapes=GAP_SWEEP, **changes)
    del report[DIGEST_FIELD]
    report["environment"] = dict(recorded_reports()[0]["environment"])
    return seal_report(report)


def write_gap_inputs(directory, report):
    paths = write_reports(directory, recorded_reports())
    timing = directory / "gpu-timing.json"
    timing.write_bytes(canonical_json_bytes(report) + b"\n")
    return InputPaths(*paths, timing)


def test_each_side_takes_one_median_from_each_clock():
    report = gap_timing_report()

    timing = timing_from_gpu_timing_report(report, recorded_sources())

    assert timing.measured is False
    assert timing.repeats == 3
    assert timing.load_average_before == (2.5, 3.0, 3.25)
    assert report[DIGEST_FIELD] in timing.producer
    assert len(timing.entries) == len(GAP_SWEEP) * len(SIDES)
    for point in report["benchmark"]["points"]:
        shape = matmul_shape(point["m"], point["n"], point["k"])
        for side in SIDES:
            clocks = point["sides"][side]["clocks"]
            entry = timing.entry(shape, side)
            assert (entry.gpu_ns, entry.host_ns, entry.dispatch_ns) == tuple(
                clocks[clock]["timing"]["median_ns"]
                for clock in (EVENT, SYNCHRONIZED, DISPATCH)
            )


def test_classify_joins_a_gpu_timing_report_and_verify_rebuilds_it(tmp_path, capsys):
    report = gap_timing_report()
    inputs = write_gap_inputs(tmp_path, report)
    output = tmp_path / "bottleneck.json"

    assert main(classify(inputs, output)) == EXIT_OK
    assert "matmul_bottleneck_synthetic_timing=" in capsys.readouterr().out
    sealed, summary = load_report(output, inputs=inputs)
    source = sealed["sources"][TIMING_SOURCE]
    assert summary["sources_checked"] is True and summary["measured"] is False
    assert source["entries"] == 9 and source["entries_used"] == 4
    assert source["producer"].startswith(
        f"ai-journey-gpu-timing report {report[DIGEST_FIELD]}:"
    )
    assert sealed == build_report(
        recorded_sources(),
        timing_from_gpu_timing_report(report, recorded_sources()),
        BOTTLENECK_CONFIG,
    )
    verify = ["verify", "--report", str(output), *input_arguments(inputs)]
    assert main([*verify, "--config", str(CONFIG)]) == EXIT_OK


def test_a_tampered_gpu_timing_report_is_rejected_before_classify(tmp_path, capsys):
    report = gap_timing_report()
    clocks = report["benchmark"]["points"][1]["sides"]["matmul_x_rows"]["clocks"]
    clocks[SYNCHRONIZED]["timing"]["median_ns"] += 1
    inputs = write_gap_inputs(tmp_path, report)

    assert main(classify(inputs, tmp_path / "out.json")) == EXIT_REJECTED
    assert "rejected: gpu-timing report:" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_a_reformatted_gpu_timing_report_is_rejected(tmp_path):
    inputs = write_gap_inputs(tmp_path, gap_timing_report())
    inputs.timing.write_text(json.dumps(gap_timing_report(), indent=1))

    with pytest.raises(MatmulBottleneckVerificationError, match="not canonical"):
        load_timing_source(inputs.timing, recorded_sources())


def test_a_gpu_timing_report_with_other_threadgroups_does_not_pair():
    report = gap_timing_report(launch__group_size=[16, 16])

    with pytest.raises(MatmulBottleneckError, match="threadgroups of"):
        timing_from_gpu_timing_report(report, recorded_sources())


def test_a_gpu_timing_report_needs_every_gap_shape(tmp_path, capsys):
    # The recorded real report times 256 x 256 x 256 but not 129 x 65 x 1000.
    report = recorded_gpu_timing()
    timing = timing_from_gpu_timing_report(report, recorded_sources())
    inputs = write_gap_inputs(tmp_path, report)

    assert timing.measured is True
    assert dict(timing.environment)["platform_system"] == "Darwin"
    assert main(classify(inputs, tmp_path / "out.json")) == EXIT_INVALID
    assert "no matmul_x_columns entry at 129 x 65 x 1000" in capsys.readouterr().err


def test_the_declared_timing_schema_still_loads_through_the_same_path(tmp_path):
    inputs = write_inputs(tmp_path)

    loaded = load_timing_source(inputs.timing, recorded_sources())

    assert loaded == synthetic_timing()
