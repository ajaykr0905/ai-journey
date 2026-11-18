import json

import pytest

from ai_journey.matmul_bottleneck_config import (
    MatmulBottleneckError,
    MatmulBottleneckVerificationError,
)
from ai_journey.matmul_bottleneck_io import (
    InputPaths,
    load_report,
    report_from_files,
    save_report,
)
from ai_journey.memory_access_plan import canonical_json_bytes
from test_matmul_bottleneck_report import CONFIG, synthetic_report
from test_matmul_bottleneck_sources import recorded_reports, write_reports
from test_matmul_bottleneck_timing import synthetic_timing_mapping
from test_matmul_bottleneck_verify import resealed


def write_inputs(directory, timing=None):
    matmul, ceilings, gap = write_reports(directory, recorded_reports())
    path = directory / "timing.json"
    path.write_text(
        json.dumps(timing or synthetic_timing_mapping(), indent=2), encoding="utf-8"
    )
    return InputPaths(matmul, ceilings, gap, path)


def test_a_report_from_files_saves_loads_and_binds_to_every_input(tmp_path):
    inputs = write_inputs(tmp_path)
    report = report_from_files(inputs, CONFIG)
    output = tmp_path / "bottleneck.json"
    save_report(output, report, protected=inputs.all())

    assert report == synthetic_report()
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"
    loaded, summary = load_report(output, config=CONFIG, inputs=inputs)
    assert loaded == report
    assert summary["sources_checked"] is True
    _, embedded = load_report(output, inputs=inputs)
    assert embedded["sources_checked"] is True
    _, alone = load_report(output)
    assert alone["sources_checked"] is False


def test_a_resealed_report_with_other_loads_fails_only_against_its_inputs(tmp_path):
    inputs = write_inputs(tmp_path)

    def busier(report):
        report["sources"]["timing"]["load_average_after"] = [9.0, 9.0, 9.0]

    output = tmp_path / "bottleneck.json"
    save_report(output, resealed(busier))

    assert load_report(output)[1]["sources_checked"] is False
    with pytest.raises(MatmulBottleneckVerificationError, match="supplied sources"):
        load_report(output, inputs=inputs)


def test_a_timing_input_that_changes_after_the_report_is_caught(tmp_path):
    inputs = write_inputs(tmp_path)
    output = tmp_path / "bottleneck.json"
    save_report(output, report_from_files(inputs, CONFIG))
    timing = synthetic_timing_mapping()
    timing["points"][0]["median_host_ns"] += 1
    inputs.timing.write_text(json.dumps(timing), encoding="utf-8")

    with pytest.raises(MatmulBottleneckVerificationError, match="supplied sources"):
        load_report(output, inputs=inputs)


def test_non_canonical_and_unreadable_files_are_refused(tmp_path):
    report = synthetic_report()
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=1), encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    with pytest.raises(MatmulBottleneckVerificationError, match="not canonical"):
        load_report(pretty)
    with pytest.raises(MatmulBottleneckError, match="not strict UTF-8 JSON"):
        load_report(broken)


def test_the_writer_refuses_an_output_that_aliases_an_input(tmp_path):
    inputs = write_inputs(tmp_path)

    with pytest.raises(MatmulBottleneckError, match="alias a protected input"):
        save_report(inputs.gap_report, synthetic_report(), protected=inputs.all())
