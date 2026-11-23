import dataclasses
import json

import pytest

from ai_journey.kernel_metal_report import ENVIRONMENT_FIELDS, host_environment
from ai_journey.metal_fused_ceiling import (
    DEFINITION,
    SOURCE,
    load_ceiling,
)
from ai_journey.metal_fused_source import MetalFusedError
from test_gpu_ceiling_recorded import RECORDED, RECORDED_REPORT_JSON


def write_recorded(directory):
    path = directory / "ceilings.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")
    return path


def recorded_ceiling(directory):
    return load_ceiling(write_recorded(directory))


def host_drift(ceiling):
    """The environment fields where this host differs from the ceiling's host.

    A measured run refuses a ceiling from another environment, so a real GPU
    test that pairs this host with the recorded ceiling can run only where
    this list is empty, for example not after an operating system update.
    """

    import torch

    environment = host_environment(torch.__version__)
    return [
        f"{field} {ceiling.environment[field]} != {environment.get(field)}"
        for field in ENVIRONMENT_FIELDS
        if environment.get(field) != ceiling.environment[field]
    ]


def test_the_recorded_day_74_report_gives_its_measured_bandwidth_ceiling(tmp_path):
    ceiling = recorded_ceiling(tmp_path)

    assert ceiling.measured is True
    assert ceiling.label == "host-specific Apple GPU (Metal) measurement"
    assert (
        ceiling.gb_per_second == RECORDED["summary"]["bandwidth_ceiling_gb_per_second"]
    )
    assert round(ceiling.gb_per_second, 2) == 82.28
    assert (ceiling.kernel, ceiling.buffer_bytes) == ("stream_read_reduce", 1 << 22)
    assert ceiling.report_sha256 == RECORDED["report_sha256"]
    assert ceiling.load_average_before == RECORDED["host_load"]["before"]
    assert set(ceiling.environment) == set(ENVIRONMENT_FIELDS)
    assert ceiling.same_environment(dict(RECORDED["environment"]))
    other = {**RECORDED["environment"], "torch_version": "2.16.0"}
    assert not ceiling.same_environment(other)
    mapping = ceiling.to_mapping()
    assert mapping["source"] == SOURCE and mapping["definition"] == DEFINITION
    assert mapping["gb_per_second"] == ceiling.gb_per_second


def test_a_tampered_ceiling_report_sets_no_ceiling(tmp_path):
    report = json.loads(RECORDED_REPORT_JSON)
    report["summary"]["bandwidth_ceiling_gb_per_second"] = 500.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report), encoding="ascii")

    with pytest.raises(MetalFusedError, match="ceiling report was refused"):
        load_ceiling(path)
    with pytest.raises(MetalFusedError, match="ceiling report was refused"):
        load_ceiling(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"report_sha256": "x"}, "SHA-256"),
        ({"measured": False}, "label must match"),
        ({"gb_per_second": 0.0}, "positive float"),
        ({"gb_per_second": 82}, "positive float"),
        ({"kernel": "fma_chains_8"}, "stream kernel"),
        ({"buffer_bytes": 0}, "positive byte count"),
        ({"load_average_before": [1.0, 2.0]}, "three nonnegative floats"),
        ({"environment": {"torch_version": "2.15.0"}}, "host fields"),
    ],
)
def test_a_ceiling_refuses_an_invalid_field(tmp_path, changes, message):
    ceiling = recorded_ceiling(tmp_path)

    with pytest.raises(MetalFusedError, match=message):
        dataclasses.replace(ceiling, **changes)
