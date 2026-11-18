import copy
import json
from pathlib import Path

import pytest

from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.profile_trace_config import (
    MAX_CAPTURE_FLOPS,
    AnalysisControls,
    Day74Ceilings,
    ProfileTraceConfig,
    config_from_mapping,
    load_config,
)
from ai_journey.profile_trace_events import ProfileTraceError

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config" / "day-75-profile-trace.json"


def make_config_mapping(**changes):
    """A small valid config for tests that should not touch the pinned file."""

    mapping = {
        "schema_version": 1,
        "name": "profile-trace-test",
        "seed": 75,
        "group_size": [32, 8],
        "shapes": [[64, 48, 32], [33, 17, 9]],
        "warmup": 1,
        "iterations": 3,
        "analysis": {
            "top_n": 4,
            "gap_count": 2,
            "timeline_width": 32,
            "timeline_depth": 3,
        },
        "day_74_ceilings": {
            "fma_gflops_per_second": 4618.0,
            "bandwidth_gb_per_second": 100.7,
            "torch_matmul_gflops_per_second": 3963.0,
        },
    }
    mapping.update(changes)
    return mapping


def test_the_pinned_config_profiles_three_shapes_against_the_day_74_ceilings():
    config = load_config(PINNED)

    assert config.name == "day-75-profile-trace"
    assert config.group_size == (32, 8)
    assert config.shapes == (
        MatmulShape(256, 256, 256),
        MatmulShape(1024, 1024, 1024),
        MatmulShape(2048, 2048, 2048),
    )
    assert (config.warmup, config.iterations) == (2, 5)
    assert config.analysis == AnalysisControls(8, 4, 96, 3)
    assert config.ceilings == Day74Ceilings(4618.0, 100.7, 3963.0)
    assert config.to_mapping() == json.loads(PINNED.read_text())
    assert config.capture_flops() == 21 * sum(2 * n**3 for n in (256, 1024, 2048))
    assert len(config.fingerprint()) == 64


def test_integer_ceilings_become_floats_and_round_trip():
    mapping = make_config_mapping()
    mapping["day_74_ceilings"]["bandwidth_gb_per_second"] = 100
    config = config_from_mapping(mapping)
    assert config.ceilings.bandwidth_gb_per_second == 100.0
    assert config_from_mapping(config.to_mapping()) == config


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("schema_version",), 2, "schema_version"),
        (("name",), "Bad Name", "lowercase slug"),
        (("seed",), True, "seed"),
        (("seed",), -1, "seed"),
        (("group_size",), [32], "group_size"),
        (("group_size",), (32, 8), "group_size"),
        (("group_size",), [64, 32], "1024"),
        (("shapes",), [], "1 to 3 shapes"),
        (("shapes",), [[1, 1, 1]] * 2, "repeat"),
        (("shapes",), [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], "1 to 3"),
        (("shapes",), [[1, 1]], "triples"),
        (("shapes",), [[1, 1, 0]], "k"),
        (("shapes",), [[1, 1, True]], "k"),
        (("shapes",), {}, "JSON array"),
        (("warmup",), 11, "warmup"),
        (("iterations",), 1, "iterations"),
        (("iterations",), 21, "iterations"),
        (("analysis", "top_n"), 0, "top_n"),
        (("analysis", "gap_count"), 65, "gap_count"),
        (("analysis", "timeline_width"), 15, "timeline_width"),
        (("analysis", "timeline_depth"), 9, "timeline_depth"),
        (("day_74_ceilings", "fma_gflops_per_second"), 0, r"\(0, "),
        (("day_74_ceilings", "fma_gflops_per_second"), True, "finite JSON number"),
        (("day_74_ceilings", "fma_gflops_per_second"), "1", "finite JSON number"),
        (("day_74_ceilings", "bandwidth_gb_per_second"), 2e6, r"\(0, "),
    ],
)
def test_invalid_controls_are_refused(path, value, message):
    mapping = make_config_mapping()
    target = mapping
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ProfileTraceError, match=message):
        config_from_mapping(mapping)


@pytest.mark.parametrize("section", [None, "analysis", "day_74_ceilings"])
def test_unknown_and_missing_fields_are_refused(section):
    for change in ("extra", "missing"):
        mapping = make_config_mapping()
        target = mapping if section is None else mapping[section]
        if change == "extra":
            target["other"] = 1
        else:
            target.pop(next(iter(target)))
        with pytest.raises(ProfileTraceError, match="fields do not match"):
            config_from_mapping(mapping)


def test_the_capture_work_is_bounded():
    big = make_config_mapping(
        shapes=[[8192, 8192, 8192], [8192, 8192, 4096]], warmup=10, iterations=20
    )
    with pytest.raises(ProfileTraceError, match=f"bound is {MAX_CAPTURE_FLOPS}"):
        config_from_mapping(big)


def test_direct_construction_checks_types():
    config = config_from_mapping(make_config_mapping())
    fields = dict(config.__dict__)
    for key, value in (
        ("shapes", list(config.shapes)),
        ("shapes", ((1, 1, 1),)),
        ("analysis", {}),
        ("ceilings", {}),
    ):
        with pytest.raises(ProfileTraceError):
            ProfileTraceConfig(**{**fields, key: value})
    with pytest.raises(ProfileTraceError, match="must be a float"):
        Day74Ceilings(1, 1.0, 1.0)


def test_files_go_through_the_bounded_strict_reader(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"name": "a", "name": "b"}')
    with pytest.raises(ProfileTraceError, match="duplicate"):
        load_config(path)
    path.write_text(" " * (64 * 1024 + 1))
    with pytest.raises(ProfileTraceError, match="exceeds"):
        load_config(path)
    path.write_text(json.dumps(copy.deepcopy(make_config_mapping())))
    assert load_config(path).name == "profile-trace-test"
