import copy
import json
from pathlib import Path

import pytest

from ai_journey.metal_fused_config import (
    SIDES_PER_POINT,
    MetalFusedConfig,
    TimingShape,
    compulsory_bytes,
    config_from_mapping,
    load_config,
)
from ai_journey.metal_fused_plan import plan_sha256
from ai_journey.metal_fused_reference import CORRECTNESS_GAP, RowLayout, ragged_layout
from ai_journey.metal_fused_source import MetalFusedError, slots_used

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-79-metal-fused.json"


def make_config_mapping(**sections):
    """A small valid config: two cases and two points, for quick tests."""

    mapping = {
        "schema_version": 1,
        "name": "metal-fused-test",
        "seed": 79,
        "plan_sha256": plan_sha256(),
        "correctness": {
            "cases": [
                {"lengths": [1, 31, 33], "inputs": "uniform_80"},
                {"rows": 2, "cols": 64, "inputs": "offset_plus_1000"},
            ],
            "repeats": 2,
        },
        "benchmark": {
            "inputs": "normal",
            "shapes": [{"rows": 8, "cols": 64}, {"rows": 4, "max_length": 100}],
            "warmup": 1,
            "repeats": 5,
            "target_bytes_per_sample": 4096,
            "max_runtime_seconds": 60,
        },
        "comparison": {"min_faster_ratio": 1.1},
    }
    for name, changes in sections.items():
        if isinstance(changes, dict):
            mapping[name] = {**mapping[name], **changes}
        else:
            mapping[name] = changes
    return mapping


def make_config(**sections):
    return config_from_mapping(make_config_mapping(**sections))


def test_the_pinned_config_runs_the_declared_plan_on_every_named_shape():
    config = load_config(PINNED)

    assert config.plan_sha256 == plan_sha256()
    assert [case.to_mapping() for case in config.correctness.cases][-1] == {
        "lengths": [1, 31, 33, 1000, 4096, 16384, 2049, 7],
        "inputs": "uniform_80",
    }
    assert [shape.to_mapping() for shape in config.benchmark.shapes] == [
        {"rows": 4096, "cols": 128},
        {"rows": 4096, "cols": 1024},
        {"rows": 1024, "cols": 4096},
        {"rows": 256, "cols": 16384},
        {"rows": 2048, "max_length": 4096},
    ]
    assert config.comparison.min_faster_ratio == 1.1
    assert config.largest_group() == 1024
    assert config_from_mapping(config.to_mapping()) == config
    payload = json.loads(PINNED.read_text(encoding="utf-8"))
    assert config.to_mapping() == payload


def test_the_pinned_work_is_counted_before_anything_runs():
    config = load_config(PINNED)
    work = config.planned_work().to_mapping()

    layouts = config.timing_layouts()
    assert [layout.group_size for layout in layouts] == [32, 64, 256, 1024, 256]
    assert layouts[-1] == ragged_layout(79, 4, 2048, 4096)
    assert work["correctness_cases"] == 8
    assert work["correctness_launches"] == 8 * 2 * 6 + 1
    assert work["exp_probe_arguments"] == 1 << 18
    assert work["timed_samples"] == SIDES_PER_POINT * 5 * 50
    batches = [
        -(-config.benchmark.target_bytes_per_sample // compulsory_bytes(layout))
        for layout in layouts
    ]
    assert batches[0] == 64 and batches[1:4] == [8, 8, 8]
    assert work["calls"] == sum(5 * 55 * batch for batch in batches)
    assert work["kernel_launches"] == sum(7 * 55 * batch for batch in batches)


@pytest.mark.parametrize("longest", [33, 100, 512, 513, 1000, 4096, 9000, 16384])
def test_the_longest_row_above_a_simd_group_fills_two_or_more_slots(longest):
    layout = RowLayout((1, longest), CORRECTNESS_GAP)

    assert slots_used(longest, layout.group_size) >= 2


def test_cases_use_the_gapped_layout_and_keep_their_spelling():
    config = make_config()
    ragged, dense = config.correctness.cases

    assert ragged.ragged and not dense.ragged
    assert ragged.layout().gap == CORRECTNESS_GAP
    assert dense.layout().lengths == (64, 64)
    assert dense.to_mapping() == {"rows": 2, "cols": 64, "inputs": "offset_plus_1000"}


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        (
            [
                {"lengths": [2, 31, 33], "inputs": "uniform_80"},
                {"rows": 1, "cols": 8, "inputs": "offset_plus_1000"},
            ],
            "a row of 1 element",
        ),
        (
            [
                {"lengths": [1, 32, 33], "inputs": "uniform_80"},
                {"rows": 1, "cols": 64, "inputs": "offset_plus_1000"},
            ],
            "shorter than a SIMD group",
        ),
        (
            [
                {"lengths": [1, 31, 64], "inputs": "uniform_80"},
                {"rows": 1, "cols": 96, "inputs": "offset_plus_1000"},
            ],
            "not a multiple of 32",
        ),
        (
            [
                {"rows": 1, "cols": 1, "inputs": "uniform_80"},
                {"rows": 1, "cols": 31, "inputs": "uniform_80"},
                {"rows": 1, "cols": 33, "inputs": "offset_plus_1000"},
            ],
            "a ragged case",
        ),
        (
            [
                {"lengths": [1, 31, 33], "inputs": "normal"},
                {"rows": 1, "cols": 8, "inputs": "offset_plus_1000"},
            ],
            "a uniform_80 case",
        ),
        (
            [
                {"lengths": [1, 31, 33], "inputs": "uniform_80"},
                {"rows": 1, "cols": 8, "inputs": "normal"},
            ],
            "an offset case",
        ),
    ],
)
def test_the_cases_must_cover_every_path_of_the_kernels(cases, message):
    with pytest.raises(MetalFusedError, match=message):
        make_config(correctness={"cases": cases})


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ({"plan_sha256": "0" * 64}, "does not name the declared plan"),
        ({"plan_sha256": "X"}, "SHA-256"),
        ({"name": "Day 79"}, "lowercase slug"),
        ({"seed": -1}, "seed"),
        ({"schema_version": 2}, "schema_version"),
        ({"benchmark": {"repeats": 6}}, "multiple of the 5 timed sides"),
        ({"benchmark": {"repeats": 55}}, "repeats"),
        ({"benchmark": {"warmup": 11}}, "warmup"),
        ({"benchmark": {"inputs": "cauchy"}}, "inputs must be one of"),
        ({"benchmark": {"target_bytes_per_sample": 512}}, "target_bytes"),
        ({"benchmark": {"target_bytes_per_sample": 1 << 30}}, "calls"),
        ({"benchmark": {"max_runtime_seconds": 601}}, "max_runtime_seconds"),
        ({"benchmark": {"shapes": []}}, "1 to 8 shapes"),
        ({"benchmark": {"shapes": [{"rows": 2}]}}, "benchmark shape fields"),
        ({"benchmark": {"shapes": [{"rows": 2, "cols": 16385}]}}, "row length"),
        ({"correctness": {"repeats": 1}}, "repeats"),
        ({"correctness": {"cases": []}}, "1 to 16 cases"),
        ({"comparison": {"min_faster_ratio": 1.0}}, "min_faster_ratio"),
        ({"comparison": {"min_faster_ratio": 1}}, "finite float"),
        ({"comparison": {"min_faster_ratio": 1.1, "x": 1}}, "comparison fields"),
    ],
)
def test_a_config_refuses_each_invalid_control(sections, message):
    mapping = make_config_mapping()
    for name, changes in sections.items():
        if isinstance(changes, dict) and name in ("benchmark", "correctness"):
            mapping[name] = {**mapping[name], **changes}
        else:
            mapping[name] = changes

    with pytest.raises(MetalFusedError, match=message):
        config_from_mapping(mapping)


@pytest.mark.parametrize(
    "case",
    [
        {"rows": True, "cols": 4, "inputs": "normal"},
        {"rows": 2, "cols": 4.0, "inputs": "normal"},
        {"lengths": [3, 3], "inputs": "normal"},
        {"lengths": "1,2", "inputs": "normal"},
        {"lengths": [1, 2], "inputs": "normal", "rows": 2},
    ],
)
def test_a_malformed_case_is_named_as_a_config_error(case):
    mapping = make_config_mapping()
    mapping["correctness"]["cases"] = [*mapping["correctness"]["cases"], case]

    with pytest.raises(MetalFusedError):
        config_from_mapping(mapping)


def test_a_timing_shape_holds_cols_or_max_length_only():
    assert TimingShape(rows=3, cols=7).padded_elements == 21
    assert TimingShape(rows=3, max_length=9).padded_elements == 27
    with pytest.raises(MetalFusedError, match="not both"):
        TimingShape(rows=3, cols=7, max_length=9)
    with pytest.raises(MetalFusedError, match="not both"):
        TimingShape(rows=3)


def test_files_are_read_bounded_and_without_duplicate_keys(tmp_path):
    path = tmp_path / "config.json"
    text = json.dumps(make_config_mapping())
    path.write_text(text.replace('"seed": 79', '"seed": 79, "seed": 80'))

    with pytest.raises(MetalFusedError, match="duplicate"):
        load_config(path)
    path.write_text(text)
    assert isinstance(load_config(path), MetalFusedConfig)
    with pytest.raises(MetalFusedError):
        load_config(tmp_path / "missing.json")


def test_the_fingerprint_binds_every_control():
    base = make_config()
    other = copy.deepcopy(make_config_mapping())
    other["benchmark"]["warmup"] = 2

    assert base.fingerprint() == make_config().fingerprint()
    assert config_from_mapping(other).fingerprint() != base.fingerprint()
