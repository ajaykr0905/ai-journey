import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.matmul_bottleneck_config import (
    MAX_CONFIG_BYTES,
    BottleneckRule,
    MatmulBottleneckError,
    config_from_mapping,
    load_config,
)
from ai_journey.memory_access_plan import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-75-matmul-bottleneck.json"


def pinned_mapping():
    return json.loads(PINNED.read_bytes())


def changed(path, value):
    mapping = copy.deepcopy(pinned_mapping())
    target = mapping
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return mapping


def test_pinned_config_round_trips_and_states_every_threshold():
    config = load_config(PINNED)
    mapping = pinned_mapping()

    assert config.to_mapping() == mapping
    assert config.fingerprint() == sha256(canonical_json_bytes(mapping)).hexdigest()
    assert config.rule == BottleneckRule(
        intensity_model="ideal_reuse",
        dispatch_overhead_share_min=0.5,
        near_roof_share_min=0.7,
        timing_tolerance=0.05,
        gpu_agreement_max_ratio=1.5,
        roof_excess_tolerance=0.05,
        require_fma_scales_with_work=True,
    )
    assert config.mapping.slower_ratio_min == 1.1


def test_a_changed_threshold_changes_the_fingerprint():
    pinned = load_config(PINNED).fingerprint()
    other = config_from_mapping(changed(("rule", "near_roof_share_min"), 0.8))

    assert other.fingerprint() != pinned


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("rule", "near_roof_share_min"), 1, "must be a finite float"),
        (("rule", "near_roof_share_min"), True, "must be a finite float"),
        (("rule", "near_roof_share_min"), 0.0, r"must lie in \(0.0, 1.0\]"),
        (("rule", "near_roof_share_min"), 1.5, r"must lie in \(0.0, 1.0\]"),
        (("rule", "dispatch_overhead_share_min"), 0.0, "must lie in"),
        (("rule", "timing_tolerance"), -0.01, r"must lie in \[0.0, 1.0\]"),
        (("rule", "gpu_agreement_max_ratio"), 0.9, r"must lie in \[1.0, 10.0\]"),
        (("rule", "roof_excess_tolerance"), 2.0, "must lie in"),
        (("rule", "intensity_model"), "sector_model", "intensity_model must be"),
        (("rule", "require_fma_scales_with_work"), 1, "must be a boolean"),
        (("rule", "extra"), 0.5, "rule fields do not match"),
        (("rule", "timing_tolerance"), KeyError, "rule fields do not match"),
        (("mapping", "slower_ratio_min"), 0.99, "must lie in"),
        (("mapping",), [], "mapping fields do not match"),
        (("schema_version",), 2, "schema_version must be 1"),
        (("schema_version",), True, "schema_version must be 1"),
        (("name",), "Day 75", "lowercase slug"),
        (("statement",), "x", "config fields do not match"),
    ],
)
def test_invalid_configs_are_refused_with_a_named_reason(path, value, message):
    with pytest.raises(MatmulBottleneckError, match=message):
        config_from_mapping(changed(path, value))


def test_the_file_reader_refuses_duplicates_oversize_and_symlinks(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"name": "a", "name": "b"}', encoding="utf-8")
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
    link = tmp_path / "link.json"
    link.symlink_to(PINNED)

    for path, message in (
        (duplicate, "duplicate JSON field"),
        (large, "exceeds"),
        (link, "regular non-symlink"),
    ):
        with pytest.raises(MatmulBottleneckError, match=message):
            load_config(path)
