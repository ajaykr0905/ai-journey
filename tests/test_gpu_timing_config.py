import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.gpu_timing_clocks import GpuTimingError
from ai_journey.gpu_timing_config import (
    MAX_TINY_FLOPS,
    GpuTimingConfig,
    config_from_mapping,
    load_config,
)
from ai_journey.matmul_metal_config import MAX_RUN_FLOPS
from ai_journey.memory_access_plan import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-75-gpu-timing.json"
SMALL = {
    "schema_version": 1,
    "name": "gpu-timing-small",
    "seed": 75,
    "launch": {"group_size": [32, 8]},
    "sweep": {
        "shapes": [[1, 1, 1], [16, 16, 16], [64, 32, 128]],
        "warmup": 1,
        "repeats": 3,
        "max_runtime_seconds": 60,
    },
    "reference": {
        "fma_ceiling_gflops_per_second": 4618.0,
        "source": "docs/gpu-ceilings-lab.md run 1",
    },
}


def small(**changes):
    payload = copy.deepcopy(SMALL)
    for path, value in changes.items():
        target = payload
        *parents, leaf = path.split("__")
        for parent in parents:
            target = target[parent]
        target[leaf] = value
    return payload


def test_a_valid_config_round_trips_and_counts_its_work():
    config = config_from_mapping(small())

    assert isinstance(config, GpuTimingConfig)
    assert config.to_mapping() == SMALL
    work = config.planned_work().to_mapping()
    # One empty-kernel side plus three sides per shape, three clocks each.
    assert work == {
        "sweep_points": 3,
        "sides": 10,
        "timed_samples": 10 * 3 * 3,
        "launches": 10 * 3 * 4,
        "flops": 3 * 3 * 4 * (2 + 2 * 16**3 + 2 * 64 * 32 * 128),
    }
    assert len(config.fingerprint()) == 64


def test_the_pinned_config_loads_and_spans_tiny_to_large():
    config = load_config(PINNED)

    shapes = config.sweep.shapes
    assert shapes[0].flops == 2
    assert shapes[-1].flops == 2 * 4096**3
    assert config.reference.fma_ceiling_gflops_per_second == 4618.0
    assert "gpu-ceilings-lab.md" in config.reference.source
    assert config.planned_work().flops <= MAX_RUN_FLOPS
    assert config.to_mapping() == json.loads(PINNED.read_bytes())


def test_the_fingerprint_is_the_digest_of_canonical_json():
    config = config_from_mapping(small())

    expected = sha256(canonical_json_bytes(SMALL)).hexdigest()
    assert config.fingerprint() == expected
    other = config_from_mapping(small(seed=76))
    assert other.fingerprint() != expected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"extra": 1}, "config fields"),
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": True}, "schema_version"),
        ({"name": "Bad Name"}, "slug"),
        ({"seed": True}, "integer"),
        ({"seed": -1}, "seed"),
        ({"launch__group_size": [32, 8, 1]}, "group_size"),
        ({"launch__group_size": [64, 32]}, "threads"),
        ({"sweep__shapes": [[1, 1, 1], [2, 2, 2]]}, "3 to 16"),
        ({"sweep__shapes": [[1, 1, 1], [2, 2, 2], [2, 2, 2]]}, "rise strictly"),
        ({"sweep__shapes": [[1, 1, 1], [4, 4, 4], [2, 2, 2]]}, "rise strictly"),
        ({"sweep__shapes": [[1, 1, 1], [2, 2], [4, 4, 4]]}, "triples"),
        ({"sweep__shapes": [[16, 16, 16], [32, 32, 32], [64, 64, 64]]}, "tiny"),
        ({"sweep__shapes": [[1, 1, 1], [2, 2, 2], [9000, 1, 1]]}, "m must be"),
        ({"sweep__shapes": "[[1, 1, 1]]"}, "JSON array"),
        ({"sweep__warmup": 11}, "warmup"),
        ({"sweep__repeats": 2}, "repeats"),
        ({"sweep__repeats": 3.0}, "integer"),
        ({"sweep__max_runtime_seconds": 0}, "max_runtime_seconds"),
        ({"reference__fma_ceiling_gflops_per_second": 4618}, "float"),
        ({"reference__fma_ceiling_gflops_per_second": 0.0}, "float"),
        ({"reference__source": ""}, "printable ASCII"),
        ({"reference__source": "line\nbreak"}, "printable ASCII"),
        ({"reference__source": "x" * 201}, "printable ASCII"),
    ],
)
def test_invalid_configs_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        config_from_mapping(small(**changes))


def test_a_missing_section_field_is_refused():
    payload = small()
    del payload["sweep"]["warmup"]

    with pytest.raises(GpuTimingError, match="sweep fields"):
        config_from_mapping(payload)


def test_a_run_past_the_flop_bound_is_refused():
    shapes = [[1, 1, 1], [4096, 4096, 4096], [8192, 8192, 4096]]

    with pytest.raises(GpuTimingError, match="FLOPs; the bound"):
        config_from_mapping(small(sweep__shapes=shapes, sweep__repeats=20))


def test_the_tiny_bound_admits_an_8_cubed_matmul_first():
    shapes = [[8, 8, 8], [16, 16, 16], [32, 32, 32]]

    config = config_from_mapping(small(sweep__shapes=shapes))

    assert config.sweep.shapes[0].flops == MAX_TINY_FLOPS


def test_duplicate_keys_in_a_file_are_refused(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"name": "a", "name": "b"}', encoding="utf-8")

    with pytest.raises(GpuTimingError, match="duplicate"):
        load_config(path)
