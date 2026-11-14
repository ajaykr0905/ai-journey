import json
from copy import deepcopy
from pathlib import Path

import pytest

from ai_journey.memory_hierarchy_config import (
    MAX_ALLOCATION_BYTES,
    MAX_CONFIG_BYTES,
    PlannedWork,
    config_from_mapping,
    load_config,
)
from ai_journey.memory_hierarchy_stream import MemoryHierarchyError


def make_config_mapping() -> dict:
    return {
        "schema_version": 1,
        "name": "memory-hierarchy-test",
        "dtype": "float64",
        "seed": 71,
        "timing": {
            "warmup": 1,
            "repeats": 3,
            "target_bytes_per_sample": 16384,
            "max_runtime_seconds": 60,
        },
        "stream": {
            "kernels": ["copy", "scale", "add", "triad"],
            "elements": [256, 1024, 4096],
        },
        "access": {"elements": 4096, "stride_elements": 16},
        "matmul": {"m": 32, "n": 24, "k": 16},
        "littles_law": {
            "latency_ns": 100,
            "latency_note": "Stated test latency, not a measurement.",
        },
    }


def changed(path, value):
    mapping = make_config_mapping()
    target = mapping
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return mapping


def test_config_round_trips_to_its_canonical_mapping_and_fingerprint():
    mapping = make_config_mapping()
    config = config_from_mapping(mapping)

    assert config.to_mapping() == mapping
    assert config.element_bytes == 8
    assert config.timing.samples == 4
    assert len(config.fingerprint()) == 64
    assert config.fingerprint() == config_from_mapping(deepcopy(mapping)).fingerprint()
    other = changed(("stream", "elements"), [256, 1024, 8192])
    assert config_from_mapping(other).fingerprint() != config.fingerprint()


def test_planned_work_counts_the_batches_the_run_will_use():
    work = config_from_mapping(make_config_mapping()).planned_work()

    stream_calls = sum(
        4 * -(-16384 // (transfers * count * 8))
        for count in (256, 1024, 4096)
        for transfers in (2, 2, 3, 5)
    )
    access_calls = 4 * (1 + 1 + 1 + 1)
    assert work == PlannedWork(
        peak_allocation_bytes=max(
            3 * 4096 * 8 + 2 * 4096 * 8,
            4096 * (2 * 8 + 8) + 2 * 4096 * 8,
            (32 * 16 + 16 * 24 + 32 * 24) * 8 + 2 * 16 * 24 * 8 + 32 * (32 + 96) * 8,
        ),
        accounted_bytes=work.accounted_bytes,
        calls=stream_calls + access_calls + 4,
        flops=work.flops,
        measurements=3 * 4 + 4 + 1,
    )
    assert work.accounted_bytes >= 4 * 16384 * (12 + 4)
    assert work.flops == 4 * 2 * 32 * 24 * 16 + sum(
        4 * -(-16384 // (transfers * count * 8)) * flops * count
        for count in (256, 1024, 4096)
        for transfers, flops in ((2, 0), (2, 1), (3, 1), (5, 2))
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("unknown",), 1, "config fields"),
        (("seed",), KeyError, "config fields"),
        (("timing", "jitter"), 1, "timing fields"),
        (("timing", "warmup"), KeyError, "timing fields"),
        (("stream", "arrays"), 3, "stream fields"),
        (("access", "elements"), KeyError, "access fields"),
        (("matmul", "batch"), 1, "matmul fields"),
        (("littles_law", "latency_cycles"), 300, "littles_law fields"),
        (("schema_version",), 2, "schema_version must be 1"),
        (("schema_version",), True, "schema_version must be 1"),
        (("name",), "Day 71", "lowercase slug"),
        (("dtype",), "float16", "dtype"),
        (("seed",), True, "seed must be an integer"),
        (("seed",), 2**32, "seed must be in"),
        (("timing", "repeats"), 2, "repeats must be in"),
        (("timing", "repeats"), 7.0, "repeats must be an integer"),
        (("timing", "warmup"), 11, "warmup must be in"),
        (("timing", "target_bytes_per_sample"), 512, "target_bytes_per_sample"),
        (("timing", "max_runtime_seconds"), 0, "max_runtime_seconds"),
        (("timing", "max_runtime_seconds"), 301, "max_runtime_seconds"),
        (("stream", "kernels"), ["copy", "add"], "must include triad"),
        (("stream", "kernels"), ["triad", "copy"], "STREAM order"),
        (("stream", "kernels"), "triad", "JSON array"),
        (("stream", "elements"), [4096, 256], "strictly increasing"),
        (("stream", "elements"), [256, False], "sweep elements must be an integer"),
        (("stream", "elements"), [], "1 to 16 points"),
        (("access", "stride_elements"), 3, "multiple of stride_elements"),
        (("access", "elements"), 512, "access elements must be in"),
        (("matmul", "k"), 8, "k must be in"),
        (("littles_law", "latency_ns"), 0, "latency_ns must be in"),
        (("littles_law", "latency_ns"), 1e999, "latency_ns must be an integer"),
        (("littles_law", "latency_note"), "", "latency_note"),
        (("littles_law", "latency_note"), "two  spaces", "latency_note"),
        (("littles_law", "latency_note"), "x" * 401, "latency_note"),
        (("littles_law", "latency_note"), "café", "latency_note"),
    ],
)
def test_every_contract_violation_is_one_error_type(path, value, message):
    with pytest.raises(MemoryHierarchyError, match=message):
        config_from_mapping(changed(path, value))


def heavy(dtype="float64", elements=(256,), target=16384, matmul=32):
    mapping = make_config_mapping()
    mapping["dtype"] = dtype
    mapping["stream"]["elements"] = list(elements)
    mapping["timing"].update(warmup=10, repeats=50, target_bytes_per_sample=target)
    mapping["matmul"] = {"m": matmul, "n": matmul, "k": matmul}
    return mapping


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        (heavy(elements=(256, 1 << 24)), f"allocation; the bound is {1 << 28}"),
        (heavy(elements=(1 << 16,), target=1 << 30), "accounted bytes"),
        (heavy(target=1 << 28), "kernel calls"),
        (heavy(matmul=2048), "FLOPs"),
        (heavy(elements=(64,), target=1 << 30), "at most 65536"),
    ],
)
def test_whole_run_bounds_reject_configs_before_anything_is_allocated(mapping, message):
    with pytest.raises(MemoryHierarchyError, match=message):
        config_from_mapping(mapping)


def test_bounds_admit_the_largest_float32_sweep_and_reject_non_objects():
    config = config_from_mapping(heavy(dtype="float32", elements=(256, 1 << 24)))

    assert config.planned_work().peak_allocation_bytes <= MAX_ALLOCATION_BYTES
    with pytest.raises(MemoryHierarchyError, match="config fields"):
        config_from_mapping([make_config_mapping()])


def test_files_are_bounded_strict_json_without_duplicate_keys(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(make_config_mapping(), indent=2), encoding="utf-8")
    assert load_config(path).to_mapping() == make_config_mapping()

    for payload, message in (
        (b'{"seed": 1, "seed": 2}', "duplicate JSON field: seed"),
        (b'{"seed": NaN}', "non-finite"),
        (b"[1]", "root must be a JSON object"),
        (b'{"seed": 1,}', "strict UTF-8 JSON"),
        (b'{"seed": ' + b"9" * 5000 + b"}", "integer string conversion"),
        (b"{" + b" " * MAX_CONFIG_BYTES, "exceeds"),
    ):
        path.write_bytes(payload)
        with pytest.raises(MemoryHierarchyError, match=message):
            load_config(path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    for target in (link, tmp_path, tmp_path / "missing.json"):
        with pytest.raises(MemoryHierarchyError, match="regular non-symlink"):
            load_config(target)
