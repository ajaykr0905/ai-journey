import json

import pytest

from ai_journey.gpu_ceiling_config import (
    MAX_CONFIG_BYTES,
    PlannedWork,
    config_from_mapping,
    load_config,
)
from ai_journey.gpu_ceiling_source import GpuCeilingError


def make_config_mapping() -> dict:
    """A small config whose every point runs in milliseconds on a stand-in."""

    return {
        "schema_version": 1,
        "name": "gpu-ceiling-test",
        "seed": 74,
        "correctness": {
            "threads": 1000,
            "fma_steps": 64,
            "stream_float4": 4099,
            "bank_trips": 9,
        },
        "fma": {
            "group_size": 256,
            "threads": 4096,
            "chain_counts": [1, 4],
            "fma_steps": [8, 32, 128],
        },
        "bandwidth": {
            "group_size": 256,
            "buffer_bytes": [16384, 65536],
            "copy_float4_per_thread": 4,
            "reduce_float4_per_thread": 16,
            "target_bytes_per_sample": 262144,
        },
        "bank": {"group_size": 256, "threads": 512, "strides": [1, 2, 32], "trips": 3},
        "timing": {"warmup": 1, "repeats": 3, "max_runtime_seconds": 60},
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
    assert config.fma.chain_counts == (1, 4)
    assert config.bandwidth.float4_counts == (1024, 4096)
    assert config.bandwidth.per_thread("stream_read_reduce") == 16
    other = config_from_mapping(changed(("seed",), 75))
    assert config.fingerprint() == config_from_mapping(mapping).fingerprint()
    assert config.fingerprint() != other.fingerprint()


def test_planned_work_counts_the_run_with_its_own_batch_sizes():
    config = config_from_mapping(make_config_mapping())

    assert config.stream_batches() == (
        ("stream_copy", 1024, 32768, 8),
        ("stream_copy", 4096, 131072, 2),
        ("stream_read_reduce", 1024, 16640, 16),
        ("stream_read_reduce", 4096, 66560, 4),
    )
    assert config.planned_work() == PlannedWork(
        correctness_cases=10,
        fma_points=6,
        stream_points=4,
        bank_points=6,
        timed_samples=48,
        calls=168,
        flops=4 * 2 * 4096 * (1 + 4) * (8 + 32 + 128),
        stream_bytes=4 * (8 * 32768 + 2 * 131072 + 16 * 16640 + 4 * 66560),
        bank_accesses=4 * 6 * 512 * 3 * 16,
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("extra",), 1, "config fields"),
        (("fma", "threads"), KeyError, "fma fields"),
        (("schema_version",), 2, "schema_version must be 1"),
        (("name",), "Not A Slug", "lowercase slug"),
        (("seed",), True, "seed must be an integer"),
        (("seed",), -1, "seed must be in"),
        (("correctness", "fma_steps"), 60, "multiple of 8"),
        (("correctness", "threads"), 1024, "multiple of the fma group size"),
        (("correctness", "stream_float4"), 4096, "past the data"),
        (("fma", "chain_counts"), [1, 3], "must come from 1, 2, 4, 8, 16, 32"),
        (("fma", "chain_counts"), [4, 1], "strictly increasing"),
        (("fma", "chain_counts"), [], "must hold 1 to 6 values"),
        (("fma", "fma_steps"), [8, 32], "must hold 3 to 8 values"),
        (("fma", "fma_steps"), [8, 32, 100], "multiples of 8"),
        (("fma", "fma_steps"), [8.0, 32, 128], "must be an integer"),
        # Rounding has stopped every chain by 32768 steps, so the timed outputs
        # of these points could not show that their trips ran.
        (("fma", "fma_steps"), [8192, 16384, 32768], "fma_steps must be in"),
        (("correctness", "fma_steps"), 32768, "fma_steps must be in"),
        (("fma", "group_size"), 2048, "fma group_size must be in"),
        (("bandwidth", "buffer_bytes"), [16384, 65544], "multiples of 16"),
        (("bandwidth", "buffer_bytes"), [8, 65536], "buffer_bytes must be in"),
        (("bandwidth", "reduce_float4_per_thread"), 2048, "must be in"),
        (("bandwidth", "target_bytes_per_sample"), 512, "target_bytes_per_sample"),
        (("bank", "group_size"), 240, "multiple of 32"),
        (("bank", "threads"), 1000, "whole threadgroups"),
        (("bank", "strides"), [2, 4], "start at 1"),
        (("bank", "strides"), [1, 64], "bank strides must be in"),
        (("bank", "strides"), [1], "must hold 2 to 32 values"),
        (("timing", "repeats"), 2, "repeats must be in"),
        (("timing", "max_runtime_seconds"), 601, "max_runtime_seconds must be in"),
        (("bank", "strides"), "1,2", "must be a JSON array"),
    ],
)
def test_invalid_controls_are_refused_with_one_error_type(path, value, message):
    with pytest.raises(GpuCeilingError, match=message):
        config_from_mapping(changed(path, value))


def test_work_bounds_and_oversized_samples_are_refused_before_a_run():
    huge = changed(("fma", "threads"), 1 << 24)
    huge["fma"]["chain_counts"] = [32]
    huge["fma"]["fma_steps"] = [4096, 8192, 16384]
    with pytest.raises(GpuCeilingError, match="FLOPs; the bound is"):
        config_from_mapping(huge)
    tiny_batches = changed(("bandwidth", "buffer_bytes"), [16, 32])
    tiny_batches["bandwidth"]["target_bytes_per_sample"] = 1 << 30
    with pytest.raises(GpuCeilingError, match="at most 4096"):
        config_from_mapping(tiny_batches)


def test_files_are_read_bounded_strict_and_duplicate_free(tmp_path):
    good = tmp_path / "config.json"
    good.write_text(json.dumps(make_config_mapping()), encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"name": "a", "name": "b"}', encoding="utf-8")
    large = tmp_path / "large.json"
    large.write_text(" " * (MAX_CONFIG_BYTES + 1), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(good)

    assert load_config(good).name == "gpu-ceiling-test"
    for path, message in (
        (duplicate, "duplicate JSON field"),
        (large, "exceeds"),
        (link, "regular non-symlink"),
        (tmp_path / "missing.json", "regular non-symlink"),
    ):
        with pytest.raises(GpuCeilingError, match=message):
            load_config(path)
