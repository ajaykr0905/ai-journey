from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from ai_journey.simt_device import PINNED_ARCHITECTURES, PINNED_DEVICES, SimtError
from ai_journey.simt_json import canonical_json_bytes, loads_strict, read_bounded
from ai_journey.simt_plan import (
    MAX_SIMT_PLAN_BYTES,
    AccessPatternSpec,
    SimtPlan,
    load_simt_plan,
)


def plan_mapping() -> dict:
    """Return a small valid plan over two pinned devices for fast tests."""

    return {
        "schema_version": 1,
        "name": "test-simt-plan",
        "statement": "Declared test inputs; no GPU is executed.",
        "architectures": [
            PINNED_ARCHITECTURES["7.5"].to_mapping(),
            PINNED_ARCHITECTURES["8.0"].to_mapping(),
        ],
        "devices": [
            PINNED_DEVICES["tesla-t4"].to_mapping(),
            PINNED_DEVICES["a100-sxm4-80gb"].to_mapping(),
        ],
        "vector_add": {
            "elements": 5_000,
            "dtype": "float32",
            "threads_per_block": 256,
            "registers_per_thread": 16,
            "devices": ["tesla-t4", "a100-sxm4-80gb"],
            "basis": "Declared register budget; nothing is compiled.",
        },
        "rgb_to_grayscale": {
            "width": 40,
            "height": 21,
            "block_x": 16,
            "block_y": 16,
            "input_bytes_per_pixel": 3,
            "output_bytes_per_pixel": 1,
            "flops_per_pixel": 5,
            "registers_per_thread": 16,
            "devices": ["a100-sxm4-80gb"],
            "basis": "Three multiplies and two adds per pixel.",
        },
        "occupancy_sweep": {
            "devices": ["tesla-t4", "a100-sxm4-80gb"],
            "threads_per_block": [128, 1024],
            "registers_per_thread": [32, 128],
            "static_shared_bytes": 0,
            "dynamic_shared_bytes": 0,
            "basis": "Declared sweep points.",
        },
        "memory_latency": {
            "devices": ["a100-sxm4-80gb"],
            "latency_cycles": [400, 800],
            "access_patterns": [
                {
                    "name": "fp32-scalar",
                    "bytes_per_lane": 4,
                    "active_lanes": 32,
                    "independent_loads": 1,
                }
            ],
            "basis": "Declared sensitivity points, not measurements.",
        },
        "arithmetic_latency": {
            "architectures": ["7.5"],
            "pipelines": ["issue", "fp32"],
            "latency_cycles": 4,
            "independent_instructions": [1, 2],
            "basis": "Guide figure for compute capability 7.x.",
        },
        "wave_quantization": {
            "total_blocks": 1_024,
            "threads_per_block": 256,
            "registers_per_thread": 64,
            "devices": ["tesla-t4", "a100-sxm4-80gb"],
            "basis": "Fixed grid.",
        },
        "sources": ["CUDA C++ Programming Guide v12.6 section 5.2.3"],
    }


def write_plan(path: Path, mapping: dict | None = None) -> Path:
    path.write_bytes(
        canonical_json_bytes(plan_mapping() if mapping is None else mapping)
    )
    return path


def test_plan_round_trips_through_its_canonical_mapping(tmp_path: Path):
    plan = SimtPlan.from_mapping(plan_mapping())

    assert plan.to_mapping() == plan_mapping()
    assert plan.device("tesla-t4") == PINNED_DEVICES["tesla-t4"]
    assert plan.architecture("8.0") == PINNED_ARCHITECTURES["8.0"]
    assert plan.memory_latency.access_patterns == (
        AccessPatternSpec("fp32-scalar", 4, 32, 1),
    )
    pretty = tmp_path / "plan.json"
    pretty.write_text(json.dumps(plan_mapping(), indent=2), encoding="utf-8")
    assert load_simt_plan(pretty) == plan


def test_plan_rejects_declared_values_that_drift_from_the_pinned_catalog():
    # 1,591 MHz still reproduces 8.1 TFLOPS within rounding (8.146), but it is not
    # the source-backed value, so the plan refuses it.
    mapping = plan_mapping()
    mapping["devices"][0]["boost_clock_mhz"] = 1_591
    with pytest.raises(SimtError, match="tesla-t4 must equal the pinned"):
        SimtPlan.from_mapping(mapping)

    mapping = plan_mapping()
    mapping["architectures"][1]["max_blocks_per_sm"] = 16
    with pytest.raises(SimtError, match="8.0 must equal the pinned"):
        SimtPlan.from_mapping(mapping)

    plan = SimtPlan.from_mapping(plan_mapping())
    with pytest.raises(SimtError, match="tesla-t4 needs a declared architecture"):
        replace(plan, architectures=(PINNED_ARCHITECTURES["8.0"],))
    with pytest.raises(TypeError, match="vector_add must be VectorAddScenario"):
        replace(plan, vector_add=plan.wave_quantization)
    with pytest.raises(SimtError, match="tuple of SimtDevice"):
        replace(plan, devices=("tesla-t4",))
    with pytest.raises(SimtError, match="'tesla-v100-sxm2' is not declared"):
        plan.device("tesla-v100-sxm2")
    with pytest.raises(SimtError, match="'7.0' is not declared"):
        plan.architecture("7.0")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.update(extra=1), "extra=\\['extra'\\]"),
        (lambda m: m.pop("sources"), "missing=\\['sources'\\]"),
        (lambda m: m.update(schema_version=True), "schema_version"),
        (lambda m: m["vector_add"].update(grid=4), "vector_add fields mismatch"),
        (
            lambda m: m["memory_latency"]["access_patterns"][0].update(stride=2),
            "access pattern fields mismatch",
        ),
        (
            lambda m: m["vector_add"].update(devices=["tesla-v100-sxm2"]),
            "undeclared devices: \\['tesla-v100-sxm2'\\]",
        ),
        (
            lambda m: m["arithmetic_latency"].update(architectures=["7.0"]),
            "undeclared architectures",
        ),
        (lambda m: m["architectures"].pop(0), "not a declared architecture"),
        (
            lambda m: m["arithmetic_latency"].update(pipelines=["tensor"]),
            "pipelines must be drawn from",
        ),
        (
            lambda m: m["occupancy_sweep"].update(threads_per_block=[256, 128]),
            "strictly increasing",
        ),
        (
            lambda m: m["memory_latency"]["access_patterns"].append(
                dict(m["memory_latency"]["access_patterns"][0])
            ),
            "names must not repeat",
        ),
        (lambda m: m["vector_add"].update(dtype="float8"), "supported tensor dtype"),
        (lambda m: m["vector_add"].update(devices=[]), "one to 16 entries"),
        (lambda m: m.update(name="Day 71"), "lowercase slug"),
        (lambda m: m["wave_quantization"].update(basis=""), "basis"),
        (lambda m: m.update(devices={}), "must be a JSON array"),
    ],
)
def test_plan_rejects_unknown_missing_and_inconsistent_fields(mutate, message):
    mapping = plan_mapping()
    mutate(mapping)

    with pytest.raises(SimtError, match=message):
        SimtPlan.from_mapping(mapping)


def test_plan_files_are_bounded_strict_regular_json(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_bytes(b'{"schema_version":1,"schema_version":1}')
    with pytest.raises(SimtError, match="duplicate JSON field: schema_version"):
        load_simt_plan(path)

    path.write_bytes(b'{"schema_version":NaN}')
    with pytest.raises(SimtError, match="non-finite JSON number"):
        load_simt_plan(path)

    path.write_bytes(b"\xff\xfe{}")
    with pytest.raises(SimtError, match="not strict UTF-8 JSON"):
        load_simt_plan(path)

    path.write_bytes(b" " * (MAX_SIMT_PLAN_BYTES + 1))
    with pytest.raises(SimtError, match="exceeds 65536 bytes"):
        load_simt_plan(path)

    target = write_plan(tmp_path / "target.json")
    alias = tmp_path / "alias.json"
    os.symlink(target, alias)
    with pytest.raises(SimtError, match="regular non-symlink file"):
        load_simt_plan(alias)
    with pytest.raises(SimtError, match="regular non-symlink file"):
        load_simt_plan(tmp_path)
    with pytest.raises(SimtError, match="unable to read SIMT plan"):
        load_simt_plan(tmp_path / "missing.json")
    with pytest.raises(TypeError, match="pathlib.Path"):
        load_simt_plan(str(target))

    mapping = plan_mapping()
    mapping["memory_latency"]["access_patterns"] = 4
    with pytest.raises(SimtError, match="invalid SIMT plan"):
        load_simt_plan(write_plan(tmp_path / "bad.json", mapping))


def test_canonical_json_is_sorted_compact_ascii_and_finite():
    assert canonical_json_bytes({"b": 1, "a": "é"}) == b'{"a":"\\u00e9","b":1}'
    with pytest.raises(SimtError, match="canonical JSON"):
        canonical_json_bytes({"x": float("inf")})
    assert loads_strict(b'{"a":[1,2.5]}', "value") == {"a": [1, 2.5]}
    with pytest.raises(TypeError, match="bytes"):
        loads_strict("{}", "value")
    with pytest.raises(TypeError, match="pathlib.Path"):
        read_bounded("plan.json", limit=1, label="plan")
