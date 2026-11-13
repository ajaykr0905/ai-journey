from copy import deepcopy

import pytest

from ai_journey.memory_access_pattern import MemoryAccessError
from ai_journey.memory_access_plan import (
    PATTERN_KINDS,
    canonical_json_bytes,
    plan_from_mapping,
)
from ai_journey.memory_access_tiling import MatmulShape


def make_plan_mapping() -> dict:
    return {
        "schema_version": 1,
        "name": "memory-access-test",
        "geometry": {
            "warp_size": 32,
            "sector_bytes": 32,
            "line_bytes": 128,
            "shared_memory_banks": 32,
            "bank_width_bytes": 4,
        },
        "patterns": [
            {"name": "contiguous", "kind": "contiguous", "access_bytes": 4},
            {
                "name": "offset",
                "kind": "contiguous",
                "access_bytes": 4,
                "base_bytes": 4,
            },
            {
                "name": "stride-2",
                "kind": "strided",
                "access_bytes": 4,
                "stride_elements": 2,
            },
            {
                "name": "broadcast",
                "kind": "broadcast",
                "access_bytes": 4,
                "address_bytes": 64,
            },
            {
                "name": "padded-row",
                "kind": "row",
                "access_bytes": 4,
                "rows": 8,
                "columns": 33,
                "leading_dimension": 33,
                "row": 1,
            },
            {
                "name": "column",
                "kind": "column",
                "access_bytes": 4,
                "rows": 64,
                "columns": 64,
                "leading_dimension": 64,
                "column": 3,
            },
            {
                "name": "permutation",
                "kind": "permutation",
                "access_bytes": 4,
                "seed": 74,
            },
            {
                "name": "half-warp",
                "kind": "contiguous",
                "access_bytes": 4,
                "active_mask": 65535,
            },
            {"name": "float64", "kind": "contiguous", "access_bytes": 8},
        ],
        "transposes": [
            {
                "name": "unpadded",
                "leading_dimension": 32,
                "element_bytes": 4,
                "matrix_size": 256,
            },
            {
                "name": "padded",
                "leading_dimension": 33,
                "element_bytes": 4,
                "matrix_size": 256,
            },
        ],
        "matmul": {
            "dtype": "float32",
            "shapes": [{"m": 64, "n": 64, "k": 64}, {"m": 10, "n": 6, "k": 7}],
            "tiles": [4, 8],
        },
        "roofline": {
            "name": "toy-roofline",
            "declared": True,
            "peak_flops_per_second": 40,
            "memory_bandwidth_bytes_per_second": 10,
            "source_url": "https://example.org/toy-datasheet",
            "source_note": "Test ceilings, not a device specification.",
        },
        "benchmark": {
            "backends": ["numpy", "torch"],
            "dtype": "float32",
            "shapes": [{"m": 8, "n": 8, "k": 8}, {"m": 16, "n": 12, "k": 8}],
            "warmup": 1,
            "repeats": 3,
            "seed": 74,
        },
    }


def test_plan_round_trips_to_its_canonical_mapping_and_fingerprint():
    mapping = make_plan_mapping()
    plan = plan_from_mapping(mapping)

    assert plan.to_mapping() == mapping
    assert plan_from_mapping(plan.to_mapping()) == plan
    assert len(plan.fingerprint()) == 64
    changed = deepcopy(mapping)
    changed["benchmark"]["seed"] = 75
    assert plan_from_mapping(changed).fingerprint() != plan.fingerprint()


def test_pattern_specs_build_the_declared_warp_requests():
    plan = plan_from_mapping(make_plan_mapping())
    patterns = {pattern.name: pattern.build() for pattern in plan.patterns}

    assert patterns["offset"].lane_addresses[0] == 4
    assert patterns["stride-2"].lane_addresses[1] == 8
    assert set(patterns["broadcast"].active_addresses) == {64}
    assert patterns["padded-row"].lane_addresses[0] == 132
    assert patterns["column"].lane_addresses[1] == 268
    assert patterns["half-warp"].active_lanes == tuple(range(16))
    assert patterns["float64"].access_bytes == 8
    assert set(PATTERN_KINDS) == {
        "contiguous",
        "strided",
        "broadcast",
        "row",
        "column",
        "permutation",
    }


def test_benchmark_section_expands_backends_by_shapes():
    plan = plan_from_mapping(make_plan_mapping())

    assert [(case.backend, case.shape) for case in plan.benchmark.cases()] == [
        ("numpy", MatmulShape(8, 8, 8)),
        ("numpy", MatmulShape(16, 12, 8)),
        ("torch", MatmulShape(8, 8, 8)),
        ("torch", MatmulShape(16, 12, 8)),
    ]
    assert plan.roofline.hardware.ridge_point == 4


def edit(path, value):
    def apply(mapping):
        target = mapping
        for key in path[:-1]:
            target = target[key]
        if value is KeyError:
            del target[path[-1]]
        else:
            target[path[-1]] = value
        return mapping

    return apply


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (edit(("extra",), 1), "plan fields"),
        (edit(("benchmark",), KeyError), "plan fields"),
        (edit(("schema_version",), 2), "schema_version"),
        (edit(("schema_version",), True), "schema_version"),
        (edit(("name",), "Memory Access"), "plan name"),
        (edit(("geometry", "warp_size"), 64), "documented CUDA contract"),
        (edit(("geometry", "l2_bytes"), 1), "geometry fields"),
        (edit(("patterns",), []), "patterns must hold"),
        (edit(("patterns", 0, "kind"), "diagonal"), "pattern kind"),
        (edit(("patterns", 0, "stride_elements"), 2), "contiguous pattern fields"),
        (edit(("patterns", 2, "stride_elements"), KeyError), "strided pattern fields"),
        (edit(("patterns", 0, "access_bytes"), "4"), "access_bytes must be"),
        (edit(("patterns", 0, "access_bytes"), True), "access_bytes must be"),
        (edit(("patterns", 1, "base_bytes"), 2), "aligned"),
        (edit(("patterns", 1, "name"), "contiguous"), "names must be unique"),
        (edit(("patterns", 7, "active_mask"), 0), "active_mask"),
        (edit(("transposes", 0, "matrix_size"), 100), "multiple of 128"),
        (edit(("transposes", 0, "padding"), 1), "transpose fields"),
        (edit(("matmul", "tiles"), [8, 4]), "unique and increasing"),
        (edit(("matmul", "tiles"), [64]), "tile must be in"),
        (edit(("matmul", "tiles"), [4.0]), "tile must be an integer"),
        (edit(("matmul", "shapes"), []), "matmul shapes"),
        (edit(("matmul", "shapes", 1), {"m": 64, "n": 64, "k": 64}), "repeat"),
        (edit(("matmul", "shapes", 0, "batch"), 2), "shape fields"),
        (edit(("matmul", "dtype"), "int8"), "dtype"),
        (edit(("roofline", "declared"), False), "declared inputs"),
        (edit(("roofline", "source_url"), "http://example.org"), "https"),
        (edit(("roofline", "source_note"), " padded "), "canonical"),
        (edit(("roofline", "peak_flops_per_second"), True), "positive"),
        (edit(("roofline", "peak_flops_per_second"), float("inf")), "positive"),
        (edit(("roofline", "measured"), False), "roofline fields"),
        (edit(("benchmark", "backends"), ["numpy", "numpy"]), "backends"),
        (edit(("benchmark", "backends"), ["cupy"]), "backends"),
        (edit(("benchmark", "dtype"), "float64"), "float32"),
        (edit(("benchmark", "repeats"), 0), "repeats"),
        (edit(("benchmark", "warmup"), False), "warmup"),
        (edit(("benchmark", "seed"), -1), "seed"),
        (
            lambda mapping: edit(("benchmark", "repeats"), 20)(
                edit(("benchmark", "shapes"), [{"m": 1024, "n": 1024, "k": 1024}])(
                    mapping
                )
            ),
            "FLOP budget",
        ),
        (
            edit(("benchmark", "shapes"), [{"m": 2048, "n": 8, "k": 8}]),
            "exceed",
        ),
    ],
)
def test_plan_rejects_unknown_missing_and_out_of_contract_fields(change, message):
    with pytest.raises(MemoryAccessError, match=message):
        plan_from_mapping(change(make_plan_mapping()))


def test_oversized_sections_are_rejected_before_any_entry_is_analyzed():
    mapping = make_plan_mapping()
    invalid = {
        "name": "invalid",
        "leading_dimension": 32,
        "element_bytes": 4,
        "matrix_size": 0,
    }
    mapping["transposes"] = [invalid] * 17
    with pytest.raises(MemoryAccessError, match="transposes must hold 1 to 16"):
        plan_from_mapping(mapping)
    mapping["transposes"] = [invalid]
    with pytest.raises(MemoryAccessError, match="matrix_size"):
        plan_from_mapping(mapping)
    mapping = make_plan_mapping()
    mapping["patterns"] = [{"name": "bad", "kind": "diagonal"}] * 65
    with pytest.raises(MemoryAccessError, match="patterns must hold 1 to 64"):
        plan_from_mapping(mapping)


def test_plan_rejects_non_object_payloads():
    for payload in ([], "plan", None):
        with pytest.raises(MemoryAccessError):
            plan_from_mapping(payload)


def test_canonical_json_is_sorted_compact_ascii_and_finite():
    assert canonical_json_bytes({"b": 1, "a": "é"}) == b'{"a":"\\u00e9","b":1}'
    with pytest.raises(MemoryAccessError, match="finite JSON"):
        canonical_json_bytes({"value": float("nan")})
    with pytest.raises(MemoryAccessError, match="finite JSON"):
        canonical_json_bytes({"value": object()})
