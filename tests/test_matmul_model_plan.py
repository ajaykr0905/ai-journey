import copy
import json
from pathlib import Path

import pytest

from ai_journey.kernel_launch_config import KernelLaunchError, LaunchLimitError
from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.kernel_launch_simulator import LaunchOrder
from ai_journey.matmul_model_kernels import X_COLUMNS, X_ROWS
from ai_journey.matmul_model_plan import (
    MAX_PLAN_BYTES,
    MatmulCase,
    MatmulModelPlan,
    load_plan,
)
from ai_journey.matmul_model_reference import MatmulModelError
from ai_journey.matmul_model_roofline import Ceilings
from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.simt_occupancy import LaunchNotResidentError

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-74-matmul-model.json"


def plan_mapping() -> dict:
    """Return a small valid plan whose run takes a fraction of a second."""

    return {
        "schema_version": 1,
        "name": "test-matmul-model",
        "statement": "Small test inputs; no GPU runs.",
        "compute_capability": "8.0",
        "block": [32, 2],
        "cases": [
            {"name": "ragged", "m": 3, "n": 35, "k": 3, "input_seed": 7},
            {"name": "aligned", "m": 32, "n": 32, "k": 2, "input_seed": 7},
        ],
        "permuted_order_seeds": [5],
        "accounting_shapes": [{"m": 64, "n": 64, "k": 64}],
        "registers_per_thread": [32, 64],
        "ceilings": [
            {"label": "declared-ridge-4", "compute_gflops": "4", "bandwidth_gbps": "1"}
        ],
    }


def write_plan(path: Path, mapping: dict | None = None) -> Path:
    path.write_bytes(
        canonical_json_bytes(plan_mapping() if mapping is None else mapping)
    )
    return path


def test_pinned_plan_declares_ragged_and_aligned_cases_and_declared_inputs():
    plan = load_plan(PINNED)

    assert plan.name == "day-74-matmul-model"
    assert (plan.compute_capability, plan.block, plan.threads_per_block) == (
        "8.0",
        (32, 8),
        256,
    )
    assert plan.cases == (
        MatmulCase("ragged", MatmulShape(37, 45, 29), 74),
        MatmulCase("aligned", MatmulShape(32, 64, 16), 74),
    )
    assert plan.orders == (
        LaunchOrder.canonical(),
        LaunchOrder.permuted(1),
        LaunchOrder.permuted(2),
        LaunchOrder.permuted(3),
    )
    ragged, aligned = plan.cases
    assert plan.launch(ragged, X_COLUMNS).config.grid.to_list() == [2, 5, 1]
    assert plan.launch(ragged, X_ROWS).config.grid.to_list() == [2, 6, 1]
    for mapping in (X_COLUMNS, X_ROWS):
        config = plan.launch(aligned, mapping).config
        assert config.launched_threads == 32 * 64
    assert plan.accounting_shapes == (
        MatmulShape(1024, 1024, 1024),
        MatmulShape(4096, 4096, 4096),
    )
    assert plan.registers_per_thread == (32, 40, 64, 128)
    assert plan.ceilings == (
        Ceilings("declared-ridge-4", "4", "1"),
        Ceilings("declared-ridge-16", "16", "1"),
    )
    assert "not measurements" in plan.statement
    assert "No GPU runs, no CUDA source is compiled" in plan.statement
    assert plan.to_mapping() == json.loads(PINNED.read_text(encoding="ascii"))


def test_plans_round_trip_through_their_mapping(tmp_path):
    plan = load_plan(write_plan(tmp_path / "plan.json"))

    assert plan.to_mapping() == plan_mapping()
    assert MatmulModelPlan.from_mapping(plan.to_mapping()) == plan


def change(*path):
    *keys, value = path

    def apply(mapping):
        target = mapping
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return apply


@pytest.mark.parametrize(
    ("edit", "error", "message"),
    [
        (change("extra", 1), KernelLaunchError, "plan fields mismatch"),
        (change("cases", 0, "p", 1), KernelLaunchError, "case fields mismatch"),
        (change("schema_version", 2), MatmulModelError, "schema_version"),
        (change("schema_version", True), MatmulModelError, "schema_version"),
        (change("name", "Bad Name"), MatmulModelError, "lowercase slug"),
        (change("statement", " padded"), MatmulModelError, "printable ASCII"),
        (change("compute_capability", "8.6"), MatmulModelError, "compute_capability"),
        (change("block", [16, 16]), MatmulModelError, "multiple of 32"),
        (change("block", [64, 32]), LaunchLimitError, "max_threads_per_block"),
        (change("block", [32]), MatmulModelError, "block_x, block_y"),
        (change("block", {"x": 32}), KernelLaunchError, "block must be a JSON array"),
        (change("cases", []), MatmulModelError, "cases must hold 1 to 4"),
        (change("cases", 1, "name", "ragged"), MatmulModelError, "case names must be unique"),
        (change("cases", 0, "k", 0), KernelLaunchError, "between 1 and 65536"),
        (change("cases", 0, "input_seed", -1), KernelLaunchError, "input_seed"),
        (change("cases", 0, "k", 20000), MatmulModelError, "2100000 multiply-adds"),
        (change("cases", 0, "m", 1000), MatmulModelError, "launches 64000 threads with x_columns"),
        (change("cases", 1, "n", 2000), MatmulModelError, "launches 64512 threads with x_columns"),
        (change("permuted_order_seeds", []), MatmulModelError, "1 to 4"),
        (change("permuted_order_seeds", [5, 5]), MatmulModelError, "must not repeat"),
        (change("accounting_shapes", [{"m": 1, "n": 1, "k": i} for i in range(1, 6)]), MatmulModelError, "0 to 4"),
        (change("accounting_shapes", [{"m": 2, "n": 2, "k": 2}] * 2), MatmulModelError, "must not repeat"),
        (change("accounting_shapes", 0, "m", 0), KernelLaunchError, "accounting shape.m"),
        (change("registers_per_thread", [64, 32]), MatmulModelError, "must increase"),
        (change("registers_per_thread", [256]), KernelLaunchError, "between 1 and 255"),
        (change("registers_per_thread", []), MatmulModelError, "1 to 6"),
        (change("ceilings", []), MatmulModelError, "1 to 4"),
        (change("ceilings", 0, "compute_gflops", "4.0"), MatmulModelError, "decimal string"),
        (change("ceilings", [plan_mapping()["ceilings"][0]] * 2), MatmulModelError, "must not repeat"),
    ],
)  # fmt: skip
def test_invalid_plans_name_the_rule_they_break(edit, error, message):
    mapping = copy.deepcopy(plan_mapping())
    edit(mapping)

    with pytest.raises(error, match=message):
        MatmulModelPlan.from_mapping(mapping)


def test_ceiling_labels_and_resident_occupancy_are_required():
    mapping = plan_mapping()
    first = mapping["ceilings"][0]
    mapping["ceilings"] = [first, {**first, "compute_gflops": "8"}]
    with pytest.raises(MatmulModelError, match="ceiling labels must be unique"):
        MatmulModelPlan.from_mapping(mapping)

    # A 1,024-thread block with 128 registers per thread needs twice the file.
    wide = {**plan_mapping(), "block": [32, 32], "registers_per_thread": [128]}
    with pytest.raises(LaunchNotResidentError, match="registers_per_block"):
        MatmulModelPlan.from_mapping(wide)


def test_booleans_and_floats_never_pass_as_integers(tmp_path):
    for edit in (
        change("block", [32.0, 2]),
        change("cases", 0, "m", True),
        change("permuted_order_seeds", [True]),
        change("registers_per_thread", [32.0]),
    ):
        mapping = copy.deepcopy(plan_mapping())
        edit(mapping)
        with pytest.raises(MatmulModelError, match="invalid plan"):
            load_plan(write_plan(tmp_path / "plan.json", mapping))


def test_plan_files_must_be_small_strict_regular_json(tmp_path):
    path = tmp_path / "plan.json"
    path.write_bytes(b'{"name":"a","name":"b"}')
    with pytest.raises(KernelLaunchError, match="duplicate JSON key: name"):
        load_plan(path)
    path.write_bytes(b" " * (MAX_PLAN_BYTES + 1))
    with pytest.raises(KernelLaunchError, match="exceeds"):
        load_plan(path)
    link = tmp_path / "link.json"
    link.symlink_to(write_plan(tmp_path / "real.json"))
    with pytest.raises(KernelLaunchError, match="non-symlink"):
        load_plan(link)
    path.write_bytes(b'{"schema_version": NaN}')
    with pytest.raises(KernelLaunchError, match="non-finite"):
        load_plan(path)
