import copy
import json
from pathlib import Path

import pytest

from ai_journey.kernel_launch_config import (
    Dim3,
    KernelLaunchError,
    LaunchConfig,
    LaunchLimitError,
)
from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.kernel_launch_plan import (
    MAX_PLAN_BYTES,
    KernelLaunchPlan,
    UnguardedDemo,
    load_plan,
)
from ai_journey.kernel_launch_simulator import LaunchOrder

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-73-kernel-launch.json"


def plan_mapping() -> dict:
    """Return a small valid plan whose runs take milliseconds."""

    return {
        "schema_version": 1,
        "name": "test-kernel-launch",
        "statement": "Small test inputs; no GPU runs.",
        "compute_capability": "8.0",
        "vector_add": {"n": 100, "threads_per_block": 32, "input_seed": 7},
        "rgb_to_grayscale": {
            "width": 10,
            "height": 6,
            "block": [4, 4],
            "input_seed": 7,
        },
        "permuted_order_seeds": [5],
        "unguarded_demos": [
            {"kernel": "vector_add", "extent": [100]},
            {"kernel": "rgb_to_grayscale", "extent": [10, 6]},
            {"kernel": "rgb_to_grayscale", "extent": [12, 8]},
        ],
    }


def write_plan(path: Path, mapping: dict | None = None) -> Path:
    path.write_bytes(
        canonical_json_bytes(plan_mapping() if mapping is None else mapping)
    )
    return path


def test_pinned_plan_declares_ragged_and_aligned_exercise_launches():
    plan = load_plan(PINNED)

    assert plan.name == "day-73-kernel-launch"
    assert plan.compute_capability == "8.0"
    assert plan.vector_add.launch_for(1000) == LaunchConfig(Dim3(4), Dim3(256))
    assert plan.rgb_to_grayscale.launch_for(40, 30) == LaunchConfig(
        Dim3(3, 2), Dim3(16, 16)
    )
    assert plan.orders == (
        LaunchOrder.canonical(),
        LaunchOrder.permuted(1),
        LaunchOrder.permuted(2),
        LaunchOrder.permuted(3),
    )
    assert plan.unguarded_demos == (
        UnguardedDemo("vector_add", (1000,)),
        UnguardedDemo("vector_add", (1024,)),
        UnguardedDemo("rgb_to_grayscale", (40, 30)),
        UnguardedDemo("rgb_to_grayscale", (48, 32)),
    )
    assert "No GPU runs, and no CUDA source is compiled." in plan.statement
    text = PINNED.read_text(encoding="ascii")
    assert plan.to_mapping() == json.loads(text)
    for marker in ("/Users/", "/home/", "/private/", "C:\\", "localhost"):
        assert marker not in text


def test_plans_round_trip_exactly():
    plan = KernelLaunchPlan.from_mapping(plan_mapping())

    assert plan.to_mapping() == plan_mapping()
    assert plan.demo_launch(plan.unguarded_demos[2]) == LaunchConfig(
        Dim3(3, 2), Dim3(4, 4)
    )


def mutated(path, value):
    mapping = copy.deepcopy(plan_mapping())
    *parents, last = path
    target = mapping
    for key in parents:
        target = target[key]
    if value is KeyError:
        del target[last]
    else:
        target[last] = value
    return mapping


@pytest.mark.parametrize(
    ("path", "value", "error", "message"),
    [
        (("extra",), 1, KernelLaunchError, "extra=\\['extra'\\]"),
        (("statement",), KeyError, KernelLaunchError, "missing=\\['statement'\\]"),
        (("schema_version",), 2, KernelLaunchError, "schema_version"),
        (("schema_version",), True, KernelLaunchError, "schema_version"),
        (("name",), "Test Plan", KernelLaunchError, "lowercase slug"),
        (("statement",), " padded", KernelLaunchError, "printable ASCII"),
        (("compute_capability",), "9.0", KernelLaunchError, "must be '8.0'"),
        (("vector_add", "seed"), 1, KernelLaunchError, "vector_add fields"),
        (("vector_add", "n"), 0, KernelLaunchError, "vector_add.n"),
        (("vector_add", "n"), 100.0, TypeError, "vector_add.n"),
        (("vector_add", "threads_per_block"), 2048, LaunchLimitError, "max_block_dim_x"),
        (("vector_add", "threads_per_block"), True, TypeError, "threads_per_block"),
        (("vector_add", "input_seed"), -1, KernelLaunchError, "input_seed"),
        (("rgb_to_grayscale", "block"), [64, 32], LaunchLimitError, "max_threads_per_block"),
        (("rgb_to_grayscale", "block"), [16, 16, 1], KernelLaunchError, "pair"),
        (("rgb_to_grayscale", "block"), (16, 16), KernelLaunchError, "JSON array"),
        (("rgb_to_grayscale", "width"), 4097, KernelLaunchError, "width"),
        (("unguarded_demos", 1, "extent"), [4096, 4096], KernelLaunchError, "at most 65536"),
        (("permuted_order_seeds",), [], KernelLaunchError, "1 to 8 seeds"),
        (("permuted_order_seeds",), [1, 1], KernelLaunchError, "repeat a seed"),
        (("permuted_order_seeds",), [False], TypeError, "order seed"),
        (("unguarded_demos",), [], KernelLaunchError, "1 to 8"),
        (("unguarded_demos", 0, "kernel"), "matmul", KernelLaunchError, "demo kernel"),
        (("unguarded_demos", 0, "extent"), [10, 10], KernelLaunchError, "1 axes"),
        (("unguarded_demos", 1, "extent"), [10], KernelLaunchError, "2 axes"),
        (("unguarded_demos", 1, "extent"), [0, 6], KernelLaunchError, "demo extent"),
        (("unguarded_demos", 2), {"kernel": "rgb_to_grayscale", "extent": [10, 6]}, KernelLaunchError, "repeat a demo"),
        (("unguarded_demos", 0, "extent"), [70_000], KernelLaunchError, "demo extent"),
    ],
)  # fmt: skip
def test_invalid_plans_name_the_field_or_limit_they_break(path, value, error, message):
    with pytest.raises(error, match=message):
        KernelLaunchPlan.from_mapping(mutated(path, value))


def test_plan_files_are_bounded_strict_and_never_coerced(tmp_path: Path):
    path = write_plan(tmp_path / "plan.json")
    assert load_plan(path) == KernelLaunchPlan.from_mapping(plan_mapping())

    path.write_bytes(b'{"name":"a","name":"b"}')
    with pytest.raises(KernelLaunchError, match="duplicate JSON key: name"):
        load_plan(path)
    write_plan(path, mutated(("vector_add", "n"), True))
    with pytest.raises(KernelLaunchError, match="invalid plan: vector_add.n"):
        load_plan(path)
    path.write_bytes(b" " * (MAX_PLAN_BYTES + 1))
    with pytest.raises(KernelLaunchError, match="exceeds 65536 bytes"):
        load_plan(path)
    link = tmp_path / "link.json"
    link.symlink_to(write_plan(tmp_path / "real.json"))
    with pytest.raises(KernelLaunchError, match="non-symlink"):
        load_plan(link)
    path.write_bytes(b"[]")
    with pytest.raises(KernelLaunchError, match="plan must be a JSON object"):
        load_plan(path)
