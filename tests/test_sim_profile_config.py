import copy
import json
from pathlib import Path

import pytest

from ai_journey.kernel_launch_config import KernelLaunchError, LaunchLimitError
from ai_journey.sim_profile_config import (
    BASELINE,
    WORKLOADS,
    RoundSpec,
    SimProfileConfig,
    SimProfileError,
    load_config,
    stage_name,
)
from ai_journey.sim_profile_workloads import build_workloads

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/day-75-sim-profile.json"


def raw():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def parse(mapping):
    return SimProfileConfig.from_mapping(mapping)


def test_the_pinned_config_round_trips_and_names_three_ragged_workloads():
    config = load_config(CONFIG)

    assert config.to_mapping() == raw()
    assert config.name == "day-75-sim-profile"
    assert tuple(config.workloads_mapping()) == WORKLOADS
    assert (config.top_functions, config.warmup, config.repeats) == (10, 2, 21)
    assert config.stages[0] == BASELINE
    assert len(config.stages) == len(config.rounds) + 1
    launched = [
        workload.config.launched_threads for workload in build_workloads(config)
    ]
    assert launched == [8192, 7680, 2304]


def test_the_workload_digest_ignores_the_rounds_only():
    mapping = raw()
    base = parse(mapping).workload_sha256
    mapping["rounds"] = [
        {"target": "ai_journey.kernel_launch_buffer:to_float32", "change": "x"}
    ]
    assert parse(mapping).workload_sha256 == base
    mapping["timing"]["repeats"] = 22
    assert parse(mapping).workload_sha256 != base


def test_every_workload_pass_matches_its_reference_digest():
    for workload in build_workloads(load_config(CONFIG)):
        run = workload.checked_run()
        assert run.output_sha256 == workload.reference_sha256
        assert run.result.log.memory_safe and run.result.log.race_free
        assert run.result.threads_run == workload.config.launched_threads


def test_a_pass_with_another_output_is_refused():
    workload = build_workloads(load_config(CONFIG))[0]
    other = build_workloads(load_config(CONFIG))[1].run()

    with pytest.raises(SimProfileError, match="differs from its reference"):
        workload.check(other)
    with pytest.raises(SimProfileError, match="differs from its reference"):
        workload.check("run")


@pytest.mark.parametrize(
    ("path", "value", "error", "message"),
    [
        (("schema_version",), 2, SimProfileError, "schema_version"),
        (("schema_version",), True, SimProfileError, "schema_version"),
        (("name",), "Day 75", SimProfileError, "lowercase slug"),
        (("statement",), " padded", SimProfileError, "printable ASCII"),
        (("workloads", "vector_add", "n"), 0, KernelLaunchError, "between 1"),
        (("workloads", "vector_add", "n"), 8000.0, TypeError, "integer"),
        (("workloads", "vector_add", "threads_per_block"), 2048, LaunchLimitError, "max_block_dim_x"),
        (("workloads", "rgb_to_grayscale", "width"), 300, SimProfileError, "16384 pixels"),
        (("workloads", "rgb_to_grayscale", "block"), [16], SimProfileError, "pair"),
        (("workloads", "rgb_to_grayscale", "block"), [64, 32], LaunchLimitError, "max_threads_per_block"),
        (("workloads", "naive_matmul", "k"), 65, SimProfileError, "at most 64"),
        (("workloads", "naive_matmul", "m"), 128, SimProfileError, "multiply-adds"),
        (("workloads", "naive_matmul", "mapping"), "x_cols", SimProfileError, "mapping"),
        (("workloads", "naive_matmul", "input_seed"), -1, KernelLaunchError, "input_seed"),
        (("profile", "top_functions"), 0, KernelLaunchError, "top_functions"),
        (("timing", "warmup"), 11, KernelLaunchError, "warmup"),
        (("timing", "repeats"), 4, KernelLaunchError, "repeats"),
        (("timing", "repeats"), False, TypeError, "repeats"),
    ],
)  # fmt: skip
def test_invalid_fields_are_refused(path, value, error, message):
    mapping = raw()
    target = mapping
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(error, match=message):
        parse(mapping)


def test_missing_extra_and_launch_heavy_fields_are_refused():
    for mutate, message in (
        (lambda m: m.pop("timing"), "missing=\\['timing'\\]"),
        (lambda m: m.update(seed=1), "extra=\\['seed'\\]"),
        (lambda m: m["workloads"].pop("naive_matmul"), "workloads fields"),
        (lambda m: m["profile"].update(sort="calls"), "profile fields"),
        (lambda m: m["workloads"]["vector_add"].update(n=16384, threads_per_block=1000), "threads"),
    ):  # fmt: skip
        mapping = copy.deepcopy(raw())
        mutate(mapping)
        with pytest.raises(ValueError, match=message):
            parse(mapping)


def test_rounds_name_package_functions_and_stages_follow_them():
    mapping = raw()
    mapping["rounds"] = [
        {"target": "ai_journey.kernel_launch_buffer:DeviceBuffer.__getitem__", "change": "Inline the read record."},
        {"target": "ai_journey.kernel_launch_buffer:to_float32", "change": "Reuse one Struct."},
    ]  # fmt: skip
    config = parse(mapping)

    assert config.stages == ("baseline", "round-1", "round-2")
    assert config.rounds[0] == RoundSpec(
        "ai_journey.kernel_launch_buffer:DeviceBuffer.__getitem__",
        "Inline the read record.",
    )
    assert stage_name(3) == "round-3"
    for target in ("struct:pack", "ai_journey.x", "numpy.core:dot", ""):
        with pytest.raises(SimProfileError, match="module:qualname"):
            RoundSpec(target, "change")
    with pytest.raises(SimProfileError, match="printable ASCII"):
        RoundSpec("ai_journey.kernel_launch_buffer:to_float32", "")
    mapping["rounds"] = mapping["rounds"] * 5
    with pytest.raises(SimProfileError, match="0 to 8 rounds"):
        parse(mapping)


def test_the_config_loader_reads_only_strict_bounded_json(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"name": 1, "name": 2}')
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (64 * 1024 + 1))
    link = tmp_path / "link.json"
    link.symlink_to(CONFIG)

    with pytest.raises(KernelLaunchError, match="duplicate JSON key"):
        load_config(duplicate)
    with pytest.raises(KernelLaunchError, match="exceeds"):
        load_config(big)
    with pytest.raises(KernelLaunchError, match="non-symlink"):
        load_config(link)
    with pytest.raises(TypeError, match="pathlib.Path"):
        load_config(str(CONFIG))
