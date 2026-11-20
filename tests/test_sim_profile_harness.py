import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ai_journey import kernel_launch_buffer
from ai_journey.kernel_launch_config import KernelLaunchError
from ai_journey.sim_profile_config import SimProfileError, load_config
from ai_journey.sim_profile_harness import (
    FunctionProfile,
    function_name,
    merge_functions,
    profile_call,
    rank_hotspots,
)
from ai_journey.sim_profile_workloads import build_workloads

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/day-75-sim-profile.json"
GETITEM = "ai_journey.kernel_launch_buffer:DeviceBuffer.__getitem__"
SETITEM = "ai_journey.kernel_launch_buffer:DeviceBuffer.__setitem__"
LAUNCH = "ai_journey.kernel_launch_simulator:launch"


@pytest.fixture(scope="module")
def workloads():
    return build_workloads(load_config(CONFIG))


def outer():
    def inner():
        return [value for value in (1, 2)]

    return inner


def code_name(function):
    return function_name(function.__code__)


@dataclass(frozen=True)
class Pair:
    left: int
    right: int


@dataclass(frozen=True)
class Single:
    value: int


def countdown(steps):
    return steps if steps == 0 else countdown(steps - 1)


def test_function_names_carry_module_and_qualname():
    getitem = kernel_launch_buffer.DeviceBuffer.__getitem__
    nested = outer()

    assert code_name(getitem) == GETITEM
    assert code_name(kernel_launch_buffer.to_float32) == (
        "ai_journey.kernel_launch_buffer:to_float32"
    )
    assert code_name(nested) == "test_sim_profile_harness:outer.<locals>.inner"
    assert code_name(json.dumps) == "json:dumps"
    assert function_name("<built-in method builtins.len>") == (
        "builtins:<built-in method builtins.len>"
    )
    assert code_name(Single.__init__) == "<string>:__init__"
    # The qualname comes from the code object, not from the source file.
    moved = nested.__code__.replace(co_filename=str(ROOT / "missing.py"))
    assert function_name(moved) == "missing:outer.<locals>.inner"


def test_comprehensions_get_their_python_qualnames():
    def collect():
        return sorted({key: [item for item in range(key)] for key in range(3)})

    records, result = profile_call(collect)
    names = {record.function for record in records}

    assert result == [0, 1, 2]
    prefix = "test_sim_profile_harness:test_comprehensions_get_their_python_qualnames"
    assert f"{prefix}.<locals>.collect.<locals>.<dictcomp>" in names
    # Python adds no "<locals>" part for a comprehension in a comprehension.
    assert f"{prefix}.<locals>.collect.<locals>.<dictcomp>.<listcomp>" in names


def test_call_counts_are_exact_and_repeat(workloads):
    vector, gray, _ = workloads
    first, run = profile_call(vector.checked_run)
    second, _ = profile_call(vector.checked_run)
    calls = {record.function: record.calls for record in first}

    assert [(r.function, r.calls) for r in first] == [
        (r.function, r.calls) for r in second
    ]
    assert run.output_sha256 == vector.reference_sha256
    log = run.result.log
    assert calls[GETITEM] == log.reads + log.out_of_bounds_reads == 16000
    assert calls[SETITEM] == log.writes == 8000
    assert calls[LAUNCH] == 1
    assert not any("_lsprof.Profiler" in name for name in calls)
    assert [record.function for record in first] == sorted(calls)
    gray_calls = {r.function: r.calls for r in profile_call(gray.checked_run)[0]}
    assert gray_calls[GETITEM] == 3 * 90 * 70


def test_generated_inits_of_two_dataclasses_add_up():
    # Both generated __init__ methods have the label ("<string>", 2, "__init__").
    # pstats keeps one entry per label, so it reports 2 or 3 calls, never 5.
    records, made = profile_call(
        lambda: [Pair(1, 2), Pair(3, 4), Single(5), Single(6), Single(7)]
    )
    inits = {record.function: record for record in records}["<string>:__init__"]

    assert len(made) == 5
    assert (inits.calls, inits.primitive_calls) == (5, 5)


def test_recursive_calls_are_primitive_only_at_the_outer_call():
    records, _ = profile_call(lambda: countdown(3))
    calls = {r.function: (r.calls, r.primitive_calls) for r in records}

    assert calls["test_sim_profile_harness:countdown"] == (4, 1)


def record(name, calls, total, cumulative=None):
    return FunctionProfile(
        name, calls, calls, total, total if cumulative is None else cumulative
    )


def test_merging_sums_entries_that_share_a_name():
    merged = merge_functions(
        [record("b:f", 2, 10, 30), record("a:g", 1, 5), record("b:f", 3, 7, 9)]
    )

    assert merged == (record("a:g", 1, 5), FunctionProfile("b:f", 5, 5, 17, 39))
    with pytest.raises(TypeError, match="FunctionProfile"):
        merge_functions([("b:f", 1)])


def test_hotspots_rank_by_self_time_with_name_ties_and_floor_shares():
    records = (
        record("c:h", 1, 30),
        record("a:f", 4, 50, 90),
        record("b:g", 2, 50),
        record("d:i", 9, 0),
    )
    ranked = rank_hotspots(records, 3)

    assert [(h.rank, h.function, h.share_ppm) for h in ranked] == [
        (1, "a:f", 384615),
        (2, "b:g", 384615),
        (3, "c:h", 230769),
    ]
    assert ranked[0].to_mapping() == {
        "rank": 1,
        "function": "a:f",
        "calls": 4,
        "total_ns": 50,
        "cumulative_ns": 90,
        "share_ppm": 384615,
    }
    assert len(rank_hotspots(records, 25)) == 4
    with pytest.raises(SimProfileError, match="no time"):
        rank_hotspots((record("d:i", 9, 0),), 1)
    with pytest.raises(SimProfileError, match="at least one"):
        rank_hotspots((), 1)
    with pytest.raises(KernelLaunchError, match="top"):
        rank_hotspots(records, 26)


def test_function_profiles_are_validated():
    for args, error in (
        (("", 1, 1, 0, 0), SimProfileError),
        (("a:f", 0, 0, 0, 0), KernelLaunchError),
        (("a:f", 1, 2, 0, 0), KernelLaunchError),
        (("a:f", 1, 1, -1, 0), KernelLaunchError),
        (("a:f", 1, 1, 0, 1.0), TypeError),
    ):
        with pytest.raises(error):
            FunctionProfile(*args)
    with pytest.raises(TypeError, match="callable"):
        profile_call(None)
