"""The recorded profiling loop: pinned rounds, final call counts and the guide."""

import sys

import pytest

from ai_journey.sim_profile_config import load_config
from ai_journey.sim_profile_harness import merge_functions, profile_call
from ai_journey.sim_profile_semantics import SEMANTICS_SHA256
from ai_journey.sim_profile_workloads import build_workloads
from test_sim_profile_fixtures import CONFIG, ROOT

GUIDE = ROOT / "docs/simulator-profiling.md"
BUFFER = "ai_journey.kernel_launch_buffer"
TARGETS = [
    f"{BUFFER}:DeviceBuffer.__getitem__",
    f"{BUFFER}:to_float32",
    "ai_journey.kernel_launch_config:Index3.__post_init__",
    f"{BUFFER}:_check_value",
]
# Exact cProfile counts of one pass of each pinned workload after round 4.
# "<string>:__init__" adds the generated __init__ of every dataclass:
# 18176 ThreadContext, 1678 Index3 and 28 others.
FINAL_CALLS = {
    f"{BUFFER}:DeviceBuffer.__getitem__": 150100,
    f"{BUFFER}:DeviceBuffer.__setitem__": 16100,
    f"{BUFFER}:to_float32": 123200,
    f"{BUFFER}:_check_value": 85920,
    "ai_journey.kernel_launch_config:Index3.__post_init__": 1678,
    "<string>:__init__": 19882,
}
REMOVED = (
    f"{BUFFER}:AccessRecorder._record_read",
    f"{BUFFER}:AccessRecorder._ref",
)
FINAL_TOTAL_CALLS = 1727437
# Total calls of each stage: the new harness, then the old pstats one.
STAGE_TOTALS = (
    ("baseline", 2853577, 2785398),
    ("round 1", 2403277, 2335098),
    ("round 2", 2403277, 2335098),
    ("round 3", 1823677, 1803798),
    ("round 4", FINAL_TOTAL_CALLS, 1707558),
)
ROUNDS_SHA256 = (
    "2f5e1e4dc739784bf6421ee4136a86a29760ce683e6e24bad991c404165430a9",
    "d458263e3239c23b565e7febcc084ca6f2cb3b3984e7037188c2765657e4cb22",
)


def guide():
    return " ".join(GUIDE.read_text(encoding="utf-8").split())


@pytest.fixture(scope="module")
def final_profile():
    records = []
    for workload in build_workloads(load_config(CONFIG)):
        functions, run = profile_call(workload.run)
        workload.check(run)
        records.extend(functions)
    return merge_functions(records)


def test_the_config_records_four_rounds_in_order():
    config = load_config(CONFIG)

    assert [item.target for item in config.rounds] == TARGETS
    assert config.stages == ("baseline", "round-1", "round-2", "round-3", "round-4")


def test_the_final_tree_makes_the_recorded_call_counts(final_profile):
    calls = {record.function: record.calls for record in final_profile}

    assert {name: calls[name] for name in FINAL_CALLS} == FINAL_CALLS
    assert not set(REMOVED) & set(calls)
    if sys.version_info[:2] == (3, 11):
        # Builtin call counts can differ on other Python versions.
        assert sum(calls.values()) == FINAL_TOTAL_CALLS


def test_the_guide_states_the_recorded_loop():
    text = guide()

    assert "All numbers in this guide are host-specific CPU measurements." in text
    assert "No number comes from hardware counters." in text
    assert "The parser uses the program name `ai-journey-sim-profile`." in text
    for count in (
        "2853577 to 2403277",
        "2403277 to 1823677",
        "1823677 to 1727437",
        "33878 to 1678",
    ):
        assert count in text
    assert "171320 to 123200" in text
    for digest in ROUNDS_SHA256:
        assert digest in text
    for row in (
        "| 3 | 1.402, 1.311, 1.347 | 1.291 | 1.334 |",
        "| All | | 2.079 | 1.508 |",
    ):
        assert row in text
    assert f"`{SEMANTICS_SHA256}`" in text
    assert (
        "826 out-of-bounds reads, 308 out-of-bounds writes, 464 write conflicts" in text
    )


def test_the_guide_gives_the_call_totals_of_both_harnesses():
    text = guide()

    for stage, new, old in STAGE_TOTALS:
        assert f"| {stage} | {new} | {old} |" in text
    assert "48300 in total" in text
    assert "The digests stay valid as records of that harness." in text
    assert "Call counts are exact and repeat" not in text


def test_the_guide_makes_no_gpu_or_vendor_claim():
    text = guide().lower()

    for claim in (
        "nsight",
        "speedup on cuda",
        "speedup on the gpu",
        "production speedup",
    ):
        assert claim not in text
    assert "no gpu ran" in text


def test_the_guide_has_no_stray_indented_prose():
    # A line that starts with one space after a list joins the last item.
    lines = GUIDE.read_text(encoding="utf-8").splitlines()
    fenced = False
    for number, line in enumerate(lines, start=1):
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced:
            assert not (line.startswith(" ") and line[1:2] != " "), number
    assert all(len(line) <= 88 for line in lines if "`" not in line)
