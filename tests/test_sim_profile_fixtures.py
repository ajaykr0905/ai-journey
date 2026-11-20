"""Shared small configs and fake clocks for the sim_profile tests."""

import itertools
import json
from pathlib import Path

from ai_journey.sim_profile_config import RoundSpec, SimProfileConfig, SimProfileError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/day-75-sim-profile.json"
LOAD = (1.25, 1.5, 2.0)


def pinned_mapping():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def small_config(rounds=()):
    """The pinned config shrunk to quick workloads and five repeats."""

    mapping = pinned_mapping()
    mapping["workloads"]["vector_add"]["n"] = 300
    mapping["workloads"]["rgb_to_grayscale"].update(width=20, height=10)
    mapping["workloads"]["naive_matmul"].update(m=9, n=10, k=4)
    mapping["timing"] = {"warmup": 0, "repeats": 5}
    mapping["rounds"] = list(rounds)
    return SimProfileConfig.from_mapping(mapping)


def _may_target(function):
    try:
        RoundSpec(function, "A probe.")
    except SimProfileError:
        return False
    return True


def round_targets(report, count):
    """The first ``count`` ranked hotspots of a snapshot that a round may target.

    A round must target a package function. In these small profiles the summed
    ``<string>:__init__`` entry or a builtin can rank first, so it is skipped.
    """

    spots = report["profile"]["hotspots"]
    targets = [spot["function"] for spot in spots if _may_target(spot["function"])]
    assert len(targets) >= count, spots
    return targets[:count]


def stepping_clock(step):
    counter = itertools.count(0, step)
    return lambda: next(counter)


def fixed_load():
    return LOAD
