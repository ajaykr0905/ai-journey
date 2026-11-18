"""Mutants of the optimized buffer code must change what a launch records.

Each mutant undoes one detail of rounds 1, 3 or 4 in a private copy of
``kernel_launch_buffer``. The access plan below must record something else
under every mutant, so no optimization can drop that detail unseen. The
pinned misuse digest misses one of them: no misuse case has two writers of
one element where only the first writer reads it.
"""

import inspect
import sys
import types

import pytest

from ai_journey import kernel_launch_buffer
from ai_journey.kernel_launch_config import Dim3, KernelLaunchError, LaunchConfig

CONFIG = LaunchConfig(Dim3(2), Dim3(4))
SOURCE = inspect.getsource(kernel_launch_buffer)
READ_RACE = "read_race = bool(reading) and (racing or reading != {ordered[0]})"
MUTANTS = {
    "second-reader-lost": (
        "                readers.add(thread)\n",
        "                pass\n",
    ),
    "read-not-counted": ("            recorder._reads += 1\n", ""),
    "out-of-bounds-read-thread-lost": (
        "recorder._out_of_bounds.append((thread, self._name, READ, index))",
        "recorder._out_of_bounds.append((0, self._name, READ, index))",
    ),
    "read-race-needs-a-write-race": (
        READ_RACE,
        "read_race = bool(reading) and racing",
    ),
    "write-race-hides-a-self-read": (
        READ_RACE,
        "read_race = bool(reading) and reading != {ordered[0]}",
    ),
    "float32-overflow-stored": (
        "        except OverflowError:\n            exact = False",
        "        except OverflowError:\n            exact = True",
    ),
}
# Thread 6 and thread 7 write element 6, and only thread 6 reads it.
PLAN = {
    0: [("w", 0)],
    1: [("w", 1), ("r", 1), ("r", 9)],
    2: [("w", 2), ("r", 2), ("r", -1)],
    3: [("r", 2), ("w", 3)],
    4: [("w", 4), ("r", 3)],
    5: [("w", 5)],
    6: [("r", 6), ("w", 6), ("w", 5)],
    7: [("w", 6)],
}


def mutant(old, new):
    if old and SOURCE.count(old) != 1:
        raise AssertionError(f"the mutation site {old!r} is not unique")
    name = "ai_journey._mutant_buffer"
    module = types.ModuleType(name)
    module.__package__ = "ai_journey"
    # The dataclass decorator looks up the module of each class it builds.
    sys.modules[name] = module
    try:
        exec(compile(SOURCE.replace(old, new), "<mutant>", "exec"), module.__dict__)
    finally:
        del sys.modules[name]
    return module


def linear(ref):
    return ref["block"][0] * CONFIG.block.x + ref["thread"][0]


def records(module):
    out = module.DeviceBuffer("out", "float32", [0.0] * 8)
    recorder = module.AccessRecorder(CONFIG, [out])
    for thread_id, steps in PLAN.items():
        recorder.enter(thread_id)
        for kind, index in steps:
            if kind == "w":
                out[index] = float(thread_id)
            else:
                out[index]
        recorder.leave()
    log = recorder.finish()
    recorder = module.AccessRecorder(CONFIG, [out])
    recorder.enter(0)
    try:
        out[0] = 3.5e38
        overflow = "stored"
    except KernelLaunchError:
        overflow = "refused"
    recorder.leave()
    recorder.finish()
    return {
        "counts": log.counts(),
        "out_of_bounds": [item.to_mapping() for item in log.out_of_bounds],
        "write_conflicts": [item.to_mapping() for item in log.write_conflicts],
        "read_write_conflicts": [
            item.to_mapping() for item in log.read_write_conflicts
        ],
        "overflow": overflow,
        "values": list(out.to_host()),
    }


def test_the_unmutated_copy_records_what_the_module_records():
    expected = records(kernel_launch_buffer)

    assert records(mutant("", "")) == expected
    assert [item["index"] for item in expected["write_conflicts"]] == [5, 6]
    assert [
        (item["index"], [linear(reader) for reader in item["readers"]])
        for item in expected["read_write_conflicts"]
    ] == [(2, [2, 3]), (3, [4]), (6, [6])]
    assert expected["overflow"] == "refused"


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_every_mutant_changes_the_records(name):
    assert records(mutant(*MUTANTS[name])) != records(kernel_launch_buffer)
