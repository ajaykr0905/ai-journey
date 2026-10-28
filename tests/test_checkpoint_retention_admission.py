"""Finite journal slots admit exact global publication boundaries, not updates."""

import pytest

from ai_journey.checkpoint_journal import CheckpointJournal

RUN = "1" * 64


@pytest.mark.parametrize(
    "start,steps,interval,expected",
    [
        (0, 16, 1, 16),
        (0, 32, 2, 16),
        (0, 31, 2, 16),
        (0, 128, None, 1),
        (7, 5, 3, 2),
        (7, 6, 3, 3),
        (7, 6, 10**30, 1),
    ],
)
def test_fresh_publication_count_matches_explicit_interval_boundaries(
    tmp_path, start, steps, interval, expected
):
    store = CheckpointJournal(tmp_path / "versions", run_sha256=RUN)
    result = store.admit_schedule(
        start_step=start, steps=steps, checkpoint_interval=interval
    )
    assert result["required_new_versions"] == expected
    assert result["remaining_versions_after"] == 16 - expected
    assert result["recorded_step"] is None
    assert result["scope"] == (
        "version capacity only; future checkpoint byte sizes are not predicted"
    )
    assert store.inspect()["capacity"]["used_versions"] == 0


@pytest.mark.parametrize("steps,interval", [(17, 1), (33, 2)])
def test_interval_and_final_partial_overflow_reject_without_journal_changes(
    tmp_path, steps, interval
):
    store = CheckpointJournal(tmp_path / "versions", run_sha256=RUN)
    before = {p.name: p.read_bytes() for p in store.directory.iterdir()}
    with pytest.raises(ValueError, match="durable version capacity"):
        store.admit_schedule(start_step=0, steps=steps, checkpoint_interval=interval)
    assert {p.name: p.read_bytes() for p in store.directory.iterdir()} == before


def test_replayed_boundaries_and_corrupt_versions_still_use_correct_slots(tmp_path):
    store = CheckpointJournal(tmp_path / "versions", run_sha256=RUN, max_versions=4)
    source = tmp_path / "source.pt"
    source.write_bytes(b"committed bytes")
    store.publish(source, step=1)
    newest = store.publish(source, step=5)
    newest.path.write_bytes(b"corrupt")
    before = {p.name: p.read_bytes() for p in store.directory.iterdir()}
    result = store.admit_schedule(start_step=1, steps=7, checkpoint_interval=2)
    # Replayed2/4 are below high-water5. New6/8 consume exactly two slots.
    assert result["recorded_step"] == 5
    assert result["required_new_versions"] == 2
    assert result["remaining_versions_after"] == 0
    with pytest.raises(ValueError, match="durable version capacity"):
        store.admit_schedule(start_step=1, steps=8, checkpoint_interval=2)
    assert {p.name: p.read_bytes() for p in store.directory.iterdir()} == before


@pytest.mark.parametrize(
    "arguments",
    [
        {"start_step": True, "steps": 1},
        {"start_step": -1, "steps": 1},
        {"start_step": 0, "steps": 0},
        {"start_step": 0, "steps": 1.0},
        {"start_step": 1_000_000, "steps": 1},
        {"start_step": 0, "steps": 1, "checkpoint_interval": True},
        {"start_step": 0, "steps": 1, "checkpoint_interval": 0},
    ],
)
def test_invalid_schedule_cannot_change_committed_inventory(tmp_path, arguments):
    store = CheckpointJournal(tmp_path / "versions", run_sha256=RUN)
    before = store.inspect()
    with pytest.raises(ValueError):
        store.admit_schedule(**arguments)
    assert store.inspect() == before
