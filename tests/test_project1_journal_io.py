import json

import pytest

from ai_journey.project1_journal import Project1Event, chain_event
from ai_journey.project1_journal_io import load_event_chain, write_event_chain


def event() -> Project1Event:
    return Project1Event(
        sequence=0,
        attempt_id="attempt-1-0123456789abcdef",
        kind="started",
        recorded_at_utc="2026-11-07T07:30:00+00:00",
        completed_step=0,
        payload={},
    )


def test_journal_round_trips_through_atomic_canonical_storage(tmp_path):
    path = tmp_path / "nested" / "journal.json"
    records = [chain_event(event())]

    write_event_chain(path, records)

    assert load_event_chain(path) == (event(),)
    assert json.loads(path.read_text()) == records


def test_journal_loader_rejects_symlinks_oversize_and_invalid_json(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("[]")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        load_event_chain(link)
    target.write_bytes(b"x" * 5)
    with pytest.raises(ValueError, match="max_bytes"):
        load_event_chain(target, max_bytes=4)
    target.write_text("not json")
    with pytest.raises(ValueError, match="valid JSON"):
        load_event_chain(target)


def test_journal_writer_preserves_existing_file_after_rejected_chain(tmp_path):
    path = tmp_path / "journal.json"
    write_event_chain(path, [chain_event(event())])
    before = path.read_bytes()

    with pytest.raises(ValueError, match="sequence|ancestry"):
        write_event_chain(path, [chain_event(event()), chain_event(event())])

    assert path.read_bytes() == before
