import json

import pytest
from test_serving_recovery_load import record
from test_serving_recovery_client import PLAN
from test_serving_recovery_trace import HAPPY, events

from ai_journey.serving_recovery_io import (
    load_attempt_log,
    load_trace,
    publish_new_file,
    write_attempt_log,
    write_trace,
)
from ai_journey.serving_recovery_trace import chain_serving_trace


def test_publication_writes_exact_bytes_once_and_leaves_no_temporaries(tmp_path):
    path = tmp_path / "evidence.json"
    publish_new_file(path, b"{}", max_bytes=2)

    assert path.read_bytes() == b"{}"
    with pytest.raises(ValueError, match="refusing to overwrite evidence"):
        publish_new_file(path, b"[]", max_bytes=2)
    with pytest.raises(ValueError, match="bound"):
        publish_new_file(tmp_path / "large.json", b"[1]", max_bytes=2)
    assert path.read_bytes() == b"{}"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["evidence.json"]


def test_publication_refuses_protected_inputs_aliases_and_fake_parents(tmp_path):
    source = tmp_path / "source.txt"
    with pytest.raises(ValueError, match="protected"):
        publish_new_file(
            tmp_path / "." / "source.txt", b"x", max_bytes=1, protected=[source]
        )
    alias = tmp_path / "alias.json"
    alias.symlink_to(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="overwrite"):
        publish_new_file(alias, b"x", max_bytes=1)
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="real directory"):
        publish_new_file(linked / "x.json", b"x", max_bytes=1)
    assert not (tmp_path / "missing.json").exists()


def test_attempt_logs_round_trip_and_reject_edited_evidence(tmp_path):
    attempts = [record(PLAN[0], 1, "connection_refused"), record(PLAN[0], 2, "ok")]
    path = tmp_path / "attempts.json"
    write_attempt_log(path, attempts)

    assert load_attempt_log(path) == tuple(attempts)
    payload = json.loads(path.read_bytes())
    for edit, message in (
        (lambda p: json.dumps(p, indent=1), "canonical"),
        (
            lambda p: json.dumps(
                p | {"schema_version": 2}, sort_keys=True, separators=(",", ":")
            ),
            "schema",
        ),
        (
            lambda p: path.read_text().replace(
                '"attempt":1,', '"attempt":1,"attempt":1,'
            ),
            "duplicate",
        ),
    ):
        edited = tmp_path / "edited.json"
        edited.write_text(edit(payload))
        with pytest.raises(ValueError, match=message):
            load_attempt_log(edited)
        edited.unlink()


def test_traces_are_verified_before_and_after_persistence(tmp_path):
    records = chain_serving_trace(events(*HAPPY))
    path = tmp_path / "trace.json"
    write_trace(path, records)

    loaded, verified = load_trace(path)
    assert loaded == records
    assert [event.kind for event in verified][-1] == "finished"
    records[2]["payload"]["acknowledged_responses"] = 7
    with pytest.raises(ValueError, match="digest"):
        write_trace(tmp_path / "tampered.json", records)
    partial = chain_serving_trace(events(*HAPPY[:3]))
    write_trace(tmp_path / "partial.json", partial)
    with pytest.raises(ValueError, match="finishes"):
        load_trace(tmp_path / "partial.json")
    assert load_trace(tmp_path / "partial.json", require_finished=False)[0] == partial
