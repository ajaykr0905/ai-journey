import json
import os
from unittest.mock import patch

import pytest
from test_recovery_trace import event

from ai_journey.recovery_trace_io import (
    append_recovery_event,
    load_recovery_trace,
    write_recovery_trace,
)
from ai_journey.recovery_trace_verify import verify_recovery_trace


def test_append_persists_verified_prefix_then_completed_trace(tmp_path):
    path = tmp_path / "trace.json"
    append_recovery_event(path, event())
    prefix = load_recovery_trace(path, require_completed=False)
    assert [item.kind for item in prefix] == ["started"]

    append_recovery_event(
        path,
        event(sequence=1, kind="completed", monotonic_ns=120, completed_items=2),
    )

    complete = load_recovery_trace(path)
    assert [item.kind for item in complete] == ["started", "completed"]
    assert complete[-1].completed_items == 2


def test_append_rejects_gaps_and_completed_extension(tmp_path):
    path = tmp_path / "trace.json"
    append_recovery_event(path, event())
    with pytest.raises(ValueError, match="contiguous"):
        append_recovery_event(path, event(sequence=2, kind="heartbeat"))
    append_recovery_event(path, event(sequence=1, kind="completed"))
    with pytest.raises(ValueError, match="cannot be extended"):
        append_recovery_event(path, event(sequence=2, kind="heartbeat"))


def test_failed_trace_replace_preserves_prior_bytes_and_cleans_temporary(tmp_path):
    path = tmp_path / "trace.json"
    append_recovery_event(path, event())
    before = path.read_bytes()
    with (
        patch.object(os, "replace", side_effect=OSError("injected replace failure")),
        pytest.raises(OSError, match="injected"),
    ):
        append_recovery_event(path, event(sequence=1, kind="completed"))

    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_loader_rejects_tampered_and_oversized_trace(tmp_path):
    path = tmp_path / "trace.json"
    append_recovery_event(path, event())
    append_recovery_event(path, event(sequence=1, kind="completed"))
    records = json.loads(path.read_bytes())
    records[0]["completed_items"] = 1
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="digest"):
        load_recovery_trace(path)
    with pytest.raises(ValueError, match="exceeds"):
        load_recovery_trace(path, max_bytes=1)


def test_full_trace_writer_requires_completion(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_recovery_trace(tmp_path / "trace.json", [])
    with pytest.raises(TypeError, match="boolean"):
        verify_recovery_trace([], require_completed=1)
