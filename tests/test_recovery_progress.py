import json
import os
from unittest.mock import patch

import pytest

from ai_journey.recovery_progress import (
    WorkerProgress,
    load_worker_progress,
    write_worker_progress,
)


def progress(**changes):
    values = {
        "run_id": "day-70-worker-0123456789abcdef",
        "attempt_id": "attempt-1-0123456789abcdef",
        "completed_items": 4,
        "accumulator_sha256": "a" * 64,
        "monotonic_ns": 100,
        "state": "running",
    }
    return WorkerProgress(**(values | changes))


def test_progress_round_trip_is_atomic_and_exact(tmp_path):
    path = tmp_path / "live" / "progress.json"
    write_worker_progress(path, progress())
    assert load_worker_progress(path) == progress()
    assert list(path.parent.iterdir()) == [path]


def test_failed_progress_replace_preserves_prior_bytes(tmp_path):
    path = tmp_path / "progress.json"
    path.write_bytes(b"prior")
    with (
        patch.object(os, "replace", side_effect=OSError("injected replace failure")),
        pytest.raises(OSError, match="injected"),
    ):
        write_worker_progress(path, progress())
    assert path.read_bytes() == b"prior"
    assert list(tmp_path.iterdir()) == [path]


def test_progress_loader_rejects_unknown_fields_and_versions(tmp_path):
    path = tmp_path / "progress.json"
    write_worker_progress(path, progress())
    payload = json.loads(path.read_bytes())
    path.write_text(json.dumps({**payload, "ignored": True}))
    with pytest.raises(ValueError, match="fields"):
        load_worker_progress(path)
    payload.pop("ignored", None)
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        load_worker_progress(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("completed_items", True, "integer"),
        ("monotonic_ns", -1, "non-negative"),
        ("accumulator_sha256", "bad", "SHA-256"),
        ("state", "failed-private", "unsupported"),
    ],
)
def test_progress_rejects_invalid_observations(field, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        progress(**{field: value})
