import os
from unittest.mock import patch

import pytest
from test_recovery_checkpoint import checkpoint

from ai_journey.recovery_checkpoint_io import (
    load_recovery_checkpoint,
    write_recovery_checkpoint,
)


def test_checkpoint_round_trip_is_canonical_and_atomic(tmp_path):
    path = tmp_path / "state" / "checkpoint.json"

    digest = write_recovery_checkpoint(path, checkpoint())

    assert load_recovery_checkpoint(path) == checkpoint()
    assert digest == checkpoint().to_mapping()["checkpoint_sha256"]
    assert list(path.parent.glob("*")) == [path]


def test_failed_replace_preserves_prior_checkpoint_and_cleans_temporary(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_bytes(b"prior bytes")

    with (
        patch.object(os, "replace", side_effect=OSError("injected replace failure")),
        pytest.raises(OSError, match="injected"),
    ):
        write_recovery_checkpoint(path, checkpoint())

    assert path.read_bytes() == b"prior bytes"
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_loader_rejects_symlinks_invalid_json_and_size(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(real)
    with pytest.raises(ValueError, match="regular non-symlink"):
        load_recovery_checkpoint(alias)
    with pytest.raises(ValueError, match="valid JSON"):
        real.write_text("not-json")
        load_recovery_checkpoint(real)
    real.write_bytes(b"12345")
    with pytest.raises(ValueError, match="exceeds"):
        load_recovery_checkpoint(real, max_bytes=4)
