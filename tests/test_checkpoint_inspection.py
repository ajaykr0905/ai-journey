"""Operators can inspect committed byte integrity without altering a journal."""

import fcntl
from hashlib import sha256
import json
import os

import pytest

from ai_journey.checkpoint_journal import CheckpointJournal

RUN = "1" * 64


def snapshot(directory):
    return {
        path.name: (path.lstat().st_mtime_ns, path.read_bytes())
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }


def test_inspection_reports_reserved_capacity_and_never_mutates_history(tmp_path):
    directory = tmp_path / "versions"
    store = CheckpointJournal(
        directory, run_sha256=RUN, max_versions=3, max_total_bytes=20
    )
    source = tmp_path / "source.pt"
    source.write_bytes(b"first")
    first = store.publish(source, step=2)
    source.write_bytes(b"second")
    second = store.publish(source, step=5)
    before = snapshot(directory)
    directory_mtime = directory.stat().st_mtime_ns
    opened = CheckpointJournal(
        directory, run_sha256=RUN, max_versions=3, max_total_bytes=20, create=False
    )
    report = opened.inspect()
    assert snapshot(directory) == before
    assert directory.stat().st_mtime_ns == directory_mtime
    assert report["recorded_step"] == 5
    assert (
        report["manifest_sha256"]
        == json.loads((directory / "manifest.json").read_bytes())["manifest_sha256"]
    )
    assert report["capacity"] == {
        "max_versions": 3,
        "used_versions": 2,
        "remaining_versions": 1,
        "max_total_bytes": 20,
        "committed_bytes": 11,
        "remaining_bytes": 9,
    }
    assert [(row["step"], row["integrity"]) for row in report["versions"]] == [
        (2, "intact"),
        (5, "intact"),
    ]
    assert [row["sha256"] for row in report["versions"]] == [
        first.sha256,
        second.sha256,
    ]
    assert report["unexpected_files"] == []
    assert report["scope"] == "byte integrity only; semantic restorability is untested"


@pytest.mark.parametrize("missing", ["directory", "lock", "manifest"])
def test_readonly_open_never_initializes_missing_journal_parts(tmp_path, missing):
    directory = tmp_path / "versions"
    if missing != "directory":
        CheckpointJournal(directory, run_sha256=RUN)
        (
            directory
            / ("manifest.json" if missing == "manifest" else ".checkpoint-journal.lock")
        ).unlink()
    before = snapshot(directory) if directory.exists() else None
    with pytest.raises((OSError, ValueError)):
        CheckpointJournal(directory, run_sha256=RUN, create=False)
    assert (snapshot(directory) if directory.exists() else None) == before


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("missing", "missing"),
        ("same_size", "sha256_mismatch"),
        ("size", "size_mismatch"),
        ("symlink", "symlink"),
        ("fifo", "not_regular"),
    ],
)
def test_inspection_explains_bad_versions_and_keeps_reserved_occupancy(
    tmp_path, damage, reason
):
    store = CheckpointJournal(tmp_path / "versions", run_sha256=RUN)
    source = tmp_path / "source.pt"
    source.write_bytes(b"original")
    candidate = store.publish(source, step=7)
    candidate.path.unlink()
    if damage == "same_size":
        candidate.path.write_bytes(b"modified")
    elif damage == "size":
        candidate.path.write_bytes(b"x")
    elif damage == "symlink":
        candidate.path.symlink_to(source)
    elif damage == "fifo":
        os.mkfifo(candidate.path)
    unexpected = store.directory / "uncommitted.pt"
    unexpected.write_bytes(b"orphan")
    report = store.inspect()
    row = report["versions"][0]
    assert row["integrity"] == reason
    assert row["size_bytes"] == 8
    assert row["sha256"] == sha256(b"original").hexdigest()
    assert report["capacity"]["committed_bytes"] == 8
    assert report["capacity"]["remaining_versions"] == 15
    assert report["unexpected_files"] == [
        {"filename": "uncommitted.pt", "kind": "regular", "size_bytes": 6}
    ]


def test_inspection_rejects_writer_contention_without_touching_any_file(tmp_path):
    store = CheckpointJournal(tmp_path / "versions", run_sha256=RUN)
    before = snapshot(store.directory)
    with (store.directory / ".checkpoint-journal.lock").open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="busy"):
            store.inspect()
        with pytest.raises(RuntimeError, match="busy"):
            CheckpointJournal(store.directory, run_sha256=RUN, create=False)
    assert snapshot(store.directory) == before


def test_inspection_can_share_a_read_lease_but_never_recreates_a_missing_lock(tmp_path):
    store = CheckpointJournal(tmp_path / "versions", run_sha256=RUN)
    lock = store.directory / ".checkpoint-journal.lock"
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        assert store.inspect()["versions"] == []
    lock.unlink()
    before = snapshot(store.directory)
    with pytest.raises(FileNotFoundError):
        store.inspect()
    assert snapshot(store.directory) == before
    assert not lock.exists()


def test_readonly_inspection_cli_prints_inventory_without_creating_an_output(
    tmp_path, capsys
):
    from ai_journey.checkpoint_journal_cli import main

    store = CheckpointJournal(tmp_path / "versions", run_sha256=RUN)
    before = snapshot(store.directory)
    arguments = ["--directory", str(store.directory), "--run-sha256", RUN]
    assert main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["recorded_step"] is None
    assert report["capacity"]["remaining_versions"] == 16
    assert snapshot(store.directory) == before
    arguments[-1] = "2" * 64
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "run, budget or manifest seal differs" in capsys.readouterr().err
    assert snapshot(store.directory) == before


def test_inspection_bounds_inventory_and_never_reads_unexpected_files(tmp_path):
    store = CheckpointJournal(tmp_path / "versions", run_sha256=RUN)
    os.mkfifo(store.directory / "uncommitted.fifo")
    report = store.inspect()
    assert report["unexpected_files"] == [
        {"filename": "uncommitted.fifo", "kind": "other", "size_bytes": 0}
    ]
    for index in range(256):
        (store.directory / f"extra-{index}").touch()
    before = snapshot(store.directory)
    with pytest.raises(ValueError, match="bounded file count"):
        store.inspect()
    assert snapshot(store.directory) == before


@pytest.mark.parametrize("create", [0, 1, "false", None])
def test_open_mode_rejects_nonboolean_values_before_creating_directory(
    tmp_path, create
):
    directory = tmp_path / "versions"
    with pytest.raises(ValueError, match="create must be a boolean"):
        CheckpointJournal(directory, run_sha256=RUN, create=create)
    assert not directory.exists()
