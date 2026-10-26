"""Immutable journal publication has a fault-tested manifest commit boundary."""

import fcntl
import json
import os

import pytest

from ai_journey.checkpoint_journal import CheckpointJournal, _seal

RUN = "1" * 64


def checkpoint(tmp_path, payload=b"public checkpoint fixture"):
    path = tmp_path / "latest.pt"
    path.write_bytes(payload)
    return path


def journal(tmp_path, **kwargs):
    return CheckpointJournal(tmp_path / "versions", run_sha256=RUN, **kwargs)


def test_publish_reopen_and_idempotent_retry_preserve_immutable_versions(tmp_path):
    store = journal(tmp_path)
    source = checkpoint(tmp_path)
    first = store.publish(source, step=1)
    before = first.path.read_bytes()
    assert store.publish(source, step=1) == first
    source.write_bytes(b"new independently completed checkpoint")
    second = store.publish(source, step=2)
    assert journal(tmp_path).candidates() == (second, first)
    assert first.path.read_bytes() == before
    with pytest.raises(ValueError, match="advance"):
        store.publish(source, step=1)


@pytest.mark.parametrize("damage", ["missing", "changed", "symlink"])
def test_only_byte_intact_committed_versions_are_candidates(tmp_path, damage):
    store = journal(tmp_path)
    source = checkpoint(tmp_path)
    first = store.publish(source, step=1)
    source.write_bytes(b"second checkpoint")
    second = store.publish(source, step=2)
    if damage == "missing":
        second.path.unlink()
    elif damage == "changed":
        second.path.write_bytes(b"x" * second.path.stat().st_size)
    else:
        second.path.unlink()
        second.path.symlink_to(first.path)
    assert store.candidates() == (first,)


def test_interrupted_manifest_publication_does_not_commit_an_orphan(
    tmp_path, monkeypatch
):
    store = journal(tmp_path)
    source = checkpoint(tmp_path)
    first = store.publish(source, step=1)
    manifest = store.directory / "manifest.json"
    committed = manifest.read_bytes()
    source.write_bytes(b"second checkpoint")
    replace = os.replace

    def interrupted(src, dst):
        if dst == manifest:
            raise OSError("injected manifest publication failure")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(OSError, match="injected"):
        store.publish(source, step=2)
    assert manifest.read_bytes() == committed
    assert store.candidates() == (first,)
    assert len(list(store.directory.glob("*.pt"))) == 2
    monkeypatch.setattr(os, "replace", replace)
    second = store.publish(source, step=2)
    assert store.candidates() == (second, first)
    assert len(list(store.directory.glob("*.pt"))) == 2


def test_orphan_cannot_accumulate_under_different_uncommitted_steps(
    tmp_path, monkeypatch
):
    store = journal(tmp_path)
    source = checkpoint(tmp_path)
    first = store.publish(source, step=1)
    source.write_bytes(b"uncommitted bytes")
    with monkeypatch.context() as patched:
        patched.setattr(
            os, "replace", lambda *a: (_ for _ in ()).throw(OSError("injected"))
        )
        with pytest.raises(OSError):
            store.publish(source, step=2)
    source.write_bytes(b"another uncommitted checkpoint")
    with pytest.raises(ValueError, match="uncommitted|unknown"):
        store.publish(source, step=3)
    assert store.candidates() == (first,)
    assert len(list(store.directory.glob("*.pt"))) == 2


@pytest.mark.parametrize("budget", ["versions", "bytes"])
def test_retention_budget_rejects_without_deleting_any_version(tmp_path, budget):
    source = checkpoint(tmp_path, b"1234")
    store = journal(
        tmp_path, max_versions=1 if budget == "versions" else 16, max_total_bytes=5
    )
    first = store.publish(source, step=1)
    before = {path.name: path.read_bytes() for path in store.directory.iterdir()}
    source.write_bytes(b"56")
    with pytest.raises(ValueError, match="retention budget"):
        store.publish(source, step=2)
    assert {
        path.name: path.read_bytes() for path in store.directory.iterdir()
    } == before
    assert store.candidates() == (first,)


@pytest.mark.parametrize("mutation", ["run", "step", "size", "path", "budget", "seal"])
def test_resealed_manifest_drift_is_not_accepted(tmp_path, mutation):
    store = journal(tmp_path)
    store.publish(checkpoint(tmp_path), step=1)
    path = store.directory / "manifest.json"
    value = json.loads(path.read_bytes())
    if mutation == "run":
        value["run_sha256"] = "2" * 64
    elif mutation == "step":
        value["entries"][0]["step"] = True
    elif mutation == "size":
        value["entries"][0]["size_bytes"] = True
    elif mutation == "path":
        value["entries"][0]["filename"] = "../latest.pt"
    elif mutation == "budget":
        value["max_versions"] = 16.0
    else:
        value["manifest_sha256"] = "0" * 64
    if mutation != "seal":
        value["manifest_sha256"] = _seal(
            {key: part for key, part in value.items() if key != "manifest_sha256"}
        )
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        store.candidates()


def test_reopening_another_run_or_budget_cannot_adopt_durable_versions(tmp_path):
    store = journal(tmp_path)
    store.publish(checkpoint(tmp_path), step=1)
    with pytest.raises(ValueError, match="run, budget"):
        CheckpointJournal(store.directory, run_sha256="2" * 64)
    with pytest.raises(ValueError, match="run, budget"):
        journal(tmp_path, max_versions=8)


def test_nonblocking_lease_rejects_concurrent_publish_without_truncation(tmp_path):
    store = journal(tmp_path)
    source = checkpoint(tmp_path)
    first = store.publish(source, step=1)
    with (store.directory / ".checkpoint-journal.lock").open("rb") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="busy"):
            store.publish(source, step=2)
    assert store.candidates() == (first,)


@pytest.mark.parametrize("target", ["input", "manifest", "lock"])
def test_symlink_targets_are_never_followed(tmp_path, target):
    store = journal(tmp_path)
    source = checkpoint(tmp_path)
    victim = tmp_path / "caller.txt"
    victim.write_bytes(b"preserve caller file")
    if target == "input":
        source.unlink()
        source.symlink_to(victim)
    else:
        path = store.directory / (
            "manifest.json" if target == "manifest" else ".checkpoint-journal.lock"
        )
        path.unlink()
        path.symlink_to(victim)
    with pytest.raises((OSError, ValueError)):
        store.publish(source, step=1)
    assert victim.read_bytes() == b"preserve caller file"


def test_new_journal_does_not_adopt_existing_user_files(tmp_path):
    directory = tmp_path / "versions"
    directory.mkdir()
    caller = directory / "caller.txt"
    caller.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="nonempty"):
        journal(tmp_path)
    assert caller.read_bytes() == b"preserve"


def test_fifo_input_rejects_without_waiting_for_a_writer(tmp_path):
    store = journal(tmp_path)
    source = tmp_path / "state.fifo"
    os.mkfifo(source)
    with pytest.raises(ValueError, match="regular-file"):
        store.publish(source, step=1)
    assert store.candidates() == ()


@pytest.mark.parametrize("value", [True, 1.0, -1, 1_000_001])
def test_inexact_or_out_of_range_steps_reject_before_publication(tmp_path, value):
    store = journal(tmp_path)
    with pytest.raises(ValueError, match="integer budget"):
        store.publish(checkpoint(tmp_path), step=value)
    assert store.candidates() == ()
