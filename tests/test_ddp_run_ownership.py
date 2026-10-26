from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from ai_journey.ddp_launch import launch_training

ROOT = Path(__file__).resolve().parents[1]


class SpawnStopped(RuntimeError):
    pass


def arguments(tmp_path):
    return {
        "source": ROOT / "data/day-19-demo-names.txt",
        "spec": ROOT / "config/ddp-lab.json",
        "checkpoint": tmp_path / "state.pt",
        "output_dir": tmp_path / "workers",
        "world_size": 1,
        "steps": 1,
    }


@pytest.mark.parametrize("shared", ["output_dir", "checkpoint"])
def test_overlapping_launchers_reject_before_truncating_owned_artifacts(
    tmp_path, monkeypatch, shared
):
    first = arguments(tmp_path)
    second = dict(first)
    if shared == "output_dir":
        second["checkpoint"] = tmp_path / "other.pt"
    else:
        second["output_dir"] = tmp_path / "other-workers"
    entered, release = threading.Event(), threading.Event()
    attempts = []

    def stop_spawn(*args, stdout, **kwargs):
        attempts.append(args)
        if len(attempts) == 1:
            stdout.write(b"preserve the active run diagnostic\n")
            stdout.flush()
            entered.set()
            assert release.wait(5)
        raise SpawnStopped("no real child was spawned")

    monkeypatch.setattr(subprocess, "Popen", stop_spawn)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(launch_training, **first)
        try:
            assert entered.wait(5)
            log = first["output_dir"] / "rank-0.log"
            expected = log.read_bytes()
            with pytest.raises(ValueError, match="already owned"):
                launch_training(**second)
            assert len(attempts) == 1
            assert log.read_bytes() == expected
            if shared == "checkpoint":
                assert not (second["output_dir"] / "rank-0.log").exists()
        finally:
            release.set()
            with pytest.raises(SpawnStopped):
                future.result(timeout=5)


@pytest.mark.parametrize("collision", ["symlink", "hardlink", "fifo", "content"])
def test_reserved_checkpoint_lease_never_overwrites_existing_files(
    tmp_path, monkeypatch, collision
):
    args = arguments(tmp_path)
    valuable = tmp_path / "valuable.txt"
    valuable.write_bytes(b"keep unrelated data")
    lock = tmp_path / ".state.pt.ai-journey-ddp.lock"
    if collision == "symlink":
        lock.symlink_to(valuable)
    elif collision == "hardlink":
        lock.hardlink_to(valuable)
    elif collision == "fifo":
        os.mkfifo(lock)
    else:
        lock.write_bytes(b"keep preexisting sidecar")

    def unexpected(*args, **kwargs):
        raise AssertionError("unsafe lease must reject before opening rank logs")

    monkeypatch.setattr(subprocess, "Popen", unexpected)
    with pytest.raises(ValueError, match="lease"):
        launch_training(**args)
    assert valuable.read_bytes() == b"keep unrelated data"
    assert not (args["output_dir"] / "rank-0.log").exists()
    if collision == "content":
        assert lock.read_bytes() == b"keep preexisting sidecar"


def test_dead_supervisor_lease_is_released_without_deleting_or_killing(tmp_path):
    args = arguments(tmp_path)
    # Only this explicitly spawned helper exits. No PID probing or foreign
    # process signals are used to recover ownership after its abrupt death.
    code = """
import os, sys
from pathlib import Path
from ai_journey.ddp_launch import _owned_run
with _owned_run(Path(sys.argv[1]), Path(sys.argv[2])):
    print('owned', flush=True)
    os._exit(87)
"""
    owner = subprocess.Popen(
        [sys.executable, "-c", code, str(args["checkpoint"]), str(args["output_dir"])],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = owner.communicate(timeout=10)
        assert owner.returncode == 87, stderr
        assert stdout.strip() == "owned"
        lock = tmp_path / ".state.pt.ai-journey-ddp.lock"
        inode = lock.stat().st_ino
        ranks = launch_training(**args)
        assert ranks[0]["completed_step"] == 1
        assert lock.stat().st_ino == inode
        assert lock.read_bytes() == b""
    finally:
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=3)


@pytest.mark.parametrize("error_type", [SpawnStopped, PermissionError])
def test_failed_spawn_releases_both_leases_and_preserves_the_exception(
    tmp_path, monkeypatch, error_type
):
    args = arguments(tmp_path)
    attempts = []

    def fail(*values, **options):
        attempts.append(values)
        raise error_type("controlled child startup failure")

    monkeypatch.setattr(subprocess, "Popen", fail)
    for _ in range(2):
        with pytest.raises(error_type, match="controlled child startup failure"):
            launch_training(**args)
    assert len(attempts) == 2


def test_independent_runs_do_not_share_a_global_lease(tmp_path):
    from ai_journey.ddp_launch import _owned_run

    with _owned_run(tmp_path / "one.pt", tmp_path / "one"):
        with _owned_run(tmp_path / "two.pt", tmp_path / "two"):
            assert (tmp_path / ".one.pt.ai-journey-ddp.lock").is_file()
            assert (tmp_path / ".two.pt.ai-journey-ddp.lock").is_file()
