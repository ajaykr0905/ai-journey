"""Installed expert restart gates retain evidence and cannot reuse user outputs."""

import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib

import pytest

from ai_journey import moe_ddp_readiness as readiness

ROOT = Path(__file__).resolve().parents[1]


def test_source_import_admission_rejects_before_creating_output(tmp_path):
    output = tmp_path / "proof"
    with pytest.raises(ValueError, match="isolated installed package"):
        readiness.run_readiness(
            installed_root=tmp_path, repository=ROOT, output_dir=output
        )
    assert not output.exists()


def test_existing_output_admission_preserves_files(tmp_path, monkeypatch):
    installed = tmp_path / "installed"
    (installed / "bin").mkdir(parents=True)
    (installed / "bin/ai-journey-moe-ddp").touch()
    monkeypatch.setattr(readiness, "__file__", str(installed / "readiness.py"))
    repository = tmp_path / "repository"
    (repository / "data").mkdir(parents=True)
    (repository / "config").mkdir()
    (repository / "data/day-19-demo-names.txt").write_bytes(b"public test fixture\n")
    (repository / "config/moe-ddp-cpu.json").write_text("{}")
    output = tmp_path / "retained"
    output.mkdir()
    witness = output / "checkpoint.pt"
    witness.write_bytes(b"existing artifact")
    with pytest.raises(FileExistsError):
        readiness.run_readiness(
            installed_root=installed, repository=repository, output_dir=output
        )
    assert witness.read_bytes() == b"existing artifact"


def test_controller_timeout_reaps_only_owned_group_and_retains_logs(
    tmp_path, monkeypatch
):
    signals, options = [], []

    class OwnedProcess:
        pid = 53123
        returncode = None
        attempts = 0

        def communicate(self, *, timeout):
            self.attempts += 1
            if self.attempts == 1:
                raise subprocess.TimeoutExpired(["owned-test-command"], timeout)
            self.returncode = -9
            return "retained stdout", "retained stderr"

    def spawn(arguments, **kwargs):
        options.append(kwargs)
        return OwnedProcess()

    monkeypatch.setattr(readiness.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        readiness.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    with pytest.raises(RuntimeError, match="inspect retained installed MoE logs"):
        readiness._command(
            ["owned-test-command"],
            name="controlled",
            directory=tmp_path,
            environment={},
        )
    assert signals == [(53123, signal.SIGKILL)]
    assert options[0]["start_new_session"] is True
    assert (tmp_path / "controlled.stdout.txt").read_text() == "retained stdout"
    assert (tmp_path / "controlled.stderr.txt").read_text() == "retained stderr"


def test_actual_timeout_stops_owned_rank_even_if_it_ignores_termination(
    tmp_path, monkeypatch
):
    marker = tmp_path / "owned-rank.pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(8)"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(8)"
    )
    original = subprocess.Popen
    owned = []

    class ShortDeadline:
        def __init__(self, process):
            self.process = process

        @property
        def pid(self):
            return self.process.pid

        @property
        def returncode(self):
            return self.process.returncode

        def communicate(self, *, timeout):
            if timeout == 90:
                until = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < until:
                    time.sleep(0.01)
                assert marker.exists(), "owned test rank failed to start"
                timeout = 0.1
            return self.process.communicate(timeout=timeout)

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        owned.append(process)
        return ShortDeadline(process)

    monkeypatch.setattr(readiness.subprocess, "Popen", spawn)
    try:
        with pytest.raises(RuntimeError, match="inspect retained"):
            readiness._command(
                [sys.executable, "-c", parent_code],
                name="owned",
                directory=tmp_path,
                environment=dict(os.environ),
            )
        rank = int(marker.read_text())
        state = (
            original(
                ["ps", "-o", "stat=", "-p", str(rank)],
                stdout=subprocess.PIPE,
                text=True,
            )
            .communicate(timeout=2)[0]
            .strip()
        )
        assert not state or state.startswith("Z"), "owned rank still runs after cleanup"
        assert owned[0].returncode == -signal.SIGKILL
        assert (tmp_path / "owned.stdout.txt").is_file()
        assert (tmp_path / "owned.stderr.txt").is_file()
    finally:
        # Only our child, still verified in our created group, can authorize
        # fallback cleanup if this regression ever fails.
        if marker.exists():
            try:
                if os.getpgid(int(marker.read_text())) == owned[0].pid:
                    os.killpg(owned[0].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in owned:
            process.wait(timeout=2)


def test_console_plan_and_installed_exact_restart_are_required(capsys):
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())
    entry = package["project"]["scripts"]["ai-journey-moe-ddp"]
    module, attribute = entry.split(":")
    with pytest.raises(SystemExit) as result:
        getattr(importlib.import_module(module), attribute)(["--help"])
    assert result.value.code == 0
    assert "train" in capsys.readouterr().out
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "python -m ai_journey.moe_ddp_readiness --installed-root" in workflow
    required = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())[
        "required_files"
    ]
    for path in (
        "config/moe-ddp-cpu.json",
        "src/ai_journey/moe_ddp_readiness.py",
        "tests/test_moe_ddp_ci_contract.py",
    ):
        assert path in required and (ROOT / path).is_file()
