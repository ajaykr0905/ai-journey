import subprocess
import sys
from unittest.mock import patch

import pytest

from ai_journey.recovery_process import start_worker_process, worker_command


def options(tmp_path, **changes):
    values = {
        "source_path": tmp_path / "source.txt",
        "config_path": tmp_path / "config.json",
        "run_id": "day-70-worker-0123456789abcdef",
        "attempt_id": "attempt-1-0123456789abcdef",
        "checkpoint_path": tmp_path / "checkpoint.json",
        "progress_path": tmp_path / "progress.json",
        "start_items": 0,
        "fault_after_items": 11,
    }
    return values | changes


def test_worker_command_uses_fixed_module_and_separate_arguments(tmp_path):
    command = worker_command(**options(tmp_path))

    assert command[:3] == (
        sys.executable,
        "-m",
        "ai_journey.recovery_worker_cli",
    )
    assert "--fault-after-items" in command
    assert command[-1] == "11"
    assert all(";" not in argument for argument in command)


def test_process_launch_disables_shell_and_inherited_streams(tmp_path):
    sentinel = object()
    with patch.object(subprocess, "Popen", return_value=sentinel) as popen:
        assert start_worker_process(**options(tmp_path)) is sentinel

    command = worker_command(**options(tmp_path))
    popen.assert_called_once_with(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"start_items": True}, "integer"),
        ({"start_items": -1}, "non-negative"),
        ({"fault_after_items": True}, "integer"),
        ({"start_items": 8, "fault_after_items": 8}, "follow"),
        ({"source_path": "source.txt"}, "pathlib.Path"),
    ],
)
def test_worker_command_rejects_ambiguous_controls(tmp_path, changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        worker_command(**options(tmp_path, **changes))
