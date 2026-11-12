import json
from pathlib import Path

import pytest

from ai_journey.verifiable_reward_cli import load_reward_fixture, main


CONFIG = Path("config/day-69-verifiable-rewards.json")


def test_pinned_fixture_builds_verifies_and_replays(tmp_path, capsys):
    report = tmp_path / "report.json"
    assert main(["build", "--config", str(CONFIG), "--output", str(report)]) == 0
    built = json.loads(capsys.readouterr().out)
    assert (built["candidate_count"], built["correct_count"]) == (12, 6)

    assert main(["verify", "--config", str(CONFIG), "--report", str(report)]) == 0
    assert json.loads(capsys.readouterr().out)["candidate_count"] == 12
    assert main(["replay", "--config", str(CONFIG), "--report", str(report)]) == 0
    assert json.loads(capsys.readouterr().out)["byte_exact"] is True


def test_pinned_fixture_records_expected_failure_and_retry_metrics():
    tasks, groups, config, faults = load_reward_fixture(CONFIG)
    assert [task.task_id for task in tasks] == [group.task_id for group in groups]
    assert config.seed == 69
    assert faults.kind_for("exact-restart-term", 0, 1) == "transient"


def test_cli_refuses_to_overwrite_config_input():
    with pytest.raises(SystemExit):
        main(["build", "--config", str(CONFIG), "--output", str(CONFIG)])
