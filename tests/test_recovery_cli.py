import json

from test_recovery_config import valid_config

from ai_journey.recovery_cli import main


def test_cli_runs_and_independently_verifies_recovery_evidence(tmp_path, capsys):
    source = tmp_path / "source.txt"
    source.write_bytes(b"public recovery fixture")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(valid_config().to_mapping()))
    run_dir = tmp_path / "run"

    assert (
        main(
            [
                "run",
                "--source",
                str(source),
                "--config",
                str(config),
                "--run-dir",
                str(run_dir),
            ]
        )
        == 0
    )
    built = json.loads(capsys.readouterr().out)
    assert built["loss_assessment"]["lost_uncheckpointed_items"] == 3
    assert built["timing"]["recovery_time_ms"] > 0
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "checkpoint.json",
        "progress.json",
        "report.json",
        "trace.json",
    ]

    assert main(["verify", "--run-dir", str(run_dir)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified == built
