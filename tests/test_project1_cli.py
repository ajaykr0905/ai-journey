import json

from test_project1_operator import prepare_inputs

from ai_journey.project1_cli import main, status_report


def test_cli_initializes_runs_and_reports_durable_status(tmp_path, capsys):
    source, spec, contract = prepare_inputs(tmp_path)
    run_dir = tmp_path / "run"

    assert (
        main(
            [
                "init",
                "--source",
                str(source),
                "--spec",
                str(spec),
                "--contract",
                str(contract),
                "--run-dir",
                str(run_dir),
                "--recorded-at-utc",
                "2026-11-07T07:30:00+00:00",
            ]
        )
        == 0
    )
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["run_id"].startswith("day-64-project1-")
    assert status_report(run_dir)["state"]["status"] == "running"


def test_cli_executes_statuses_and_verifies_complete_run(tmp_path, capsys):
    source, spec, contract = prepare_inputs(tmp_path)
    run_dir = tmp_path / "run"
    common = ["--run-dir", str(run_dir)]
    main(
        [
            "init",
            "--source",
            str(source),
            "--spec",
            str(spec),
            "--contract",
            str(contract),
            *common,
            "--recorded-at-utc",
            "2026-11-07T07:30:00+00:00",
        ]
    )
    capsys.readouterr()
    assert (
        main(
            [
                "run",
                "--source",
                str(source),
                "--spec",
                str(spec),
                *common,
                "--recorded-at-utc",
                "2026-11-07T07:31:00+00:00",
            ]
        )
        == 0
    )
    bundle = json.loads(capsys.readouterr().out)
    assert bundle["schema_version"] == 1
    assert main(["status", *common]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["state"]["status"] == "completed"
    assert status["metrics"]["steps"] == 2
    assert main(["verify", *common]) == 0
    assert json.loads(capsys.readouterr().out) == bundle
