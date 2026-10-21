import json
from pathlib import Path
import pytest
from ai_journey.ddp_cli import main
from ai_journey.ddp_evidence import read_evidence


def test_cli_trains_verifies_and_prints_measured_rank_evidence(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "config/gpt2-training.json").read_text())
    payload["training"].update(
        optimizer_steps=2, sequence_length=4, micro_batch_size=1, accumulation_steps=2
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(payload))
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "train",
                "--source",
                str(root / "data/day-19-demo-names.txt"),
                "--spec",
                str(spec),
                "--output",
                str(report),
                "--checkpoint",
                str(tmp_path / "state.pt"),
                "--workdir",
                str(tmp_path / "ranks"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == read_evidence(report)
    assert main(["verify", "--report", str(report)]) == 0
    assert json.loads(capsys.readouterr().out)["world_size"] == 2


def test_cli_invalid_report_exits_with_an_actionable_error(tmp_path):
    with pytest.raises(SystemExit) as result:
        main(["verify", "--report", str(tmp_path / "missing.json")])
    assert result.value.code == 2
