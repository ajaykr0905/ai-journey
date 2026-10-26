from pathlib import Path
import shutil

import pytest

from ai_journey import ddp_cli
from ai_journey.ddp_launch import launch_training

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", ["train", "recover", "compare"])
@pytest.mark.parametrize("collision", ["source", "spec", "hardlink-source", "workdir"])
def test_cli_reports_cannot_overwrite_inputs_or_owned_worker_artifacts(
    tmp_path, monkeypatch, command, collision
):
    source = tmp_path / "source.txt"
    spec = tmp_path / "spec.json"
    shutil.copyfile(ROOT / "data/day-19-demo-names.txt", source)
    shutil.copyfile(ROOT / "config/ddp-lab.json", spec)
    output = tmp_path / "report.json"
    workdir = tmp_path / "workers"
    if collision == "source":
        output = source
    elif collision == "spec":
        output = spec
    elif collision == "hardlink-source":
        output.hardlink_to(source)
    else:
        output = workdir / "rank-0.log"
    before = (source.read_bytes(), spec.read_bytes())

    def unexpected(*args, **kwargs):
        raise AssertionError("artifact admission must precede worker creation")

    for name in ("launch_training", "certify_recovery", "certify_batch_equivalence"):
        monkeypatch.setattr(ddp_cli, name, unexpected)
    argv = [
        command,
        "--source",
        str(source),
        "--spec",
        str(spec),
        "--output",
        str(output),
        "--workdir",
        str(workdir),
    ]
    if command == "train":
        argv += ["--checkpoint", str(tmp_path / "state.pt")]
    with pytest.raises(SystemExit) as error:
        ddp_cli.main(argv)
    assert error.value.code == 2
    assert (source.read_bytes(), spec.read_bytes()) == before
    assert not workdir.exists()


def test_train_report_cannot_replace_its_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "state.pt"
    checkpoint.write_bytes(b"preserve restart state")

    def unexpected(*args, **kwargs):
        raise AssertionError("must reject report/checkpoint alias before training")

    monkeypatch.setattr(ddp_cli, "launch_training", unexpected)
    with pytest.raises(SystemExit) as error:
        ddp_cli.main(
            [
                "train",
                "--source",
                str(ROOT / "data/day-19-demo-names.txt"),
                "--spec",
                str(ROOT / "config/ddp-lab.json"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(checkpoint),
                "--workdir",
                str(tmp_path / "workers"),
            ]
        )
    assert error.value.code == 2
    assert checkpoint.read_bytes() == b"preserve restart state"


def test_worker_nonregular_outputs_reject_before_opening_or_launching(
    tmp_path, monkeypatch
):
    import os

    os.mkfifo(tmp_path / "rank-0.log")

    def unexpected(*args, **kwargs):
        raise AssertionError("must reject FIFO before a blocking output open")

    monkeypatch.setattr("builtins.open", unexpected)
    with pytest.raises(ValueError, match="worker outputs"):
        launch_training(
            source=ROOT / "data/day-19-demo-names.txt",
            spec=ROOT / "config/ddp-lab.json",
            checkpoint=tmp_path / "state.pt",
            output_dir=tmp_path,
            world_size=1,
            steps=1,
        )


def test_hardlinked_worker_artifacts_cannot_overwrite_another_rank(
    tmp_path, monkeypatch
):
    log = tmp_path / "rank-0.log"
    log.write_bytes(b"keep caller diagnostic")
    (tmp_path / "rank-1.json").hardlink_to(log)

    def unexpected(*args, **kwargs):
        raise AssertionError("must reject cross-rank hard links before launching")

    monkeypatch.setattr("subprocess.Popen", unexpected)
    with pytest.raises(ValueError, match="worker outputs"):
        launch_training(
            source=ROOT / "data/day-19-demo-names.txt",
            spec=ROOT / "config/ddp-lab.json",
            checkpoint=tmp_path / "state.pt",
            output_dir=tmp_path,
            steps=1,
        )
    assert log.read_bytes() == b"keep caller diagnostic"
