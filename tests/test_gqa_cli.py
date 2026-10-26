import copy
import json

import pytest

from ai_journey.gqa_cli import main, default_gqa_plan
from ai_journey.gqa_experiment_public import (
    run_public_gqa_comparison,
    verify_public_gqa_report,
    replay_public_gqa_report,
)
from ai_journey.decoder_variant_comparison import decoder_comparison_digest
from test_moe_cli import snapshot
from test_moe_experiment import TRAIN, HELDOUT


def test_installed_command_functions_plan_run_verify_replay_and_no_overwrite(tmp_path):
    train, heldout, output, plan = (
        tmp_path / name for name in ("train", "heldout", "report", "plan")
    )
    train.write_bytes(TRAIN)
    heldout.write_bytes(HELDOUT)
    assert main(["plan", "--output", str(plan)]) == 0
    assert (
        main(
            [
                "run",
                "--train",
                str(train),
                "--heldout",
                str(heldout),
                "--plan",
                str(plan),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    original = output.read_bytes()
    assert main(["verify", "--report", str(output)]) == 0
    assert (
        main(
            [
                "replay",
                "--report",
                str(output),
                "--train",
                str(train),
                "--heldout",
                str(heldout),
            ]
        )
        == 0
    )
    with pytest.raises(SystemExit):
        main(["plan", "--output", str(output)])
    assert output.read_bytes() == original


def test_public_fixture_pin_license_partition_and_actual_replay(tmp_path, monkeypatch):
    path = snapshot(tmp_path, monkeypatch)
    report = run_public_gqa_comparison(
        default_gqa_plan(), path, "en-pud", train_documents=4, heldout_documents=2
    )
    verify_public_gqa_report(report)
    replay_public_gqa_report(report, path)
    changed = copy.deepcopy(report)
    changed["public_corpus"]["source"]["revision"] = "0" * 40
    changed["report_sha256"] = decoder_comparison_digest(changed)
    with pytest.raises(ValueError):
        verify_public_gqa_report(changed)
    output = tmp_path / "public.json"
    output.write_text(json.dumps(report))
    assert main(["verify", "--report", str(output)]) == 0
    assert main(["replay", "--report", str(output), "--snapshot", str(path)]) == 0
