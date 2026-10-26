import copy
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from ai_journey.decoder_variant_comparison import decoder_comparison_digest
from ai_journey.moe_cli import (
    main,
    run_public_moe_comparison,
    verify_public_moe_report,
    replay_public_moe_report,
)
import ai_journey.moe_cli as cli
from ai_journey.tokenizer_sources import ENGLISH
from test_moe_experiment import TRAIN, HELDOUT
from test_moe_experiment_plan import plan


def test_cli_plan_run_internal_verification_and_actual_replay(tmp_path, capsys):
    train, heldout = tmp_path / "train", tmp_path / "heldout"
    train.write_bytes(TRAIN)
    heldout.write_bytes(HELDOUT)
    output, controls = tmp_path / "result.json", tmp_path / "plan.json"
    assert main(["plan", "--output", str(controls)]) == 0
    assert (
        main(
            [
                "run",
                "--plan",
                str(controls),
                "--train",
                str(train),
                "--heldout",
                str(heldout),
                "--output",
                str(output),
                "--progress-directory",
                str(tmp_path / "progress"),
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
    assert output.read_bytes() == original
    assert "internally_verified=" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(
            [
                "run",
                "--train",
                str(train),
                "--heldout",
                str(heldout),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert output.read_bytes() == original


def snapshot(tmp_path, monkeypatch):
    payload = b"\n".join(
        f"# sent_id = public-{i}\n# text = Authored public MoE fixture sentence {i}\n".encode()
        for i in range(12)
    )
    source = replace(
        ENGLISH, size_bytes=len(payload), sha256=sha256(payload).hexdigest()
    )
    monkeypatch.setitem(cli.PUBLIC_SOURCES, "en-pud", source)
    path = tmp_path / "source.conllu"
    path.write_bytes(payload)
    return path


def test_pinned_public_cli_replays_actual_source_partition_and_license(
    tmp_path, monkeypatch
):
    path = snapshot(tmp_path, monkeypatch)
    report = run_public_moe_comparison(
        plan(), path, "en-pud", train_documents=4, heldout_documents=2
    )
    verify_public_moe_report(report)
    replay_public_moe_report(report, path)
    assert str(path) not in json.dumps(report)
    assert report["public_corpus"]["source"]["license"] == ENGLISH.license
    output = tmp_path / "public.json"
    output.write_text(json.dumps(report))
    assert main(["verify", "--report", str(output)]) == 0
    assert main(["replay", "--report", str(output), "--snapshot", str(path)]) == 0
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity"):
        replay_public_moe_report(report, path)


@pytest.mark.parametrize("mutation", ["alias", "pin", "overlap", "count", "schema"])
def test_resealed_public_identity_corruption_is_not_accepted(
    tmp_path, monkeypatch, mutation
):
    path = snapshot(tmp_path, monkeypatch)
    report = copy.deepcopy(
        run_public_moe_comparison(
            plan(), path, "en-pud", train_documents=4, heldout_documents=2
        )
    )
    evidence = report["public_corpus"]
    if mutation == "alias":
        evidence["split_seed"] = True
    elif mutation == "pin":
        evidence["source"]["revision"] = "0" * 40
    elif mutation == "overlap":
        evidence["heldout_document_sha256"][0] = evidence["train_document_sha256"][0]
    elif mutation == "count":
        evidence["train_documents"] = 3
    else:
        report["extra"] = "silent"
    report["report_sha256"] = decoder_comparison_digest(report)
    with pytest.raises(ValueError):
        verify_public_moe_report(report)


def test_cli_rejects_duplicate_json_and_wrong_replay_source_kind(tmp_path):
    report = tmp_path / "bad.json"
    report.write_text('{"schema":1,"schema":2}')
    with pytest.raises(SystemExit) as error:
        main(["verify", "--report", str(report)])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        main(["plan", "--output", str(report)])
