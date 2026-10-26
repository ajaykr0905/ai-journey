import copy
import json

import pytest

from ai_journey.decoder_variant_cli import main as comparison_main
from ai_journey.decoder_variant_comparison import decoder_comparison_digest
from ai_journey.decoder_variant_document_cli import verify_decoder_document_report
from ai_journey.decoder_variant_verify import main as verification_main
from test_decoder_variant_document_experiment import TRAIN, HELDOUT
from test_decoder_variant_plan import plan


def test_installed_commands_plan_run_and_replay_documents_without_mutation(tmp_path):
    plan_path = tmp_path / "plan.json"
    assert comparison_main(["plan", "--output", str(plan_path)]) == 0
    assert len(json.loads(plan_path.read_text())["variants"]) == 5
    # Use a paired-seed declared protocol for the actual run.
    plan_path.write_text(json.dumps(plan(seeds=(7, 13)).to_mapping()))
    train, heldout, report = (
        tmp_path / name for name in ("train.bin", "heldout.bin", "report.json")
    )
    train.write_bytes(TRAIN)
    heldout.write_bytes(HELDOUT)
    flags = ["--train", str(train), "--heldout", str(heldout)]
    assert (
        comparison_main(
            ["documents", "--plan", str(plan_path), "--output", str(report), *flags]
        )
        == 0
    )
    paths = (plan_path, train, heldout, report)
    before = [path.read_bytes() for path in paths]
    assert verification_main(["documents", "--report", str(report), *flags]) == 0
    assert [path.read_bytes() for path in paths] == before
    with pytest.raises(SystemExit):
        comparison_main(["documents", "--output", str(report), *flags])
    assert report.read_bytes() == before[-1]


def test_document_verifier_detects_tensor_trace_and_summary_tampering(tmp_path):
    from ai_journey.decoder_variant_comparison import run_decoder_document_comparison

    report = run_decoder_document_comparison(plan(), TRAIN, HELDOUT)
    for mutate in (
        lambda changed: changed["trials"][0]["arms"][0].update(
            final_model_fingerprint="0" * 64
        ),
        lambda changed: changed["trials"][0]["arms"][0]["trace"][0].update(loss=0.0),
        lambda changed: changed["summary"].update(
            uncertainty_policy="unsupported claim"
        ),
    ):
        changed = copy.deepcopy(report)
        mutate(changed)
        changed["report_sha256"] = decoder_comparison_digest(changed)
        with pytest.raises(ValueError, match="deterministic replay"):
            verify_decoder_document_report(changed, TRAIN, HELDOUT)
    with pytest.raises(ValueError, match="source hashes"):
        verify_decoder_document_report(report, TRAIN + b"changed", HELDOUT)
