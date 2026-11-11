import pytest

from ai_journey.sft_finetune_contract import SFTFineTuneError
from ai_journey.sft_finetune_report import build_finetune_report
from ai_journey.sft_finetune_verify import verify_finetune_evidence
from tests.test_sft_finetune_experiment import dataset
from tests.test_sft_finetune_report import result


def test_verifier_requires_exact_dataset_identity_and_counts():
    value = result()
    source = dataset()
    value["dataset"] = {
        "sha256": source.sha256,
        "examples": len(source.examples),
        "train_examples": len(source.train),
        "validation_examples": len(source.validation),
    }
    report = build_finetune_report(value)
    assert verify_finetune_evidence(report, source)["verified_examples"] == 4
    value["dataset"]["sha256"] = "b" * 64
    with pytest.raises(SFTFineTuneError, match="bound"):
        verify_finetune_evidence(build_finetune_report(value), source)
