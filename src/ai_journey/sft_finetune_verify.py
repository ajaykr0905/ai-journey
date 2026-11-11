"""Independent structural verification for SFT evidence and source binding."""

from __future__ import annotations

from .sft_finetune_contract import SFTFineTuneError
from .sft_finetune_dataset import FineTuneDataset
from .sft_finetune_report import validate_finetune_report


def verify_finetune_evidence(report: dict, dataset: FineTuneDataset) -> dict:
    validate_finetune_report(report)
    if not isinstance(dataset, FineTuneDataset):
        raise TypeError("dataset must be FineTuneDataset")
    evidence = report["result"]["dataset"]
    expected = {
        "sha256": dataset.sha256,
        "examples": len(dataset.examples),
        "train_examples": len(dataset.train),
        "validation_examples": len(dataset.validation),
    }
    if evidence != expected:
        raise SFTFineTuneError("report is not bound to the supplied dataset")
    return {
        "report_sha256": report["sha256"],
        "dataset_sha256": dataset.sha256,
        "verified_examples": len(dataset.examples),
    }
