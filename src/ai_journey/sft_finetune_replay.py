"""Byte-exact replay of a sealed SFT fine-tuning experiment."""

from __future__ import annotations

from .sft_finetune_contract import SFTFineTuneConfig, SFTFineTuneError
from .sft_finetune_dataset import FineTuneDataset
from .sft_finetune_experiment import run_finetune_experiment
from .sft_finetune_report import build_finetune_report, validate_finetune_report
from .sft_finetune_verify import verify_finetune_evidence


def replay_finetune_report(report: dict, dataset: FineTuneDataset) -> dict:
    validate_finetune_report(report)
    verify_finetune_evidence(report, dataset)
    config = SFTFineTuneConfig.from_mapping(report["result"]["config"])
    rebuilt = build_finetune_report(run_finetune_experiment(dataset, config))
    if rebuilt != report:
        raise SFTFineTuneError("replayed SFT evidence differs from the supplied report")
    return {
        "report_sha256": report["sha256"],
        "replayed_steps": config.steps,
        "replayed_examples": len(dataset.examples),
    }
