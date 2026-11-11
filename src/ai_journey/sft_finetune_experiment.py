"""Compose a deterministic public-data SFT run and matched comparison."""

from __future__ import annotations

import copy
import platform
from contextlib import contextmanager

import torch

from .sft_finetune_comparison import compare_finetune_models
from .sft_finetune_contract import SFTFineTuneConfig, SFTFineTuneError
from .sft_finetune_dataset import FineTuneDataset
from .sft_finetune_model import build_finetune_model, finetune_model_evidence
from .sft_finetune_schedule import build_schedule
from .sft_finetune_training import train_finetune_schedule

LIMITATIONS = [
    "CPU float64 character-scale diagnostic; no GPU or distributed execution",
    "Eight optimizer steps cannot establish useful model quality or alignment",
    "Greedy byte generations can contain invalid UTF-8 and are diagnostic only",
    "Dataset labels come from a public snapshot and are not learner-authored evidence",
]


@contextmanager
def _single_threaded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_finetune_experiment(
    dataset: FineTuneDataset, config: SFTFineTuneConfig
) -> dict:
    if not isinstance(dataset, FineTuneDataset):
        raise TypeError("dataset must be FineTuneDataset")
    if not isinstance(config, SFTFineTuneConfig):
        raise TypeError("config must be SFTFineTuneConfig")
    if config.evaluation_examples > len(dataset.validation):
        raise SFTFineTuneError("evaluation_examples exceeds the validation partition")
    with _single_threaded_cpu():
        schedule = build_schedule(
            dataset.train,
            block_size=config.block_size,
            steps=config.steps,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        model = build_finetune_model(config)
        before = copy.deepcopy(model)
        initial = finetune_model_evidence(before)
        _, training = train_finetune_schedule(model, schedule, config)
        final = finetune_model_evidence(model)
        comparison = compare_finetune_models(before, model, dataset.validation, config)
    return {
        "config": config.as_mapping(),
        "dataset": {
            "sha256": dataset.sha256,
            "examples": len(dataset.examples),
            "train_examples": len(dataset.train),
            "validation_examples": len(dataset.validation),
        },
        "schedule_sha256": schedule.sha256,
        "initial_model": initial,
        "final_model": final,
        "training": training,
        "comparison": comparison,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "torch.float64",
            "threads": 1,
        },
        "limitations": list(LIMITATIONS),
    }
