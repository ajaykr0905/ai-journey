"""Immutable completed-seed recovery for owned CPU MoE experiment evidence."""

from hashlib import sha256
from pathlib import Path

import torch

from .decoder_variant_comparison import decoder_comparison_digest
from .decoder_variant_data import (
    document_schedule_fingerprint,
    document_training_schedule,
    prepare_decoder_documents,
)
from .decoder_variant_report_io import write_new_report
from .decoder_variant_verify import _read_report
from .moe_experiment import run_moe_document_trial
from .moe_experiment_comparison import assemble_moe_comparison
from .moe_report import canonical_moe_json, verify_moe_report, verify_moe_trial

PROGRESS_SCHEMA = "ai-journey-moe-completed-seed-v1"


def run_restarting_moe_comparison(plan, train_bytes, heldout_bytes, progress_directory):
    """Reuse complete owned seeds; interrupted seeds restart from their beginning.

    Seals detect corruption, not fabricated measurements. Run exact final source
    replay separately. No partial model, optimizer or random state is resumed.
    """

    directory = Path(progress_directory)
    if directory.is_symlink():
        raise ValueError("MoE progress directory must not be a symlink")
    documents = prepare_decoder_documents(
        train_bytes,
        heldout_bytes,
        context_length=plan.arms[0].config.gpt2.context_length,
    )
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        directory / f"seed-{index}-{seed}.json" for index, seed in enumerate(plan.seeds)
    ]
    present = [path.exists() or path.is_symlink() for path in paths]
    if any(not earlier and later for earlier, later in zip(present, present[1:])):
        raise ValueError("MoE completed seeds must form a contiguous prefix")
    trials = []
    for path, seed in zip(paths, plan.seeds, strict=True):
        if path.exists() or path.is_symlink():
            checkpoint = _read_report(path, max_bytes=4 * 1024 * 1024)
            if (
                set(checkpoint)
                != {"schema", "plan_sha256", "torch_version", "trial", "report_sha256"}
                or checkpoint["schema"] != PROGRESS_SCHEMA
                or checkpoint["plan_sha256"] != plan.fingerprint()
                or checkpoint["torch_version"] != str(torch.__version__)
                or checkpoint["report_sha256"] != decoder_comparison_digest(checkpoint)
            ):
                raise ValueError(
                    "MoE completed seed identity, runtime or digest differs"
                )
            trial = checkpoint["trial"]
            verify_moe_trial(trial, plan, seed=seed)
            schedule = document_training_schedule(
                documents, seed=seed, steps=plan.steps, batch_size=plan.batch_size
            )
            expected = {
                "train_sha256": sha256(train_bytes).hexdigest(),
                "heldout_sha256": sha256(heldout_bytes).hexdigest(),
                "documents": documents.evidence(),
                "batch_schedule_sha256": document_schedule_fingerprint(schedule),
            }
            if canonical_moe_json(
                {key: trial[key] for key in expected}
            ) != canonical_moe_json(expected):
                raise ValueError("MoE completed seed source or schedule differs")
        else:
            trial = run_moe_document_trial(plan, train_bytes, heldout_bytes, seed=seed)
            verify_moe_trial(trial, plan, seed=seed)
            checkpoint = {
                "schema": PROGRESS_SCHEMA,
                "plan_sha256": plan.fingerprint(),
                "torch_version": str(torch.__version__),
                "trial": trial,
            }
            checkpoint["report_sha256"] = decoder_comparison_digest(checkpoint)
            try:
                write_new_report(path, checkpoint)
            except FileExistsError:
                if canonical_moe_json(
                    _read_report(path, max_bytes=4 * 1024 * 1024)
                ) != canonical_moe_json(checkpoint):
                    raise ValueError(
                        "another writer published different MoE completed evidence"
                    ) from None
        trials.append(trial)
    report = assemble_moe_comparison(plan, trials)
    verify_moe_report(report)
    return report
