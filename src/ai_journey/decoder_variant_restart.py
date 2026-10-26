"""Restart orchestration from immutable completed-seed experiment artifacts."""

from hashlib import sha256
from pathlib import Path

from .decoder_variant_comparison import (
    assemble_decoder_document_report,
    decoder_comparison_digest,
)
from .decoder_variant_data import (
    document_schedule_fingerprint,
    document_training_schedule,
    prepare_decoder_documents,
)
from .decoder_variant_document_experiment import run_decoder_document_trial
from .decoder_variant_report_io import write_new_report
from .decoder_variant_training import require_finite_decoder_state
from .decoder_variant_verify import _read_report

PROGRESS_SCHEMA = "ai-journey-decoder-completed-seed-v1"


def _validate_trial(trial, plan, documents, train_hash, heldout_hash, seed):
    fields = {
        "seed",
        "plan_fingerprint",
        "train_sha256",
        "heldout_sha256",
        "documents",
        "batch_schedule_sha256",
        "arms",
    }
    if (
        type(trial) is not dict
        or set(trial) != fields
        or trial["seed"] != seed
        or trial["plan_fingerprint"] != plan.fingerprint()
        or trial["train_sha256"] != train_hash
        or trial["heldout_sha256"] != heldout_hash
        or trial["documents"] != documents.evidence()
    ):
        raise ValueError("completed seed identity or document coverage differs")
    schedule = document_training_schedule(
        documents, seed=seed, steps=plan.steps, batch_size=plan.batch_size
    )
    if trial["batch_schedule_sha256"] != document_schedule_fingerprint(schedule):
        raise ValueError("completed seed batch schedule differs")
    if (
        type(trial["arms"]) is not list
        or any(type(arm) is not dict for arm in trial["arms"])
        or [arm.get("name") for arm in trial["arms"]]
        != [name for name, _ in plan.variants]
    ):
        raise ValueError("completed seed architecture arms differ")
    require_finite_decoder_state(trial)
    targets = plan.batch_size * documents.context_length
    for arm, (_, config) in zip(trial["arms"], plan.variants, strict=True):
        trace = arm.get("trace")
        if (
            arm.get("architecture_fingerprint") != config.fingerprint()
            or type(trace) is not list
            or len(trace) != plan.steps
            or any(
                type(row) is not dict
                or row.get("step") != step
                or row.get("target_tokens") != targets
                for step, row in enumerate(trace, 1)
            )
            or arm.get("consumed_target_tokens") != targets * plan.steps
        ):
            raise ValueError("completed seed optimizer trace coverage differs")


def run_restarting_decoder_comparison(
    plan, train_bytes, heldout_bytes, progress_directory: Path
):
    """Reuse complete seeds; an interrupted seed restarts from its beginning.

    The caller must use progress from their own run. Digests detect corruption;
    the normal final-report verifier recomputes every measured value from code.
    """

    directory = Path(progress_directory)
    if directory.is_symlink():
        raise ValueError("decoder progress directory must not be a symlink")
    documents = prepare_decoder_documents(
        train_bytes,
        heldout_bytes,
        context_length=plan.variants[0][1].gpt2.context_length,
    )
    train_hash, heldout_hash = (
        sha256(train_bytes).hexdigest(),
        sha256(heldout_bytes).hexdigest(),
    )
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        directory / f"seed-{index}-{seed}.json" for index, seed in enumerate(plan.seeds)
    ]
    present = [path.exists() or path.is_symlink() for path in paths]
    if any(not earlier and later for earlier, later in zip(present, present[1:])):
        raise ValueError("completed seed progress must form a contiguous prefix")
    trials = []
    for path, seed in zip(paths, plan.seeds, strict=True):
        if path.exists() or path.is_symlink():
            checkpoint = _read_report(path, max_bytes=2 * 1024 * 1024)
            if (
                set(checkpoint)
                != {
                    "schema",
                    "plan_fingerprint",
                    "train_sha256",
                    "heldout_sha256",
                    "trial",
                    "report_sha256",
                }
                or checkpoint["schema"] != PROGRESS_SCHEMA
                or checkpoint["plan_fingerprint"] != plan.fingerprint()
                or checkpoint["train_sha256"] != train_hash
                or checkpoint["heldout_sha256"] != heldout_hash
                or checkpoint["report_sha256"] != decoder_comparison_digest(checkpoint)
            ):
                raise ValueError("completed seed progress identity or digest differs")
            trial = checkpoint["trial"]
            _validate_trial(trial, plan, documents, train_hash, heldout_hash, seed)
        else:
            trial = run_decoder_document_trial(
                plan, train_bytes, heldout_bytes, seed=seed
            )
            checkpoint = {
                "schema": PROGRESS_SCHEMA,
                "plan_fingerprint": plan.fingerprint(),
                "train_sha256": train_hash,
                "heldout_sha256": heldout_hash,
                "trial": trial,
            }
            checkpoint["report_sha256"] = decoder_comparison_digest(checkpoint)
            try:
                write_new_report(path, checkpoint)
            except FileExistsError:
                if _read_report(path, max_bytes=2 * 1024 * 1024) != checkpoint:
                    raise ValueError(
                        "another writer published different completed seed evidence"
                    ) from None
        trials.append(trial)
    return assemble_decoder_document_report(plan, trials)
