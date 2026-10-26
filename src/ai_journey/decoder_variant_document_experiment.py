"""Controlled document experiments with complete tensor and update evidence."""

from hashlib import sha256
import math

import torch
from torch.nn import functional as F

from .decoder_variant_budget import require_comparison_budget
from .decoder_variant_data import (
    document_schedule_fingerprint,
    document_training_batch,
    document_training_schedule,
    prepare_decoder_documents,
)
from .decoder_variant_evaluation import evaluate_decoder_documents
from .decoder_variant_experiment import _build_matched_model
from .decoder_variant_optimizer import build_decoder_optimizer, decoder_optimizer_groups
from .decoder_variant_plan import DecoderExperimentPlan
from .decoder_variant_runtime import isolated_decoder_runtime
from .decoder_variant_training import train_decoder_step
from .transformer_lab import model_fingerprint


def _training_nll(model, inputs, targets):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        value = float(
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        )
    if not math.isfinite(value):
        raise FloatingPointError("document training diagnostic is non-finite")
    return value


@isolated_decoder_runtime()
def run_decoder_document_trial(
    plan: DecoderExperimentPlan, train_bytes: bytes, heldout_bytes: bytes, *, seed: int
) -> dict:
    """Train every declared arm on an identical document-local batch schedule."""

    if not isinstance(plan, DecoderExperimentPlan):
        raise TypeError("document trial requires a DecoderExperimentPlan")
    if type(seed) is not int or seed not in plan.seeds:
        raise ValueError("trial seed must be declared in the experiment plan")
    base = plan.variants[0][1].gpt2
    documents = prepare_decoder_documents(
        train_bytes, heldout_bytes, context_length=base.context_length
    )
    evidence = documents.evidence()
    if evidence["heldout_target_tokens"] > 8192:
        raise ValueError("document trial exceeds 8192 held-out target tokens")
    evaluation_windows = sum(
        math.ceil((len(doc.content) - 1) / base.context_length)
        for doc in documents.heldout
    )
    require_comparison_budget(
        base,
        steps=plan.steps,
        batch_size=plan.batch_size,
        evaluation_windows=2 * evaluation_windows,
        variant_count=len(plan.variants),
    )
    schedule = document_training_schedule(
        documents, seed=seed, steps=plan.steps, batch_size=plan.batch_size
    )
    initial_inputs, initial_targets = document_training_batch(documents, schedule[0])
    arms = []
    for name, config in plan.variants:
        model = _build_matched_model(config, seed)
        optimizer = build_decoder_optimizer(
            model, learning_rate=plan.learning_rate, config=plan.optimizer
        )
        groups = decoder_optimizer_groups(model)
        groups = {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in groups.items()
        }
        initial_fingerprint = model_fingerprint(model)
        initial_train = _training_nll(model, initial_inputs, initial_targets)
        initial_heldout = evaluate_decoder_documents(
            model, documents.heldout, batch_size=plan.evaluation_batch_size
        )
        trace = []
        for step, rows in enumerate(schedule, 1):
            inputs, targets = document_training_batch(documents, rows)
            trace.append(
                {
                    "step": step,
                    **train_decoder_step(
                        model,
                        optimizer,
                        inputs,
                        targets,
                        max_gradient_norm=plan.max_gradient_norm,
                    ),
                }
            )
        final_train = _training_nll(model, initial_inputs, initial_targets)
        heldout = evaluate_decoder_documents(
            model, documents.heldout, batch_size=plan.evaluation_batch_size
        )
        arms.append(
            {
                "name": name,
                "architecture_fingerprint": config.fingerprint(),
                "parameter_count": model.parameter_count,
                "optimizer_groups": groups,
                "initial_model_fingerprint": initial_fingerprint,
                "final_model_fingerprint": model_fingerprint(model),
                "initial_train_nll": initial_train,
                "final_train_nll": final_train,
                "initial_heldout": initial_heldout,
                "heldout": heldout,
                "trace": trace,
                "consumed_target_tokens": sum(row["target_tokens"] for row in trace),
            }
        )
    return {
        "seed": seed,
        "plan_fingerprint": plan.fingerprint(),
        "train_sha256": sha256(train_bytes).hexdigest(),
        "heldout_sha256": sha256(heldout_bytes).hexdigest(),
        "documents": evidence,
        "batch_schedule_sha256": document_schedule_fingerprint(schedule),
        "arms": arms,
    }
