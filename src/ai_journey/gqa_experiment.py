"""Controlled full-MHA versus explicitly group-mean-initialized dense GQA."""

from hashlib import sha256
import json
import math

import torch

from .decoder_variant_comparison import _describe, decoder_comparison_digest
from .decoder_variant_data import (
    prepare_decoder_documents,
    document_training_schedule,
    document_training_batch,
    document_schedule_fingerprint,
)
from .decoder_variant_optimizer import decoder_optimizer_groups
from .decoder_variant_runtime import isolated_decoder_runtime
from .gqa_resources import gqa_parameter_inventory, require_gqa_execution_budget
from .gqa_training import build_gqa_optimizer, train_gqa_step
from .moe_checkpoint import _digest
from .moe_evaluation import evaluate_moe_documents
from .moe_experiment import build_matched_moe_experiment_model
from .gqa_experiment_plan import require_gqa_experiment_budget
from .training_models import build_training_model, training_model_fingerprint

GQA_EXPERIMENT_SCHEMA = "ai-journey-controlled-gqa-documents-v1"
GQA_INITIALIZATION = "exact common tensors/Q/output projection; contiguous MHA K/V head-group mean weights and biases; not full-QKV homology"
GQA_SCOPE = "internal consistency, not measurement authentication; source/model replay separate; descriptive NLL only; unequal parameters/projection work; query-score heads unchanged; no GPU/speed/generalization claim"


@isolated_decoder_runtime()
def build_controlled_gqa_model(config, seed):
    """Map one reproducible full-MHA reference to the declared compact KV heads."""
    reference = build_matched_moe_experiment_model(config.decoder, seed)
    with torch.random.fork_rng(devices=[]):
        model = build_training_model(config, max_parameters=2_000_000)
    state, shared = {}, {}
    width, heads, dim = (
        config.gpt2.embedding_dim,
        config.gpt2.head_count,
        config.gpt2.head_dim,
    )
    for name, tensor in reference.state_dict().items():
        value = tensor.detach()
        if ".attn.c_attn." in name:
            q, k, v = value.split(width, 0)
            suffix = tuple(value.shape[1:])

            def group(part):
                return (
                    part.reshape(
                        config.kv_heads, heads // config.kv_heads, dim, *suffix
                    )
                    .mean(1)
                    .reshape(config.kv_heads * dim, *suffix)
                )

            state[f"decoder.{name}"] = torch.cat((q, group(k), group(v)), 0)
            shared[name] = q
        else:
            state[f"decoder.{name}"] = value
            shared[name] = value
    model.load_state_dict(state, strict=True)
    return model, {
        "reference_mha_sha256": training_model_fingerprint(reference),
        "exact_shared_sha256": _digest(shared),
        "policy": GQA_INITIALIZATION,
    }


def summarize_gqa_trials(plan, trials):
    measurements = {
        heads: [trial["arms"][index]["heldout"]["mean_nll"] for trial in trials]
        for index, heads in enumerate(plan.kv_heads)
    }
    baseline = measurements[plan.kv_heads[0]]
    return {
        "heldout_nll": {
            str(heads): _describe(values) for heads, values in measurements.items()
        },
        "paired_deltas": [
            {
                "kv_heads": heads,
                "per_seed": [
                    {"seed": seed, "delta_nll": candidate - control}
                    for seed, candidate, control in zip(
                        plan.seeds, measurements[heads], baseline, strict=True
                    )
                ],
                "delta_nll": _describe(
                    [
                        candidate - control
                        for candidate, control in zip(
                            measurements[heads], baseline, strict=True
                        )
                    ]
                ),
            }
            for heads in plan.kv_heads[1:]
        ],
    }


@isolated_decoder_runtime()
def run_gqa_comparison(
    plan, train_bytes, heldout_bytes, *, _completed_arm=None, _cache_prefix_lengths=()
):
    documents = prepare_decoder_documents(
        train_bytes, heldout_bytes, context_length=plan.decoder.gpt2.context_length
    )
    evidence = documents.evidence()
    evidence.update(
        train_lengths=[len(doc.content) for doc in documents.train],
        heldout_lengths=[len(doc.content) for doc in documents.heldout],
    )
    if evidence["heldout_target_tokens"] > 8192:
        raise ValueError("GQA comparison exceeds held-out target coverage")
    windows = sum(
        math.ceil((len(doc.content) - 1) / documents.context_length)
        for doc in documents.heldout
    )
    budget = require_gqa_experiment_budget(
        plan, evaluation_windows=windows, cache_prefix_lengths=_cache_prefix_lengths
    )
    trials = []
    for seed in plan.seeds:
        schedule = document_training_schedule(
            documents, seed=seed, steps=plan.steps, batch_size=plan.batch_size
        )
        arms = []
        for config in plan.arms:
            model, initialization = build_controlled_gqa_model(config, seed)
            optimizer = build_gqa_optimizer(
                model, learning_rate=plan.learning_rate, config=plan.optimizer
            )
            initial_identity = training_model_fingerprint(model)
            initial = evaluate_moe_documents(
                model, documents.heldout, batch_size=plan.evaluation_batch_size
            )
            trace = []
            for step, rows in enumerate(schedule, 1):
                inputs, targets = document_training_batch(documents, rows)
                trace.append(
                    {
                        "step": step,
                        **train_gqa_step(
                            model,
                            optimizer,
                            inputs,
                            targets,
                            balance_weight=0.0,
                            z_weight=0.0,
                            max_gradient_norm=plan.max_gradient_norm,
                        ),
                    }
                )
            measured = {
                "kv_heads": config.kv_heads,
                "architecture_sha256": config.fingerprint(),
                "parameter_count": model.parameter_count,
                "parameter_inventory": gqa_parameter_inventory(config),
                "training_execution": require_gqa_execution_budget(
                    config,
                    batch_size=plan.batch_size,
                    sequence_length=documents.context_length,
                ),
                "initialization": initialization,
                "initial_model_sha256": initial_identity,
                "final_model_sha256": training_model_fingerprint(model),
                "initial_heldout": initial,
                "heldout": evaluate_moe_documents(
                    model, documents.heldout, batch_size=plan.evaluation_batch_size
                ),
                "optimizer_groups": decoder_optimizer_groups(model),
                "trace": trace,
            }
            measured = json.loads(json.dumps(measured, allow_nan=False))
            arms.append(measured)
            if _completed_arm is not None:
                _completed_arm(seed, model, measured, documents)
        trials.append(
            {
                "seed": seed,
                "batch_schedule_sha256": document_schedule_fingerprint(schedule),
                "arms": arms,
            }
        )
    report = {
        "schema": GQA_EXPERIMENT_SCHEMA,
        "plan": plan.to_mapping(),
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch_version": str(torch.__version__),
        },
        "train_sha256": sha256(train_bytes).hexdigest(),
        "heldout_sha256": sha256(heldout_bytes).hexdigest(),
        "documents": evidence,
        "evaluation_windows": windows,
        "resource_admission": require_gqa_experiment_budget(
            plan, evaluation_windows=windows
        ),
        "trials": trials,
        "summary": summarize_gqa_trials(plan, trials),
        "verification_scope": GQA_SCOPE,
    }
    # The comparison contract is stable with/without a privately admitted cache
    # observer; the separate cache report retains the larger whole-job budget.
    del budget
    report["report_sha256"] = decoder_comparison_digest(report)
    return report
