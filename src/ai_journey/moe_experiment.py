"""Matched document schedules and inspectable dense/MoE optimizer trials."""

from hashlib import sha256
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .decoder_variant_data import (
    document_schedule_fingerprint,
    document_training_batch,
    document_training_schedule,
    prepare_decoder_documents,
)
from .decoder_variant_optimizer import build_decoder_optimizer, decoder_optimizer_groups
from .decoder_variant_runtime import isolated_decoder_runtime
from .decoder_variant_training import train_decoder_step
from .moe_config import MoEConfig
from .moe_evaluation import evaluate_moe_documents
from .moe_evaluation_drift import compare_moe_route_distributions
from .moe_experiment_plan import MoEExperimentPlan, require_moe_experiment_work
from .moe_training import train_moe_step
from .training_models import build_training_model, training_model_fingerprint


def _json_safe(value):
    return json.loads(json.dumps(value, allow_nan=False))


@isolated_decoder_runtime()
def build_matched_moe_experiment_model(config, seed):
    """Initialize homologous named modules equally despite extra expert modules."""

    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("matched initialization requires a bounded seed")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = build_training_model(config, max_parameters=2_000_000)
        for name, module in model.named_modules():
            if name == "lm_head" or not isinstance(
                module, (nn.Embedding, nn.Linear, nn.LayerNorm)
            ):
                continue
            digest = sha256(seed.to_bytes(8, "big") + b"\0" + name.encode()).digest()
            torch.random.default_generator.manual_seed(
                int.from_bytes(digest[:8], "big") % (2**63)
            )
            module.reset_parameters()
    return model


def _shared_fingerprint(model):
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if ".mlp." in name:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _train_probe(model, inputs, targets):
    model.eval()
    with torch.no_grad():
        if isinstance(model.config, MoEConfig):
            result = model.forward_training(inputs)
            logits, routing = result.logits, _json_safe(result.routing)
        else:
            logits, routing = model(inputs), []
        value = float(
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        )
    if not math.isfinite(value):
        raise FloatingPointError("fixed training probe produced non-finite loss")
    return value, routing


@isolated_decoder_runtime()
def run_moe_document_trial(
    plan, train_bytes, heldout_bytes, *, seed, _completed_arm=None
):
    """Run all declared treatments with matched exposure and explicit objectives."""

    if (
        not isinstance(plan, MoEExperimentPlan)
        or type(seed) is not int
        or seed not in plan.seeds
    ):
        raise ValueError("MoE trial requires a validated plan and declared seed")
    if _completed_arm is not None and not callable(_completed_arm):
        raise TypeError("completed-arm observer must be callable")
    base = plan.arms[0].config.gpt2
    documents = prepare_decoder_documents(
        train_bytes, heldout_bytes, context_length=base.context_length
    )
    evidence = documents.evidence()
    if evidence["heldout_target_tokens"] > 8192:
        raise ValueError("MoE trial exceeds held-out target coverage")
    windows = sum(
        math.ceil((len(doc.content) - 1) / base.context_length)
        for doc in documents.heldout
    )
    budget = require_moe_experiment_work(plan, evaluation_windows=windows)
    schedule = document_training_schedule(
        documents, seed=seed, steps=plan.steps, batch_size=plan.batch_size
    )
    probe_inputs, probe_targets = document_training_batch(documents, schedule[0])
    arms = []
    for arm in plan.arms:
        model = build_matched_moe_experiment_model(arm.config, seed)
        optimizer = build_decoder_optimizer(
            model, learning_rate=plan.learning_rate, config=plan.optimizer
        )
        initial_identity = training_model_fingerprint(model)
        shared = _shared_fingerprint(model)
        initial_train, _ = _train_probe(model, probe_inputs, probe_targets)
        initial_heldout = evaluate_moe_documents(
            model, documents.heldout, batch_size=plan.evaluation_batch_size
        )
        trace = []
        for step, rows in enumerate(schedule, 1):
            inputs, targets = document_training_batch(documents, rows)
            if isinstance(arm.config, MoEConfig):
                measured = train_moe_step(
                    model,
                    optimizer,
                    inputs,
                    targets,
                    balance_weight=arm.balance_weight,
                    z_weight=arm.z_weight,
                    max_gradient_norm=plan.max_gradient_norm,
                )
            else:
                measured = train_decoder_step(
                    model,
                    optimizer,
                    inputs,
                    targets,
                    max_gradient_norm=plan.max_gradient_norm,
                )
                measured.update(
                    ce_loss=measured.pop("loss"),
                    balance_loss=0.0,
                    router_z_loss=0.0,
                    routing=[],
                )
                measured["objective"] = measured["ce_loss"]
            trace.append({"step": step, **_json_safe(measured)})
        final_train, final_routing = _train_probe(model, probe_inputs, probe_targets)
        heldout = evaluate_moe_documents(
            model, documents.heldout, batch_size=plan.evaluation_batch_size
        )
        drift = [
            {
                "layer_index": row["layer_index"],
                **compare_moe_route_distributions(
                    row["attempted_counts"],
                    next(
                        item["attempted_counts"]
                        for item in heldout["routing_summary"]
                        if item["layer_index"] == row["layer_index"]
                    ),
                ),
            }
            for row in final_routing
        ]
        arms.append(
            {
                "name": arm.name,
                "architecture_sha256": arm.config.fingerprint(),
                "parameter_count": model.parameter_count,
                "initial_shared_sha256": shared,
                "initial_model_sha256": initial_identity,
                "final_model_sha256": training_model_fingerprint(model),
                "initial_train_nll": initial_train,
                "final_train_nll": final_train,
                "initial_heldout": initial_heldout,
                "heldout": heldout,
                "final_train_routing": final_routing,
                "routing_drift": drift,
                "optimizer_groups": _json_safe(decoder_optimizer_groups(model)),
                "trace": trace,
                "consumed_target_tokens": sum(row["target_tokens"] for row in trace),
            }
        )
        if _completed_arm is not None:
            # The private publication seam consumes the completed model before
            # the next arm is allocated; reports and default execution are unchanged.
            _completed_arm(arm, model, optimizer, arms[-1], schedule)
    if len({arm["initial_shared_sha256"] for arm in arms}) != 1:
        raise RuntimeError(
            "dense and MoE arms did not share homologous initial tensors"
        )
    return {
        "seed": seed,
        "plan_sha256": plan.fingerprint(),
        "train_sha256": sha256(train_bytes).hexdigest(),
        "heldout_sha256": sha256(heldout_bytes).hexdigest(),
        "documents": evidence,
        "evaluation_windows": windows,
        "resource_admission": budget,
        "batch_schedule_sha256": document_schedule_fingerprint(schedule),
        "arms": arms,
    }
