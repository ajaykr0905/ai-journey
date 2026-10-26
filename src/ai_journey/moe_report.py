"""Offline semantic verification of sealed dense/MoE experiment observations."""

from hashlib import sha256
import json
import math
import re

import torch

from .moe_config import MoEConfig
from .moe_evaluation import summarize_moe_routing_batches
from .moe_evaluation_drift import compare_moe_route_distributions
from .moe_experiment_comparison import (
    MOE_REPORT_SCHEMA,
    MOE_REPORT_SCOPE,
    summarize_moe_trials,
)
from .moe_experiment_plan import (
    moe_experiment_plan_from_mapping,
    require_moe_experiment_work,
)
from .moe_route_audit import validate_moe_routing


def canonical_moe_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("MoE evidence is not bounded finite JSON") from exc


def _equal(actual, expected, name):
    if canonical_moe_json(actual) != canonical_moe_json(expected):
        raise ValueError(f"MoE {name} differs from recomputed evidence")


def _fields(value, expected, name):
    if type(value) is not dict or set(value) != set(expected.split()):
        raise ValueError(f"MoE {name} fields differ")


def _integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("MoE evidence integer is outside its exact bounded schema")


def _float(value):
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise ValueError("MoE measurement must be finite nonnegative float data")


def _gradient_clip_transition(before, after, maximum):
    """Check both sides of the actual CPU clip_grad_norm_ norm transition.

    PyTorch scales by min(1, maximum/(norm+1e-6)), including its epsilon even
    below the nominal clipping threshold. Float32 reductions/updates need a
    relative tolerance; the tiny absolute floor admits reduction underflow for
    the protocol's at-most-two-million gradients without admitting vanished
    ordinary gradients or swallowing the minimum 1e-9 clipping control.
    """
    expected = before * min(1.0, maximum / (before + 1e-6))
    if not math.isclose(after, expected, rel_tol=2e-5, abs_tol=1e-19):
        raise ValueError("gradient clipping norm transition contradicts its control")


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("MoE evidence requires lowercase SHA-256 identities")


def _routes(rows, config, mask, *, training):
    layers = config.moe_layers if isinstance(config, MoEConfig) else ()
    if (
        type(rows) is not list
        or any(type(row) is not dict for row in rows)
        or [row.get("layer_index") for row in rows] != list(layers)
    ):
        raise ValueError("MoE evidence routing layer order differs")
    indices = [
        index
        for index, value in enumerate(value for row in mask for value in row)
        if value
    ]
    for row in rows:
        validate_moe_routing(row, expert_count=config.expert_count)
        if any(type(value) is not float for value in row["probability_sums"]):
            raise ValueError("MoE probability sums must retain measured float types")
        _equal(row["token_indices"], indices, "valid-token routing alignment")
        if len(row["accepted_per_token"]) != len(mask) or any(
            len(values) != len(mask[0]) for values in row["accepted_per_token"]
        ):
            raise ValueError("MoE routing matrix differs from the actual batch")
        capacity = (
            max(
                config.min_capacity,
                math.ceil(
                    config.capacity_factor * len(indices) * 2 / config.expert_count
                ),
            )
            if training
            else 2 * len(indices)
        )
        _equal(row["capacity"], capacity, "configured capacity")
        _equal(row["capacity_applied"], training, "capacity policy")


def _score(score, config, *, targets, windows, batch_size):
    _fields(
        score,
        "capacity_policy target_tokens nll_sum mean_nll bits_per_byte perplexity batch_schedule_sha256 batches routing_summary",
        "evaluation",
    )
    _equal(score["capacity_policy"], "dropless_eval", "evaluation policy")
    _equal(score["target_tokens"], targets, "evaluation coverage")
    _digest(score["batch_schedule_sha256"])
    batches, losses, observed_windows = score["batches"], [], 0
    if type(batches) is not list or not 1 <= len(batches) <= windows:
        raise ValueError("MoE evaluation batches are missing or unbounded")
    for index, batch in enumerate(batches):
        _fields(batch, "shape token_mask token_nll routing", "evaluation batch")
        shape, mask = batch["shape"], batch["token_mask"]
        if type(shape) is not list or len(shape) != 2:
            raise ValueError("MoE evaluation shape is invalid")
        _integer(shape[0], 1, batch_size)
        _integer(shape[1], 1, config.gpt2.context_length)
        if index < len(batches) - 1 and shape[0] != batch_size:
            raise ValueError(
                "MoE evaluation batch grouping differs from its declared size"
            )
        if type(mask) is not list or len(mask) != shape[0]:
            raise ValueError("MoE evaluation mask shape differs")
        for row in mask:
            if (
                type(row) is not list
                or len(row) != shape[1]
                or any(type(value) is not bool for value in row)
                or not any(row)
                or row != sorted(row, reverse=True)
            ):
                raise ValueError(
                    "MoE evaluation requires nonempty right-padded boolean masks"
                )
        if max(sum(row) for row in mask) != shape[1]:
            raise ValueError("MoE evaluation padding exceeds its actual batch width")
        nll = batch["token_nll"]
        if type(nll) is not list or len(nll) != sum(sum(row) for row in mask):
            raise ValueError("MoE evaluation losses differ from valid-target exposure")
        for value in nll:
            _float(value)
        losses.extend(nll)
        observed_windows += shape[0]
        _routes(batch["routing"], config, mask, training=False)
    _equal(observed_windows, windows, "evaluation windows")
    _equal(len(losses), targets, "evaluation targets")
    total, mean = math.fsum(losses), math.fsum(losses) / targets
    for name, expected in {
        "nll_sum": total,
        "mean_nll": mean,
        "bits_per_byte": mean / math.log(2),
        "perplexity": math.exp(mean) if mean < 700 else None,
    }.items():
        _equal(score[name], expected, name)
    expected = (
        summarize_moe_routing_batches(
            batches, layers=config.moe_layers, expert_count=config.expert_count
        )
        if isinstance(config, MoEConfig)
        else []
    )
    _equal(score["routing_summary"], expected, "routing summary")


def _expected_optimizer_names(config):
    """Enumerate the declared module schema, not submitted names or live tensors.

    Ordering matches named_parameters and the fixed matrix/vector AdamW builder.
    The tied lm_head alias is absent; nonpersistent rotary buffers are not weights.
    An independent actual-model oracle tests every supported architecture choice.
    """

    decay, no_decay = ["wte.weight"], []
    if config.position_kind == "learned":
        decay.append("wpe.weight")

    def norm(prefix):
        no_decay.append(f"{prefix}.weight")
        if config.norm_kind == "layernorm":
            no_decay.append(f"{prefix}.bias")

    def mlp(prefix):
        if config.mlp_kind == "swiglu":
            decay.extend(
                f"{prefix}.0.{name}_proj.weight" for name in ("gate", "up", "down")
            )
        else:
            for projection in ("c_fc", "c_proj"):
                decay.append(f"{prefix}.{projection}.weight")
                no_decay.append(f"{prefix}.{projection}.bias")

    for index in range(config.gpt2.layer_count):
        prefix = f"h.{index}"
        norm(f"{prefix}.ln_1")
        norm(f"{prefix}.ln_2")
        for projection in ("c_attn", "c_proj"):
            decay.append(f"{prefix}.attn.{projection}.weight")
            no_decay.append(f"{prefix}.attn.{projection}.bias")
        if isinstance(config, MoEConfig) and index in config.moe_layers:
            decay.append(f"{prefix}.mlp.router.projection.weight")
            for expert in range(config.expert_count):
                mlp(f"{prefix}.mlp.experts.{expert}")
        else:
            mlp(f"{prefix}.mlp")
    norm("ln_f")
    return {"decay_names": decay, "no_decay_names": no_decay}


def _optimizer_groups(groups, config, parameters):
    _fields(
        groups, "decay_names no_decay_names trainable_parameters", "optimizer groups"
    )
    names = []
    for key in ("decay_names", "no_decay_names"):
        values = groups[key]
        if (
            type(values) is not list
            or not 1 <= len(values) <= 1024
            or any(
                type(name) is not str
                or re.fullmatch(r"[A-Za-z0-9_.]{1,200}", name) is None
                for name in values
            )
        ):
            raise ValueError("MoE optimizer group names are invalid")
        names.extend(values)
    if (
        len(set(names)) != len(names)
        or "lm_head.weight" in names
        or "wte.weight" not in groups["decay_names"]
    ):
        raise ValueError("MoE optimizer groups duplicate or omit tied parameters")
    for key, expected in _expected_optimizer_names(config).items():
        _equal(groups[key], expected, f"optimizer {key} architecture binding")
    _equal(groups["trainable_parameters"], parameters, "optimizer parameter count")


def _arm(arm, declared, plan, trial, parameters):
    _fields(
        arm,
        "name architecture_sha256 parameter_count initial_shared_sha256 initial_model_sha256 final_model_sha256 initial_train_nll final_train_nll initial_heldout heldout final_train_routing routing_drift optimizer_groups trace consumed_target_tokens",
        "arm",
    )
    config = declared.config
    _equal(arm["name"], declared.name, "arm identity")
    _equal(arm["architecture_sha256"], config.fingerprint(), "architecture")
    _equal(arm["parameter_count"], parameters, "stored parameters")
    for name in ("initial_shared_sha256", "initial_model_sha256", "final_model_sha256"):
        _digest(arm[name])
    for name in ("initial_train_nll", "final_train_nll"):
        _float(arm[name])
    for name in ("initial_heldout", "heldout"):
        _score(
            arm[name],
            config,
            targets=trial["documents"]["heldout_target_tokens"],
            windows=trial["evaluation_windows"],
            batch_size=plan.evaluation_batch_size,
        )
    _optimizer_groups(arm["optimizer_groups"], config, parameters)
    trace = arm["trace"]
    if type(trace) is not list or len(trace) != plan.steps:
        raise ValueError("MoE trace omits a declared optimizer update")
    mask = [[True] * config.gpt2.context_length for _ in range(plan.batch_size)]
    for step, row in enumerate(trace, 1):
        _fields(
            row,
            "step ce_loss balance_loss router_z_loss objective gradient_norm_before_clip gradient_norm_after_clip clipped target_tokens routing",
            "optimizer update",
        )
        _equal(row["step"], step, "update order")
        _equal(
            row["target_tokens"],
            plan.batch_size * config.gpt2.context_length,
            "update exposure",
        )
        for name in (
            "ce_loss",
            "balance_loss",
            "router_z_loss",
            "objective",
            "gradient_norm_before_clip",
            "gradient_norm_after_clip",
        ):
            _float(row[name])
        before, after = (
            row["gradient_norm_before_clip"],
            row["gradient_norm_after_clip"],
        )
        _gradient_clip_transition(before, after, plan.max_gradient_norm)
        _equal(
            row["clipped"],
            before > plan.max_gradient_norm,
            "gradient clipping decision",
        )
        if isinstance(config, MoEConfig):
            expected = float(
                torch.tensor(row["ce_loss"], dtype=torch.float32, device="cpu")
                + declared.balance_weight
                * torch.tensor(row["balance_loss"], dtype=torch.float32, device="cpu")
                + declared.z_weight
                * torch.tensor(row["router_z_loss"], dtype=torch.float64, device="cpu")
            )
        else:
            _equal(
                [row["balance_loss"], row["router_z_loss"]],
                [0.0, 0.0],
                "inactive dense objectives",
            )
            expected = row["ce_loss"]
        _equal(row["objective"], expected, "decomposed objective")
        _routes(row["routing"], config, mask, training=True)
    _equal(
        arm["consumed_target_tokens"],
        plan.steps * plan.batch_size * config.gpt2.context_length,
        "consumed targets",
    )
    _routes(arm["final_train_routing"], config, mask, training=False)
    drift = [
        {
            "layer_index": row["layer_index"],
            **compare_moe_route_distributions(
                row["attempted_counts"],
                next(
                    item["attempted_counts"]
                    for item in arm["heldout"]["routing_summary"]
                    if item["layer_index"] == row["layer_index"]
                ),
            ),
        }
        for row in arm["final_train_routing"]
    ]
    _equal(arm["routing_drift"], drift, "train/held-out route drift")


def verify_moe_report(report):
    """Recompute internal evidence without constructing or training a model."""

    _fields(
        report,
        "schema plan runtime trials summary verification_scope report_sha256",
        "report",
    )
    _equal(report["schema"], MOE_REPORT_SCHEMA, "report schema")
    _equal(report["verification_scope"], MOE_REPORT_SCOPE, "verification scope")
    _digest(report["report_sha256"])
    digest = sha256(
        canonical_moe_json(
            {key: value for key, value in report.items() if key != "report_sha256"}
        ).encode()
    ).hexdigest()
    if digest != report["report_sha256"]:
        raise ValueError("MoE report digest differs from submitted canonical bytes")
    plan = moe_experiment_plan_from_mapping(report["plan"])
    _equal(report["plan"], plan.to_mapping(), "canonical plan types")
    runtime = report["runtime"]
    _fields(runtime, "device dtype threads torch_version evaluation_policy", "runtime")
    if (
        type(runtime["torch_version"]) is not str
        or not 1 <= len(runtime["torch_version"]) <= 64
    ):
        raise ValueError("MoE recorded runtime version is invalid")
    _equal(
        runtime,
        {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch_version": runtime["torch_version"],
            "evaluation_policy": "dropless_eval",
        },
        "runtime protocol",
    )
    trials = report["trials"]
    if type(trials) is not list or len(trials) != len(plan.seeds):
        raise ValueError("MoE report omits declared seed trials")
    identities = []
    for trial, seed in zip(trials, plan.seeds, strict=True):
        identities.append(verify_moe_trial(trial, plan, seed=seed))
    for identity in identities[1:]:
        _equal(identity, identities[0], "paired source identity")
    _equal(report["summary"], summarize_moe_trials(plan, trials), "comparison summary")
    return plan


def verify_moe_trial(trial, plan, *, seed):
    """Validate one complete declared seed, shared by final reports and restart."""

    if type(seed) is not int or seed not in plan.seeds:
        raise ValueError("MoE completed trial requires a declared seed")
    _fields(
        trial,
        "seed plan_sha256 train_sha256 heldout_sha256 documents evaluation_windows resource_admission batch_schedule_sha256 arms",
        "trial",
    )
    _equal(trial["seed"], seed, "trial seed")
    _equal(trial["plan_sha256"], plan.fingerprint(), "trial plan")
    for key in ("train_sha256", "heldout_sha256", "batch_schedule_sha256"):
        _digest(trial[key])
    if trial["train_sha256"] == trial["heldout_sha256"]:
        raise ValueError("MoE training and held-out sources cannot be identical")
    documents = trial["documents"]
    _fields(
        documents,
        "train_documents heldout_documents train_window_count heldout_target_tokens",
        "document exposure",
    )
    hashes = []
    for key in ("train_documents", "heldout_documents"):
        values = documents[key]
        if type(values) is not list or not 1 <= len(values) <= 2048:
            raise ValueError("MoE document identities are missing or unbounded")
        for value in values:
            _digest(value)
        hashes.extend(values)
    if len(set(hashes)) != len(hashes) or len(hashes) > 2048:
        raise ValueError("MoE document evidence contains duplicates or leakage")
    _integer(documents["train_window_count"], 1, 1024 * 1024)
    _integer(documents["heldout_target_tokens"], 1, 8192)
    _integer(
        trial["evaluation_windows"],
        max(
            len(documents["heldout_documents"]),
            math.ceil(
                documents["heldout_target_tokens"]
                / plan.arms[0].config.gpt2.context_length
            ),
        ),
        documents["heldout_target_tokens"],
    )
    budget = require_moe_experiment_work(
        plan, evaluation_windows=trial["evaluation_windows"]
    )
    _equal(trial["resource_admission"], budget, "whole-job resource admission")
    if type(trial["arms"]) is not list or len(trial["arms"]) != len(plan.arms):
        raise ValueError("MoE report omits declared arms")
    for index, (arm, declared) in enumerate(zip(trial["arms"], plan.arms, strict=True)):
        _arm(arm, declared, plan, trial, budget["model_parameters"][index])
    if len({arm["initial_shared_sha256"] for arm in trial["arms"]}) != 1:
        raise ValueError("MoE comparison did not share homologous initial tensors")
    return {
        "train": trial["train_sha256"],
        "heldout": trial["heldout_sha256"],
        "documents": documents,
    }
