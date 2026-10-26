"""Paired-seed descriptive comparisons with honest MoE exposure denominators."""

import math

import torch

from .decoder_variant_comparison import _describe, decoder_comparison_digest
from .moe_experiment import run_moe_document_trial
from .moe_route_audit import validate_moe_routing

MOE_REPORT_SCHEMA = "ai-journey-moe-documents-v1"
UNCERTAINTY_POLICY = (
    "Paired seed deltas are descriptive, not population confidence or evidence "
    "of generalization, compute equivalence, GPU speed or architecture superiority."
)


def training_route_pressure(arm, config):
    rows = [stats for step in arm["trace"] for stats in step["routing"]]
    totals = [
        validate_moe_routing(row, expert_count=config.expert_count) for row in rows
    ]
    attempted = sum(row["attempted_routes"] for row in totals)
    dropped = sum(row["dropped_routes"] for row in totals)
    return {
        "layer_tokens": sum(row["valid_tokens"] for row in totals),
        "attempted_routes": attempted,
        "accepted_routes": attempted - dropped,
        "dropped_routes": dropped,
        "unrouted_layer_tokens": sum(row["unrouted_tokens"] for row in totals),
        "dropped_fraction": dropped / attempted if attempted else None,
    }


def summarize_moe_trials(plan, trials):
    if type(trials) is not list or [row.get("seed") for row in trials] != list(
        plan.seeds
    ):
        raise ValueError("MoE comparison requires every declared seed in order")
    names = [arm.name for arm in plan.arms]
    coverage = trials[0]["documents"]
    measurements = {name: [] for name in names}
    pressure = {name: [] for name in names[1:]}
    for trial in trials:
        if (
            trial["plan_sha256"] != plan.fingerprint()
            or trial["documents"] != coverage
            or [arm["name"] for arm in trial["arms"]] != names
        ):
            raise ValueError(
                "paired MoE trials differ in plan, document exposure or arm order"
            )
        for arm, declared in zip(trial["arms"], plan.arms, strict=True):
            score = arm["heldout"]
            if (
                type(score["target_tokens"]) is not int
                or score["target_tokens"] != coverage["heldout_target_tokens"]
                or type(score["mean_nll"]) is not float
                or not math.isfinite(score["mean_nll"])
                or score["mean_nll"] < 0
            ):
                raise ValueError(
                    "paired MoE measurements require matching finite held-out coverage"
                )
            measurements[arm["name"]].append(score["mean_nll"])
            if declared.name != "baseline":
                pressure[declared.name].append(
                    {
                        "seed": trial["seed"],
                        **training_route_pressure(arm, declared.config),
                    }
                )
    return {
        "heldout_nll": {
            name: _describe(values) for name, values in measurements.items()
        },
        "paired_deltas": [
            {
                "baseline": "baseline",
                "candidate": name,
                "per_seed": [
                    {"seed": seed, "delta_nll": candidate - baseline}
                    for seed, candidate, baseline in zip(
                        plan.seeds,
                        measurements[name],
                        measurements["baseline"],
                        strict=True,
                    )
                ],
                "delta_nll": _describe(
                    [
                        candidate - baseline
                        for candidate, baseline in zip(
                            measurements[name], measurements["baseline"], strict=True
                        )
                    ]
                ),
            }
            for name in names[1:]
        ],
        "training_route_pressure": pressure,
        "uncertainty_policy": UNCERTAINTY_POLICY,
    }


def assemble_moe_comparison(plan, trials):
    report = {
        "schema": MOE_REPORT_SCHEMA,
        "plan": plan.to_mapping(),
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch_version": str(torch.__version__),
            "evaluation_policy": "dropless_eval",
        },
        "trials": trials,
        "summary": summarize_moe_trials(plan, trials),
    }
    report["report_sha256"] = decoder_comparison_digest(report)
    return report


def run_moe_document_comparison(plan, train_bytes, heldout_bytes):
    trials = [
        run_moe_document_trial(plan, train_bytes, heldout_bytes, seed=seed)
        for seed in plan.seeds
    ]
    return assemble_moe_comparison(plan, trials)
