"""Paired decoder trials and descriptive seed-level architecture comparisons."""

from hashlib import sha256
import json
import math

import torch

from .decoder_variant_document_experiment import run_decoder_document_trial
from .decoder_variant_plan import DecoderExperimentPlan

DOCUMENT_REPORT_SCHEMA = "ai-journey-decoder-documents-v1"
UNCERTAINTY_POLICY = (
    "Deltas are candidate minus baseline within each seed; negative favors candidate. "
    "Seed spread is descriptive and does not establish population confidence or "
    "generalization to other corpora, hardware or model scales."
)


def _describe(values):
    count = len(values)
    mean = math.fsum(values) / count
    deviation = (
        math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (count - 1))
        if count > 1
        else None
    )
    return {
        "count": count,
        "mean": mean,
        "sample_stddev": deviation,
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_decoder_trials(plan: DecoderExperimentPlan, trials: list[dict]) -> dict:
    """Pair seed measurements before computing cross-seed spread."""

    if type(trials) is not list or [trial["seed"] for trial in trials] != list(
        plan.seeds
    ):
        raise ValueError("summary requires every declared seed once in plan order")
    names = [name for name, _ in plan.variants]
    measurements = {name: [] for name in names}
    coverage = None
    for trial in trials:
        if [arm["name"] for arm in trial["arms"]] != names:
            raise ValueError("summary requires every declared architecture arm once")
        if trial["plan_fingerprint"] != plan.fingerprint():
            raise ValueError("trial does not match the experiment plan")
        if coverage is None:
            coverage = trial["documents"]
        elif trial["documents"] != coverage:
            raise ValueError("paired seeds must share identical document coverage")
        for arm in trial["arms"]:
            score = arm["heldout"]
            if (
                type(score["target_tokens"]) is not int
                or score["target_tokens"] != coverage["heldout_target_tokens"]
            ):
                raise ValueError("paired measurements have different target coverage")
            loss = score["mean_nll"]
            if (
                isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(loss)
                or loss < 0
            ):
                raise ValueError("paired measurement must be a finite nonnegative NLL")
            measurements[arm["name"]].append(loss)
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
        "uncertainty_policy": UNCERTAINTY_POLICY,
    }


def decoder_comparison_digest(report: dict) -> str:
    content = {key: value for key, value in report.items() if key != "report_sha256"}
    return sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def run_decoder_document_comparison(
    plan: DecoderExperimentPlan, train_bytes: bytes, heldout_bytes: bytes
) -> dict:
    trials = [
        run_decoder_document_trial(plan, train_bytes, heldout_bytes, seed=seed)
        for seed in plan.seeds
    ]
    return assemble_decoder_document_report(plan, trials)


def assemble_decoder_document_report(
    plan: DecoderExperimentPlan, trials: list[dict]
) -> dict:
    """Assemble identical evidence for fresh and completed-seed restart runs."""

    report = {
        "schema": DOCUMENT_REPORT_SCHEMA,
        "plan": plan.to_mapping(),
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch_version": str(torch.__version__),
        },
        "trials": trials,
        "summary": summarize_decoder_trials(plan, trials),
    }
    report["report_sha256"] = decoder_comparison_digest(report)
    return report
