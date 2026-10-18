"""Validated evidence reports for the bounded preference comparison."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .lora_experiment import ADAPTER_TARGETS
from .preference_data import (
    PreferenceDataError,
    canonical_digest,
    preference_schedule_evidence,
    preference_training_batches,
)
from .preference_experiment import (
    PREFERENCE_ARMS,
    PREFERENCE_LIMITATIONS,
    prepare_preference_context,
    preference_config_from_dict,
    run_preference_comparison,
)


UNCERTAINTY_POLICY = (
    "Sample standard deviation, standard error, and observed range describe only "
    "the paired seed-fold observations from one authored fixture. They are not "
    "confidence intervals or evidence about other tasks, annotators, or model scales."
)


def _finite(value, name, *, minimum=None, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise PreferenceDataError(f"invalid finite {name}")
    return value


def _describe(values):
    if not isinstance(values, list) or len(values) < 2:
        raise PreferenceDataError("uncertainty summary requires at least two values")
    values = [_finite(value, "summary observation") for value in values]
    mean = math.fsum(values) / len(values)
    deviation = math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )
    return {
        "count": len(values),
        "mean": mean,
        "sample_stddev": deviation,
        "standard_error": deviation / math.sqrt(len(values)),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_preference_comparison(result):
    """Aggregate only matched seed-fold observations and retain paired deltas."""
    try:
        seeds = result["config"]["seeds"]
        fold_count = result["config"]["fold_count"]
        rows = result["seeds"]
        if (
            not isinstance(seeds, list)
            or not 2 <= len(seeds) <= 4
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or type(fold_count) is not int
            or fold_count < 2
            or not isinstance(rows, list)
            or [row["seed"] for row in rows] != seeds
        ):
            raise PreferenceDataError("summary requires complete paired seed controls")
        observations = {arm: [] for arm in PREFERENCE_ARMS}
        parameters = {}
        labels = []
        for seed_row in rows:
            folds = seed_row["folds"]
            if not isinstance(folds, list) or len(folds) != fold_count:
                raise PreferenceDataError(
                    "summary requires every cross-validation fold"
                )
            for fold_index, fold in enumerate(folds):
                trials = fold["trials"]
                if not isinstance(trials, list) or [
                    trial["name"] for trial in trials
                ] != list(PREFERENCE_ARMS):
                    raise PreferenceDataError("summary requires each preference arm")
                labels.append({"seed": seed_row["seed"], "fold": fold_index})
                for trial in trials:
                    arm = trial["name"]
                    heldout = trial["heldout"]
                    observation = {
                        "preference_accuracy": _finite(
                            heldout["preference_accuracy"],
                            "preference accuracy",
                            minimum=0,
                            maximum=1,
                        ),
                        "mean_policy_margin": _finite(
                            heldout["mean_policy_margin"], "policy margin"
                        ),
                        "mean_reward_margin": _finite(
                            heldout["mean_reward_margin"], "reward margin"
                        ),
                        "mean_loss": _finite(
                            heldout["mean_loss"], "preference loss", minimum=0
                        ),
                    }
                    observations[arm].append(observation)
                    counts = trial["parameters"]
                    if (
                        not isinstance(counts, dict)
                        or set(counts) != {"total", "trainable", "frozen"}
                        or any(
                            type(value) is not int or value < 0
                            for value in counts.values()
                        )
                        or counts["total"] != counts["trainable"] + counts["frozen"]
                    ):
                        raise PreferenceDataError("invalid preference parameter counts")
                    if arm in parameters and parameters[arm] != counts:
                        raise PreferenceDataError(
                            "parameter counts differ between folds"
                        )
                    parameters[arm] = dict(counts)
        expected = len(seeds) * fold_count
        if len(labels) != expected or any(
            len(values) != expected for values in observations.values()
        ):
            raise PreferenceDataError("summary observation coverage is incomplete")
        full_trainable = parameters["full"]["trainable"]
        if full_trainable <= 0:
            raise PreferenceDataError("dense preference arm must be trainable")
        arms = []
        for arm in PREFERENCE_ARMS:
            arms.append(
                {
                    "name": arm,
                    "parameters": parameters[arm],
                    "trainable_fraction_of_full": parameters[arm]["trainable"]
                    / full_trainable,
                    "preference_accuracy": _describe(
                        [row["preference_accuracy"] for row in observations[arm]]
                    ),
                    "mean_policy_margin": _describe(
                        [row["mean_policy_margin"] for row in observations[arm]]
                    ),
                    "mean_reward_margin": _describe(
                        [row["mean_reward_margin"] for row in observations[arm]]
                    ),
                    "mean_loss": _describe(
                        [row["mean_loss"] for row in observations[arm]]
                    ),
                }
            )
        paired = []
        for reference_index, reference in enumerate(PREFERENCE_ARMS):
            for candidate in PREFERENCE_ARMS[reference_index + 1 :]:
                accuracy_deltas = [
                    candidate_row["preference_accuracy"]
                    - reference_row["preference_accuracy"]
                    for candidate_row, reference_row in zip(
                        observations[candidate], observations[reference], strict=True
                    )
                ]
                reward_deltas = [
                    candidate_row["mean_reward_margin"]
                    - reference_row["mean_reward_margin"]
                    for candidate_row, reference_row in zip(
                        observations[candidate], observations[reference], strict=True
                    )
                ]
                loss_deltas = [
                    candidate_row["mean_loss"] - reference_row["mean_loss"]
                    for candidate_row, reference_row in zip(
                        observations[candidate], observations[reference], strict=True
                    )
                ]
                paired.append(
                    {
                        "reference": reference,
                        "candidate": candidate,
                        "observations": [
                            {
                                **label,
                                "delta_accuracy": accuracy,
                                "delta_reward_margin": reward,
                                "delta_loss": loss,
                            }
                            for label, accuracy, reward, loss in zip(
                                labels,
                                accuracy_deltas,
                                reward_deltas,
                                loss_deltas,
                                strict=True,
                            )
                        ],
                        "delta_accuracy": _describe(accuracy_deltas),
                        "delta_reward_margin": _describe(reward_deltas),
                        "delta_loss": _describe(loss_deltas),
                        "candidate_accuracy_wins": sum(
                            value > 0 for value in accuracy_deltas
                        ),
                        "accuracy_ties": sum(value == 0 for value in accuracy_deltas),
                    }
                )
        frozen_accuracy = arms[0]["preference_accuracy"]["mean"]
        negative_findings = []
        for arm in arms[1:]:
            if arm["preference_accuracy"]["mean"] <= frozen_accuracy:
                negative_findings.append(
                    f"{arm['name']} did not exceed the frozen arm's mean held-out "
                    "preference accuracy."
                )
            if arm["mean_reward_margin"]["mean"] <= 0:
                negative_findings.append(
                    f"{arm['name']} had a non-positive mean held-out reward margin."
                )
        rank1 = next(arm for arm in arms if arm["name"] == "lora-rank-1")
        rank2 = next(arm for arm in arms if arm["name"] == "lora-rank-2")
        if rank2["preference_accuracy"]["mean"] <= rank1["preference_accuracy"]["mean"]:
            negative_findings.append(
                "lora-rank-2 did not exceed lora-rank-1 mean held-out preference "
                "accuracy."
            )
        return {
            "observations": expected,
            "arms": arms,
            "paired": paired,
            "negative_findings": negative_findings,
            "uncertainty_policy": UNCERTAINTY_POLICY,
        }
    except (KeyError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise PreferenceDataError(f"invalid preference summary: {exc}") from exc


PREFERENCE_REPORT_SCHEMA = "ai-journey-preference-comparison-v1"


def preference_report_digest(report):
    if not isinstance(report, dict):
        raise PreferenceDataError("preference report must be an object")
    return canonical_digest(
        {key: value for key, value in report.items() if key != "sha256"}
    )


def build_preference_report(result):
    """Own, summarize, digest, and validate one complete experiment result."""
    try:
        owned = json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise PreferenceDataError(f"cannot own preference result: {exc}") from exc
    report = {
        "schema": PREFERENCE_REPORT_SCHEMA,
        "result": owned,
        "summary": summarize_preference_comparison(owned),
    }
    report["sha256"] = preference_report_digest(report)
    validate_preference_report(report)
    return report


def _expect(actual, expected, name):
    if canonical_digest(actual) != canonical_digest(expected):
        raise PreferenceDataError(f"inconsistent {name}")


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PreferenceDataError(f"invalid {name} digest")


def _validate_training(training, *, config, frozen, reference_sha256):
    if not isinstance(training, dict) or set(training) != {
        "steps",
        "comparisons",
        "optimizer_steps",
        "reference_before_sha256",
        "reference_after_sha256",
        "trace",
    }:
        raise PreferenceDataError("invalid preference training fields")
    _expect(training["steps"], config.preference_steps, "training steps")
    _expect(
        training["comparisons"],
        config.preference_steps * config.preference_batch_size,
        "training comparisons",
    )
    _expect(
        training["optimizer_steps"],
        0 if frozen else config.preference_steps,
        "preference optimizer steps",
    )
    _expect(
        training["reference_before_sha256"],
        reference_sha256,
        "reference before training",
    )
    _expect(
        training["reference_after_sha256"],
        reference_sha256,
        "reference after training",
    )
    trace = training["trace"]
    if not isinstance(trace, list) or len(trace) != config.preference_steps:
        raise PreferenceDataError("preference trace must cover every step")
    for row in trace:
        if not isinstance(row, dict) or set(row) != {
            "loss",
            "policy_margin",
            "reference_margin",
            "reward_margin",
            "preference_accuracy",
            "gradient_norm",
        }:
            raise PreferenceDataError("invalid preference trace fields")
        _finite(row["loss"], "training loss", minimum=0)
        _finite(row["policy_margin"], "training policy margin")
        _finite(row["reference_margin"], "training reference margin")
        _finite(row["reward_margin"], "training reward margin")
        _finite(
            row["preference_accuracy"],
            "training preference accuracy",
            minimum=0,
            maximum=1,
        )
        gradient = _finite(row["gradient_norm"], "gradient norm", minimum=0)
        if frozen and gradient != 0:
            raise PreferenceDataError("frozen preference arm cannot have gradients")


def _validate_heldout(heldout, pair_ids):
    if not isinstance(heldout, dict) or set(heldout) != {
        "pairs",
        "preference_accuracy",
        "mean_policy_margin",
        "mean_reference_margin",
        "mean_reward_margin",
        "mean_loss",
        "rows",
    }:
        raise PreferenceDataError("invalid held-out preference fields")
    _expect(heldout["pairs"], len(pair_ids), "held-out pair count")
    rows = heldout["rows"]
    if not isinstance(rows, list) or [row.get("pair_id") for row in rows] != pair_ids:
        raise PreferenceDataError("held-out rows must cover every pair once in order")
    numeric = (
        "policy_chosen_mean_logprob",
        "policy_rejected_mean_logprob",
        "reference_chosen_mean_logprob",
        "reference_rejected_mean_logprob",
        "policy_margin",
        "reference_margin",
        "reward_margin",
        "loss",
    )
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"pair_id", *numeric, "preferred"}:
            raise PreferenceDataError("invalid held-out preference row fields")
        for name in numeric:
            _finite(
                row[name], f"held-out {name}", minimum=0 if name == "loss" else None
            )
        if type(row["preferred"]) is not bool:
            raise PreferenceDataError("held-out preference decision must be boolean")
        _expect(
            row["policy_margin"],
            row["policy_chosen_mean_logprob"] - row["policy_rejected_mean_logprob"],
            "held-out policy margin",
        )
        _expect(
            row["reference_margin"],
            row["reference_chosen_mean_logprob"]
            - row["reference_rejected_mean_logprob"],
            "held-out reference margin",
        )
        _expect(
            row["preferred"], row["policy_margin"] > 0, "held-out preference decision"
        )
    count = len(rows)
    _expect(
        heldout["preference_accuracy"],
        sum(row["preferred"] for row in rows) / count,
        "held-out accuracy",
    )
    for output, row_name in (
        ("mean_policy_margin", "policy_margin"),
        ("mean_reference_margin", "reference_margin"),
        ("mean_reward_margin", "reward_margin"),
        ("mean_loss", "loss"),
    ):
        _expect(
            heldout[output],
            math.fsum(row[row_name] for row in rows) / count,
            f"held-out {output}",
        )


def validate_preference_report(report):
    """Validate the fixed protocol, source evidence, summary, and report digest."""
    try:
        if not isinstance(report, dict) or set(report) != {
            "schema",
            "result",
            "summary",
            "sha256",
        }:
            raise PreferenceDataError("invalid preference report fields")
        _expect(report["schema"], PREFERENCE_REPORT_SCHEMA, "report schema")
        if (
            not isinstance(report["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", report["sha256"]) is None
        ):
            raise PreferenceDataError("invalid preference report digest")
        _expect(report["sha256"], preference_report_digest(report), "report digest")
        result = report["result"]
        if not isinstance(result, dict) or set(result) != {
            "config",
            "context",
            "model",
            "adapter_targets",
            "seeds",
            "runtime",
            "limitations",
        }:
            raise PreferenceDataError("invalid preference result fields")
        config = preference_config_from_dict(result["config"])
        context = prepare_preference_context(config)
        _expect(result["context"], context.evidence(), "preference context")
        _expect(
            result["model"],
            asdict(config.model_config(len(context.codec.tokens))),
            "preference model controls",
        )
        _expect(
            result["adapter_targets"],
            list(ADAPTER_TARGETS),
            "preference adapter targets",
        )
        _expect(result["limitations"], PREFERENCE_LIMITATIONS, "limitations")
        runtime = result["runtime"]
        if not isinstance(runtime, dict) or set(runtime) != {
            "python",
            "torch",
            "device",
            "dtype",
            "threads",
        }:
            raise PreferenceDataError("invalid preference runtime fields")
        _expect(
            {key: runtime[key] for key in ("device", "dtype", "threads")},
            {"device": "cpu", "dtype": "float64", "threads": 1},
            "preference runtime controls",
        )
        if not isinstance(result["seeds"], list) or [
            row["seed"] for row in result["seeds"]
        ] != list(config.seeds):
            raise PreferenceDataError("preference report requires every seed in order")
        for seed_row in result["seeds"]:
            if not isinstance(seed_row, dict) or set(seed_row) != {
                "seed",
                "warmup",
                "source_after_sha256",
                "folds",
            }:
                raise PreferenceDataError("invalid preference seed fields")
            warmup = seed_row["warmup"]
            if not isinstance(warmup, dict) or set(warmup) != {
                "initial_sha256",
                "final_sha256",
                "schedule",
                "training",
            }:
                raise PreferenceDataError("invalid preference warmup fields")
            for name in ("initial_sha256", "final_sha256", "source_after_sha256"):
                _sha(
                    seed_row[name] if name == "source_after_sha256" else warmup[name],
                    name,
                )
            _expect(
                seed_row["source_after_sha256"],
                warmup["final_sha256"],
                "warmed source fingerprint",
            )
            _expect(warmup["schedule"]["steps"], config.sft_steps, "warmup steps")
            folds = seed_row["folds"]
            if not isinstance(folds, list) or len(folds) != config.fold_count:
                raise PreferenceDataError("seed must contain every preference fold")
            for fold_index, fold_result in enumerate(folds):
                if not isinstance(fold_result, dict) or set(fold_result) != {
                    "fold",
                    "schedule",
                    "reference_sha256",
                    "trials",
                }:
                    raise PreferenceDataError("invalid preference fold fields")
                fold = context.folds[fold_index]
                _expect(fold_result["fold"], fold.evidence(), "fold evidence")
                expected_schedule = preference_schedule_evidence(
                    preference_training_batches(
                        fold.train,
                        batch_size=config.preference_batch_size,
                        steps=config.preference_steps,
                        seed=seed_row["seed"] ^ (fold.index + 1) ^ 0xD90,
                    )
                )
                _expect(
                    fold_result["schedule"], expected_schedule, "preference schedule"
                )
                _expect(
                    fold_result["reference_sha256"],
                    warmup["final_sha256"],
                    "fold reference fingerprint",
                )
                trials = fold_result["trials"]
                if not isinstance(trials, list) or [
                    trial.get("name") for trial in trials
                ] != list(PREFERENCE_ARMS):
                    raise PreferenceDataError("fold must contain every arm once")
                pair_ids = [pair.pair_id for pair in fold.heldout]
                for trial in trials:
                    if not isinstance(trial, dict) or set(trial) != {
                        "name",
                        "adapter",
                        "parameters",
                        "initial_function_max_abs_error",
                        "final_sha256",
                        "frozen_before_sha256",
                        "frozen_after_sha256",
                        "training",
                        "heldout",
                        "merge",
                    }:
                        raise PreferenceDataError("invalid preference trial fields")
                    for name in (
                        "final_sha256",
                        "frozen_before_sha256",
                        "frozen_after_sha256",
                    ):
                        _sha(trial[name], name)
                    _expect(
                        trial["frozen_after_sha256"],
                        trial["frozen_before_sha256"],
                        "frozen preference state",
                    )
                    if (
                        _finite(
                            trial["initial_function_max_abs_error"],
                            "matched initial policy function",
                            minimum=0,
                        )
                        != 0
                    ):
                        raise PreferenceDataError(
                            "inconsistent matched initial policy function"
                        )
                    _validate_training(
                        trial["training"],
                        config=config,
                        frozen=trial["name"] == "frozen",
                        reference_sha256=fold_result["reference_sha256"],
                    )
                    _validate_heldout(trial["heldout"], pair_ids)
                    is_adapter = trial["name"].startswith("lora-rank-")
                    if is_adapter:
                        rank = int(trial["name"].rsplit("-", 1)[1])
                        _expect(trial["adapter"]["rank"], rank, "adapter rank")
                        if not isinstance(trial["merge"], dict) or set(
                            trial["merge"]
                        ) != {"probe_max_abs_error", "dense_sha256", "parameters"}:
                            raise PreferenceDataError(
                                "invalid preference merge evidence"
                            )
                        _sha(trial["merge"]["dense_sha256"], "merged dense")
                        error = _finite(
                            trial["merge"]["probe_max_abs_error"],
                            "preference merge error",
                            minimum=0,
                        )
                        if error > 1e-10:
                            raise PreferenceDataError(
                                "preference merge equivalence failed"
                            )
                    elif trial["adapter"] is not None or trial["merge"] is not None:
                        raise PreferenceDataError(
                            "dense preference arms cannot contain adapter evidence"
                        )
        _expect(
            report["summary"],
            summarize_preference_comparison(result),
            "preference summary",
        )
        return report
    except (KeyError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise PreferenceDataError(f"invalid preference report: {exc}") from exc


def render_preference_report(report):
    validate_preference_report(report)
    return json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_preference_report(path, report):
    """Durably replace a validated report and clean a failed temporary file."""
    rendered = render_preference_report(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _replay_content(report):
    return {
        "schema": report["schema"],
        "result": {
            key: value for key, value in report["result"].items() if key != "runtime"
        },
        "summary": report["summary"],
    }


def replay_preference_report(report):
    """Rebuild every seed-fold arm and require exact non-environmental evidence."""
    validate_preference_report(report)
    config = preference_config_from_dict(report["result"]["config"])
    replay = build_preference_report(run_preference_comparison(config))
    if canonical_digest(_replay_content(report)) != canonical_digest(
        _replay_content(replay)
    ):
        raise PreferenceDataError(
            "exact preference replay mismatch: fingerprints, schedules, traces, "
            "held-out metrics, or paired summaries do not reproduce"
        )
    return {
        "sha256": report["sha256"],
        "verified_bases": len(config.seeds),
        "verified_folds": len(config.seeds) * config.fold_count,
        "verified_trials": len(config.seeds) * config.fold_count * len(PREFERENCE_ARMS),
        "policy": (
            "exact CPU float64 replay; only result.runtime and the report digest excluded"
        ),
    }
