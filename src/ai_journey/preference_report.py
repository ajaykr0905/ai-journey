"""Validated evidence reports for the bounded preference comparison."""

from __future__ import annotations

import math

from .preference_data import PreferenceDataError
from .preference_experiment import PREFERENCE_ARMS


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
        full_accuracy = next(
            arm["preference_accuracy"]["mean"] for arm in arms if arm["name"] == "full"
        )
        negative_findings = [
            (
                f"{arm['name']} did not exceed the full arm's mean held-out "
                "preference accuracy."
            )
            for arm in arms
            if arm["name"].startswith("lora-")
            and arm["preference_accuracy"]["mean"] <= full_accuracy
        ]
        return {
            "observations": expected,
            "arms": arms,
            "paired": paired,
            "negative_findings": negative_findings,
            "uncertainty_policy": UNCERTAINTY_POLICY,
        }
    except (KeyError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise PreferenceDataError(f"invalid preference summary: {exc}") from exc
