"""Descriptive paired-seed comparisons and verifiable CPU experiment evidence."""

from __future__ import annotations

import math

from .lora_data import LoRAExperimentError
from .lora_experiment import COMPARISON_ARMS


UNCERTAINTY_POLICY = (
    "Sample standard deviation, standard error, and observed range describe only "
    "these paired seeds on one fixed synthetic split. They are not confidence "
    "intervals or evidence of generalization to other tasks or larger models. "
    "Paired delta = candidate NLL minus reference NLL; negative favors candidate."
)


def _finite_number(value, name, *, minimum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        raise LoRAExperimentError(f"invalid finite {name}")
    return value


def _describe(values):
    count = len(values)
    mean = math.fsum(values) / count
    deviation = math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (count - 1)
    )
    return {
        "count": count,
        "mean": mean,
        "sample_stddev": deviation,
        "standard_error": deviation / math.sqrt(count),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_comparison(result):
    """Compute differences within each seed before reporting cross-seed spread."""
    try:
        seeds = result["config"]["seeds"]
        if (
            not isinstance(seeds, list)
            or not 2 <= len(seeds) <= 4
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or not isinstance(result["seeds"], list)
            or [row["seed"] for row in result["seeds"]] != seeds
        ):
            raise LoRAExperimentError("summary requires complete distinct paired seeds")
        target_tokens = result["heldout"]["target_tokens"]
        if type(target_tokens) is not int or target_tokens <= 0:
            raise LoRAExperimentError("summary requires held-out targets")
        measurements = {arm: [] for arm in COMPARISON_ARMS}
        counts = {}
        for row in result["seeds"]:
            if not isinstance(row["trials"], list) or [
                trial["name"] for trial in row["trials"]
            ] != list(COMPARISON_ARMS):
                raise LoRAExperimentError(
                    "summary requires every comparison arm once in order"
                )
            for trial in row["trials"]:
                arm = trial["name"]
                score = trial["heldout"]
                if (
                    type(score["target_tokens"]) is not int
                    or score["target_tokens"] != target_tokens
                ):
                    raise LoRAExperimentError(
                        "paired trials require identical held-out coverage"
                    )
                measurements[arm].append(
                    _finite_number(score["mean_nll"], "mean_nll", minimum=0)
                )
                parameters = trial["parameters"]
                if (
                    not isinstance(parameters, dict)
                    or set(parameters) != {"total", "trainable", "frozen"}
                    or any(
                        type(value) is not int or value < 0
                        for value in parameters.values()
                    )
                    or parameters["total"]
                    != parameters["trainable"] + parameters["frozen"]
                    or parameters["total"] <= 0
                ):
                    raise LoRAExperimentError("invalid parameter accounting")
                if arm in counts and counts[arm] != parameters:
                    raise LoRAExperimentError(
                        "parameter counts differ between paired seeds"
                    )
                counts[arm] = parameters
        full_parameters = counts["full"]["trainable"]
        if full_parameters <= 0:
            raise LoRAExperimentError("full fine-tuning requires trainable parameters")
        arms = [
            {
                "name": arm,
                "parameters": dict(counts[arm]),
                "trainable_fraction_of_full": counts[arm]["trainable"]
                / full_parameters,
                "heldout_mean_nll": _describe(measurements[arm]),
            }
            for arm in COMPARISON_ARMS
        ]
        paired = []
        for index, reference in enumerate(COMPARISON_ARMS):
            for candidate in COMPARISON_ARMS[index + 1 :]:
                deltas = [
                    candidate_nll - reference_nll
                    for candidate_nll, reference_nll in zip(
                        measurements[candidate], measurements[reference], strict=True
                    )
                ]
                paired.append(
                    {
                        "reference": reference,
                        "candidate": candidate,
                        "per_seed": [
                            {"seed": seed, "delta_nll": delta}
                            for seed, delta in zip(seeds, deltas, strict=True)
                        ],
                        "delta_nll": _describe(deltas),
                        "candidate_wins": sum(delta < 0 for delta in deltas),
                        "ties": sum(delta == 0 for delta in deltas),
                    }
                )
        return {
            "arms": arms,
            "paired": paired,
            "uncertainty_policy": UNCERTAINTY_POLICY,
        }
    except (KeyError, TypeError, OverflowError) as exc:
        raise LoRAExperimentError(f"invalid comparison summary input: {exc}") from exc
