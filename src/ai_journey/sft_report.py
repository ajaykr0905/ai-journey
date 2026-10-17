"""Validated summaries and evidence for paired chat SFT comparisons."""

from __future__ import annotations

import math

from .sft_data import SFTDataError
from .sft_experiment import SFT_ARMS

UNCERTAINTY_POLICY = (
    "Sample standard deviation, standard error, and observed range describe only "
    "the paired seeds on one fixed authored split. They are not confidence "
    "intervals or evidence about other tasks or model scales. Paired delta is "
    "candidate assistant NLL minus reference assistant NLL; negative favors the "
    "candidate on this held-out split."
)


def _finite(value, name, *, minimum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        raise SFTDataError(f"invalid finite {name}")
    return value


def _describe(values):
    if not isinstance(values, list) or len(values) < 2:
        raise SFTDataError("uncertainty summary requires at least two values")
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


def summarize_sft_comparison(result):
    """Summarize paired assistant NLL before aggregating across seeds."""
    try:
        seeds = result["config"]["seeds"]
        rows = result["seeds"]
        target_tokens = result["heldout"]["target_tokens"]
        if (
            not isinstance(seeds, list)
            or not 2 <= len(seeds) <= 4
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or not isinstance(rows, list)
            or [row["seed"] for row in rows] != seeds
            or type(target_tokens) is not int
            or target_tokens <= 0
        ):
            raise SFTDataError("summary requires complete paired seeds and targets")
        measurements = {arm: [] for arm in SFT_ARMS}
        counts = {}
        for row in rows:
            trials = row["trials"]
            if not isinstance(trials, list) or [
                trial["name"] for trial in trials
            ] != list(SFT_ARMS):
                raise SFTDataError("summary requires every chat arm once in order")
            for trial in trials:
                name = trial["name"]
                score = trial["heldout"]
                if score["target_tokens"] != target_tokens:
                    raise SFTDataError("chat arms require identical held-out coverage")
                measurements[name].append(
                    _finite(score["mean_nll"], "assistant mean_nll", minimum=0)
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
                    raise SFTDataError("invalid chat parameter accounting")
                if name in counts and counts[name] != parameters:
                    raise SFTDataError("chat parameter counts differ between seeds")
                counts[name] = dict(parameters)
        full_trainable = counts["full"]["trainable"]
        if full_trainable <= 0:
            raise SFTDataError("full chat SFT requires trainable parameters")
        arms = [
            {
                "name": name,
                "parameters": counts[name],
                "trainable_fraction_of_full": counts[name]["trainable"]
                / full_trainable,
                "assistant_mean_nll": _describe(measurements[name]),
            }
            for name in SFT_ARMS
        ]
        paired = []
        for index, reference in enumerate(SFT_ARMS):
            for candidate in SFT_ARMS[index + 1 :]:
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
        raise SFTDataError(f"invalid chat comparison summary: {exc}") from exc
