"""Validated summaries and evidence for paired chat SFT comparisons."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

from .sft_data import (
    CHAT_FIXTURE_ID,
    SFTDataError,
    authored_chat_fixture,
    canonical_digest,
    sft_validation_evidence,
)
from .sft_experiment import (
    SFT_ARMS,
    SFT_LIMITATIONS,
    run_sft_comparison,
    sft_config_dict,
    sft_config_from_dict,
)

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


SFT_REPORT_SCHEMA = "ai-journey-chat-sft-comparison-v1"


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise SFTDataError(f"invalid {name} fields")


def _expect(value, expected, name):
    if canonical_digest(value) != canonical_digest(expected):
        raise SFTDataError(f"inconsistent {name}")


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SFTDataError(f"invalid {name} digest")


def sft_report_digest(report):
    return canonical_digest(
        {key: value for key, value in report.items() if key != "sha256"}
    )


def build_sft_report(result):
    try:
        owned = json.loads(json.dumps(result, allow_nan=False))
        report = {
            "schema": SFT_REPORT_SCHEMA,
            "result": owned,
            "summary": summarize_sft_comparison(owned),
        }
        report["sha256"] = sft_report_digest(report)
        validate_sft_report(report)
        return report
    except (TypeError, ValueError, RecursionError) as exc:
        raise SFTDataError(f"cannot build SFT report: {exc}") from exc


def _validate_training(trace, *, steps, input_tokens, supervised_tokens, frozen):
    _keys(
        trace,
        {"losses", "input_tokens", "supervised_tokens", "optimizer_steps"},
        "SFT training trace",
    )
    if not isinstance(trace["losses"], list) or len(trace["losses"]) != steps:
        raise SFTDataError("SFT trace must cover every scheduled step")
    for loss in trace["losses"]:
        _finite(loss, "SFT loss", minimum=0)
    _expect(trace["input_tokens"], input_tokens, "SFT input exposure")
    _expect(trace["supervised_tokens"], supervised_tokens, "SFT supervised exposure")
    _expect(trace["optimizer_steps"], 0 if frozen else steps, "SFT optimizer steps")


def _validate_score(score, targets):
    _keys(
        score,
        {"nll_sum", "target_tokens", "mean_nll", "perplexity"},
        "assistant metrics",
    )
    _expect(score["target_tokens"], targets, "assistant target coverage")
    for name in ("nll_sum", "mean_nll", "perplexity"):
        _finite(score[name], name, minimum=0)
    _expect(score["mean_nll"], score["nll_sum"] / targets, "assistant mean NLL")
    _expect(score["perplexity"], math.exp(score["mean_nll"]), "assistant perplexity")


def validate_sft_report(report):
    """Validate schema, accounting, fixed data evidence, summary, and digest."""
    try:
        _keys(report, {"schema", "result", "summary", "sha256"}, "SFT report")
        _expect(report["schema"], SFT_REPORT_SCHEMA, "SFT report schema")
        _sha(report["sha256"], "SFT report")
        _expect(report["sha256"], sft_report_digest(report), "SFT report digest")
        result = report["result"]
        _keys(
            result,
            {
                "config",
                "fixture",
                "model",
                "heldout",
                "seeds",
                "runtime",
                "limitations",
            },
            "SFT result",
        )
        config = sft_config_from_dict(result["config"])
        _expect(result["config"], sft_config_dict(config), "SFT config")
        split = authored_chat_fixture()
        _expect(
            result["fixture"],
            {"id": CHAT_FIXTURE_ID, **split.evidence()},
            "SFT fixture evidence",
        )
        _expect(
            result["heldout"],
            sft_validation_evidence(
                split.validation, split.codec(), block_size=config.block_size
            ),
            "SFT heldout evidence",
        )
        _expect(result["limitations"], SFT_LIMITATIONS, "SFT limitations")
        runtime = result["runtime"]
        _keys(runtime, {"python", "torch", "device", "dtype", "threads"}, "runtime")
        _expect(
            {key: runtime[key] for key in ("device", "dtype", "threads")},
            {"device": "cpu", "dtype": "float64", "threads": 1},
            "SFT runtime controls",
        )
        if not isinstance(result["seeds"], list) or [
            row["seed"] for row in result["seeds"]
        ] != list(config.seeds):
            raise SFTDataError("SFT report requires every paired seed in order")
        targets = result["heldout"]["target_tokens"]
        expected_input = config.sft_steps * config.batch_size * config.block_size
        for seed in result["seeds"]:
            _keys(
                seed,
                {
                    "seed",
                    "initial_sha256",
                    "warmed_sha256",
                    "source_after_sha256",
                    "base_schedule",
                    "sft_schedule",
                    "base_training",
                    "trials",
                },
                "SFT seed result",
            )
            for name in ("initial_sha256", "warmed_sha256", "source_after_sha256"):
                _sha(seed[name], name)
            _expect(
                seed["source_after_sha256"],
                seed["warmed_sha256"],
                "unmodified warmed source",
            )
            for schedule_name in ("base_schedule", "sft_schedule"):
                _sha(seed[schedule_name]["sha256"], schedule_name)
            schedule = seed["sft_schedule"]
            _expect(schedule["steps"], config.sft_steps, "SFT schedule steps")
            _expect(
                schedule["input_tokens"], expected_input, "SFT schedule input tokens"
            )
            supervised = schedule["supervised_tokens"]
            if type(supervised) is not int or not 0 < supervised <= expected_input:
                raise SFTDataError("invalid SFT supervised token count")
            trials = seed["trials"]
            if not isinstance(trials, list) or [
                trial["name"] for trial in trials
            ] != list(SFT_ARMS):
                raise SFTDataError("SFT report requires every arm once in order")
            for trial in trials:
                _keys(
                    trial,
                    {
                        "name",
                        "adapter",
                        "merge",
                        "parameters",
                        "initial_sha256",
                        "final_sha256",
                        "frozen_before_sha256",
                        "frozen_after_sha256",
                        "initial_function_max_abs_error",
                        "training",
                        "heldout",
                    },
                    "SFT trial",
                )
                for name in (
                    "initial_sha256",
                    "final_sha256",
                    "frozen_before_sha256",
                    "frozen_after_sha256",
                ):
                    _sha(trial[name], name)
                _expect(
                    trial["frozen_after_sha256"],
                    trial["frozen_before_sha256"],
                    "frozen SFT state",
                )
                if (
                    _finite(
                        trial["initial_function_max_abs_error"],
                        "matched SFT initial function",
                        minimum=0,
                    )
                    != 0
                ):
                    raise SFTDataError("SFT arms must start from the same function")
                _validate_training(
                    trial["training"],
                    steps=config.sft_steps,
                    input_tokens=expected_input,
                    supervised_tokens=supervised,
                    frozen=trial["name"] == "frozen",
                )
                _validate_score(trial["heldout"], targets)
                is_adapter = trial["name"].startswith("lora-rank-")
                if is_adapter:
                    rank = int(trial["name"].rsplit("-", 1)[1])
                    _expect(trial["adapter"]["rank"], rank, "SFT adapter rank")
                    _keys(
                        trial["merge"],
                        {"probe_max_abs_error", "dense_sha256", "parameters"},
                        "SFT merge",
                    )
                    _sha(trial["merge"]["dense_sha256"], "SFT merged dense")
                    if (
                        not 0
                        <= _finite(
                            trial["merge"]["probe_max_abs_error"],
                            "SFT merge error",
                        )
                        <= 1e-10
                    ):
                        raise SFTDataError("SFT merge equivalence failed")
                elif trial["adapter"] is not None or trial["merge"] is not None:
                    raise SFTDataError("dense SFT arms cannot include adapter evidence")
        _expect(
            report["summary"],
            summarize_sft_comparison(result),
            "SFT comparison summary",
        )
        return report
    except (KeyError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise SFTDataError(f"invalid SFT report: {exc}") from exc


def render_sft_report(report):
    validate_sft_report(report)
    return json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_sft_report(path, report):
    """Durably replace a validated report and clean any failed temporary file."""
    rendered = render_sft_report(report)
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
    """Exclude only the digest and explicitly environmental runtime metadata."""
    return {
        "schema": report["schema"],
        "result": {
            key: value for key, value in report["result"].items() if key != "runtime"
        },
        "summary": report["summary"],
    }


def replay_sft_report(report):
    """Rebuild every paired chat adaptation arm and require exact evidence."""
    validate_sft_report(report)
    config = sft_config_from_dict(report["result"]["config"])
    replay = build_sft_report(run_sft_comparison(config))
    if canonical_digest(_replay_content(report)) != canonical_digest(
        _replay_content(replay)
    ):
        raise SFTDataError(
            "exact chat SFT replay mismatch: model fingerprints, training traces, "
            "assistant metrics, or paired summaries do not reproduce"
        )
    return {
        "sha256": report["sha256"],
        "verified_bases": len(config.seeds),
        "verified_trials": len(config.seeds) * len(SFT_ARMS),
        "policy": (
            "exact CPU float64 replay; only result.runtime and the report digest "
            "excluded"
        ),
    }
