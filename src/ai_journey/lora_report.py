"""Descriptive paired-seed comparisons and verifiable CPU experiment evidence."""

from __future__ import annotations

import math
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import tempfile

from .lora_data import (
    FIXTURE_ID,
    LoRAExperimentError,
    canonical_digest,
    heldout_evidence,
    schedule_evidence,
    synthetic_fixture,
    training_batches,
)
from .lora_experiment import (
    ADAPTER_TARGETS,
    COMPARISON_ARMS,
    LIMITATIONS,
    config_from_dict,
)
from .transformer_lab import expected_parameter_count


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


SCHEMA = "ai-journey-lora-comparison-v1"


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise LoRAExperimentError(f"invalid {name} fields")


def _expect(value, expected, name):
    # JSON equality distinguishes booleans from ints and owns collection order.
    if canonical_digest(value) != canonical_digest(expected):
        raise LoRAExperimentError(f"inconsistent {name}")


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise LoRAExperimentError(f"invalid {name} digest")


def report_digest(report):
    return canonical_digest(
        {key: value for key, value in report.items() if key != "sha256"}
    )


def _validate_trace(trace, *, steps, tokens, frozen=False):
    _keys(
        trace, {"losses", "consumed_target_tokens", "optimizer_steps"}, "training trace"
    )
    if not isinstance(trace["losses"], list) or len(trace["losses"]) != steps:
        raise LoRAExperimentError("training trace must cover every scheduled step")
    for value in trace["losses"]:
        _finite_number(value, "training loss", minimum=0)
    _expect(trace["consumed_target_tokens"], tokens, "training target exposure")
    _expect(trace["optimizer_steps"], 0 if frozen else steps, "optimizer steps")


def _validate_score(score, targets):
    _keys(
        score, {"nll_sum", "target_tokens", "mean_nll", "perplexity"}, "heldout metrics"
    )
    _expect(score["target_tokens"], targets, "heldout target coverage")
    for name in ("nll_sum", "mean_nll", "perplexity"):
        _finite_number(score[name], name, minimum=0)
    _expect(score["mean_nll"], score["nll_sum"] / targets, "token-weighted heldout NLL")
    _expect(score["perplexity"], math.exp(score["mean_nll"]), "perplexity")


def _validate_trial(trial, name, seed, config, base_count, targets, pretrained):
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
        "trial",
    )
    _expect(trial["name"], name, "ordered comparison arm")
    for field in (
        "initial_sha256",
        "final_sha256",
        "frozen_before_sha256",
        "frozen_after_sha256",
    ):
        _sha(trial[field], field)
    _expect(
        trial["frozen_after_sha256"],
        trial["frozen_before_sha256"],
        "frozen state preservation",
    )
    _finite_number(
        trial["initial_function_max_abs_error"], "initial probe error", minimum=0
    )
    if trial["initial_function_max_abs_error"] != 0:
        raise LoRAExperimentError("all arms must start from the base function")
    if name in ("full", "frozen"):
        _expect(trial["initial_sha256"], pretrained, "dense fork source")
        _expect(trial["adapter"], None, "dense adapter controls")
        _expect(trial["merge"], None, "dense merge evidence")
        trainable = base_count if name == "full" else 0
        counts = {
            "total": base_count,
            "trainable": trainable,
            "frozen": base_count - trainable,
        }
    else:
        rank = 1 if name == "lora-rank-1" else 2
        _expect(
            trial["adapter"],
            {
                "rank": rank,
                "alpha": rank * config.adapter_scale,
                "dropout": 0.0,
                "seed": seed,
            },
            "adapter controls",
        )
        trainable = 6 * config.embedding_dim * rank
        counts = {
            "total": base_count + trainable,
            "trainable": trainable,
            "frozen": base_count,
        }
        merge = trial["merge"]
        _keys(merge, {"probe_max_abs_error", "dense_sha256", "parameters"}, "merge")
        _sha(merge["dense_sha256"], "merged dense")
        _finite_number(merge["probe_max_abs_error"], "merge error", minimum=0)
        if merge["probe_max_abs_error"] > 1e-10:
            raise LoRAExperimentError("dense merge equivalence failed")
        _expect(
            merge["parameters"],
            {"total": base_count, "trainable": 0, "frozen": base_count},
            "merged parameter accounting",
        )
    _expect(trial["parameters"], counts, "parameter accounting")
    if name == "frozen":
        _expect(trial["final_sha256"], pretrained, "unchanged frozen baseline")
    elif trial["initial_sha256"] == trial["final_sha256"]:
        raise LoRAExperimentError("trainable comparison arm did not change")
    _validate_trace(
        trial["training"],
        steps=config.finetune_steps,
        tokens=config.finetune_steps * config.batch_size * config.block_size,
        frozen=name == "frozen",
    )
    _validate_score(trial["heldout"], targets)


def validate_report(report):
    """Check schema, internal accounting, fixture/schedules, and digest, without training.

    This detects inconsistent evidence. A self-consistent forged metric requires
    the separate full replay check; a digest is never treated as authenticity.
    """
    try:
        _keys(report, {"schema", "result", "summary", "sha256"}, "report")
        _expect(report["schema"], SCHEMA, "report schema")
        _sha(report["sha256"], "report")
        _expect(report["sha256"], report_digest(report), "report digest")
        result = report["result"]
        _keys(
            result,
            {
                "config",
                "fixture",
                "model",
                "adapter_targets",
                "heldout",
                "seeds",
                "runtime",
                "limitations",
            },
            "result",
        )
        config = config_from_dict(result["config"])
        split = synthetic_fixture()
        codec = split.codec()
        _expect(
            result["fixture"],
            {"id": FIXTURE_ID, **split.evidence()},
            "fixture identity and splits",
        )
        model_config = config.model_config(len(codec.tokens))
        _expect(result["model"], asdict(model_config), "bounded model architecture")
        _expect(result["adapter_targets"], list(ADAPTER_TARGETS), "adapter targets")
        coverage = heldout_evidence(
            split.validation, codec, block_size=config.block_size
        )
        _expect(result["heldout"], coverage, "document-local target coverage")
        _expect(result["limitations"], LIMITATIONS, "scope limitations")
        runtime = result["runtime"]
        _keys(runtime, {"python", "torch", "device", "dtype", "threads"}, "runtime")
        for field in ("python", "torch"):
            if (
                not isinstance(runtime[field], str)
                or not 1 <= len(runtime[field]) <= 128
            ):
                raise LoRAExperimentError("invalid runtime version metadata")
        for field, expected in (
            ("device", "cpu"),
            ("dtype", "float64"),
            ("threads", 1),
        ):
            _expect(runtime[field], expected, "runtime scope")
        if not isinstance(result["seeds"], list) or len(result["seeds"]) != len(
            config.seeds
        ):
            raise LoRAExperimentError("missing or extra paired seed evidence")
        base_count = expected_parameter_count(model_config)
        for row, seed in zip(result["seeds"], config.seeds, strict=True):
            _keys(
                row,
                {
                    "seed",
                    "initial_sha256",
                    "pretrained_sha256",
                    "source_after_sha256",
                    "pretrain_schedule",
                    "adaptation_schedule",
                    "pretraining",
                    "trials",
                },
                "seed",
            )
            _expect(row["seed"], seed, "ordered seed")
            for field in ("initial_sha256", "pretrained_sha256", "source_after_sha256"):
                _sha(row[field], field)
            _expect(
                row["source_after_sha256"],
                row["pretrained_sha256"],
                "pretrained source preservation",
            )
            if row["initial_sha256"] == row["pretrained_sha256"]:
                raise LoRAExperimentError("pretraining did not change the model")
            for name, documents, steps, salt in (
                ("pretrain_schedule", split.pretrain, config.pretrain_steps, 0xB45E),
                ("adaptation_schedule", split.train, config.finetune_steps, 0xA11CE),
            ):
                batches = training_batches(
                    documents,
                    codec,
                    block_size=config.block_size,
                    batch_size=config.batch_size,
                    steps=steps,
                    seed=seed ^ salt,
                )
                _expect(row[name], schedule_evidence(batches), name)
            _validate_trace(
                row["pretraining"],
                steps=config.pretrain_steps,
                tokens=config.pretrain_steps * config.batch_size * config.block_size,
            )
            if not isinstance(row["trials"], list) or len(row["trials"]) != len(
                COMPARISON_ARMS
            ):
                raise LoRAExperimentError("each seed requires all four comparison arms")
            for trial, name in zip(row["trials"], COMPARISON_ARMS, strict=True):
                _validate_trial(
                    trial,
                    name,
                    seed,
                    config,
                    base_count,
                    coverage["target_tokens"],
                    row["pretrained_sha256"],
                )
        _expect(report["summary"], summarize_comparison(result), "paired summary")
    except (TypeError, KeyError, ValueError, OverflowError, RecursionError) as exc:
        raise LoRAExperimentError(f"invalid comparison report: {exc}") from exc


def build_report(result):
    """Own nested values and compute a canonical digest with no timestamps."""
    try:
        owned = json.loads(json.dumps(result, allow_nan=False))
        report = {
            "schema": SCHEMA,
            "result": owned,
            "summary": summarize_comparison(owned),
        }
        report["sha256"] = report_digest(report)
        validate_report(report)
        return report
    except (TypeError, ValueError, RecursionError) as exc:
        raise LoRAExperimentError(f"cannot build comparison report: {exc}") from exc


def render_report(report):
    validate_report(report)
    return json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_report(path, report):
    """Validate before atomically replacing a report; clean failed temp files."""
    rendered = render_report(report)
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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
