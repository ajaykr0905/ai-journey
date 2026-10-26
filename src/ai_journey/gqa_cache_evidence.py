"""Actual source-replay-backed native KV storage and numerical cache evidence."""

from hashlib import sha256
import math
import struct

import torch

from .ddp_cache import cached_forward
from .decoder_variant_comparison import decoder_comparison_digest
from .gqa_experiment import run_gqa_comparison
from .gqa_experiment_plan import require_gqa_experiment_budget
from .gqa_experiment_public import PUBLIC_GQA_SCHEMA, verify_public_gqa_report
from .gqa_experiment_report import verify_gqa_experiment_report
from .gqa_resources import audit_gqa_cache_storage, require_gqa_execution_budget
from .moe_report import _digest, _equal, _fields, _integer, canonical_moe_json

CACHE_SCHEMA = "ai-journey-controlled-gqa-cache-v1"
CACHE_POLICY = "one-token native KV growth from empty; public document-local prefixes; no context overflow"
CACHE_SCOPE = "actual bounded float32 CPU backing-store inventory and numerical full/cache comparison; no latency/GPU/production/quality claim; integrity is not authentication"
ATOL, RTOL = 2e-6, 1e-5


def _source(report):
    if type(report) is dict and report.get("schema") == PUBLIC_GQA_SCHEMA:
        return verify_public_gqa_report(report), report["comparison"]
    return verify_gqa_experiment_report(report), report


def _prefixes(comparison, plan, count):
    _integer(count, 1, 4)
    documents = comparison["documents"]
    return [
        {
            "document_index": index,
            "document_sha256": digest,
            "length": min(length, plan.decoder.gpt2.context_length),
        }
        for index, (digest, length) in enumerate(
            zip(
                documents["heldout_documents"][:count],
                documents["heldout_lengths"][:count],
                strict=True,
            )
        )
    ]


def _logits(values):
    if type(values) is not list or len(values) != 256:
        raise ValueError("cache evidence requires complete byte-vocabulary logit rows")
    for value in values:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("cache evidence logits must be measured finite floats")
        try:
            exact = struct.unpack("f", struct.pack("f", value))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError("cache evidence logits exceed float32") from exc
        if exact != value:
            raise ValueError("cache evidence logits do not retain float32 values")


def _parity(full, cached):
    _logits(full)
    _logits(cached)
    residuals = [abs(left - right) for left, right in zip(full, cached, strict=True)]
    ratio = max(
        delta / (ATOL + RTOL * abs(value))
        for delta, value in zip(residuals, full, strict=True)
    )
    if ratio > 1:
        raise ValueError("actual cached logits exceed the declared numerical tolerance")
    return {
        "compared_logits": len(full),
        "max_absolute_error": max(residuals),
        "max_tolerance_ratio": ratio,
    }


def _summarize(plan, observations):
    return [
        {
            "kv_heads": heads,
            "observed_prefixes": sum(row["kv_heads"] == heads for row in observations),
            "growth_steps": sum(
                len(row["steps"]) for row in observations if row["kv_heads"] == heads
            ),
            "compared_logits": sum(
                step["parity"]["compared_logits"]
                for row in observations
                if row["kv_heads"] == heads
                for step in row["steps"]
            ),
            "max_absolute_error": max(
                step["parity"]["max_absolute_error"]
                for row in observations
                if row["kv_heads"] == heads
                for step in row["steps"]
            ),
            "max_tolerance_ratio": max(
                step["parity"]["max_tolerance_ratio"]
                for row in observations
                if row["kv_heads"] == heads
                for step in row["steps"]
            ),
            "max_retained_cache_bytes": max(
                step["storage"]["retained_storage_bytes"]
                for row in observations
                if row["kv_heads"] == heads
                for step in row["steps"]
            ),
        }
        for heads in plan.kv_heads
    ]


def run_gqa_cache_evidence(
    source_report, train_bytes, heldout_bytes, *, prefix_count=4
):
    """Rerun the comparison, inspecting actual final-model caches as it completes.

    No checkpoint or untrusted model identity is substituted for actual source
    replay. Prefix logits retain every compared float32 value; storage accounting
    inspects real CPU backing stores, not expanded logical tensor dimensions.
    """
    plan, comparison = _source(source_report)
    if (
        type(train_bytes) is not bytes
        or type(heldout_bytes) is not bytes
        or sha256(train_bytes).hexdigest() != comparison["train_sha256"]
        or sha256(heldout_bytes).hexdigest() != comparison["heldout_sha256"]
        or comparison["runtime"]["torch_version"] != str(torch.__version__)
    ):
        raise ValueError(
            "cache evidence requires original source bytes and CPU runtime"
        )
    prefixes = _prefixes(comparison, plan, prefix_count)
    lengths = tuple(row["length"] for row in prefixes)
    budget = require_gqa_experiment_budget(
        plan,
        evaluation_windows=comparison["evaluation_windows"],
        cache_prefix_lengths=lengths,
    )
    observations = []

    def completed(seed, model, measured, documents):
        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            with torch.inference_mode():
                for declared in prefixes:
                    prefix = documents.heldout[declared["document_index"]].content[
                        : declared["length"]
                    ]
                    inputs = torch.tensor([list(prefix)], dtype=torch.long)
                    full = model(inputs)[0].tolist()
                    cache, steps = None, []
                    for index in range(len(prefix)):
                        output, cache = cached_forward(
                            model, inputs[:, index : index + 1], cache
                        )
                        cached = output[0, 0].tolist()
                        steps.append(
                            {
                                "length": cache.length,
                                "shape": list(cache.keys[0].shape),
                                "storage": audit_gqa_cache_storage(model, cache),
                                "execution": require_gqa_execution_budget(
                                    model.config,
                                    batch_size=1,
                                    sequence_length=index + 1,
                                    query_tokens=1,
                                ),
                                "full_logits": full[index],
                                "cached_logits": cached,
                                "parity": _parity(full[index], cached),
                            }
                        )
                    observations.append(
                        {
                            "seed": seed,
                            "kv_heads": model.config.kv_heads,
                            "model_sha256": measured["final_model_sha256"],
                            "prefix": {
                                **declared,
                                "sha256": sha256(prefix).hexdigest(),
                            },
                            "full_execution": require_gqa_execution_budget(
                                model.config, batch_size=1, sequence_length=len(prefix)
                            ),
                            "steps": steps,
                        }
                    )
        finally:
            for module, mode in modes:
                module.training = mode

    actual = run_gqa_comparison(
        plan,
        train_bytes,
        heldout_bytes,
        _completed_arm=completed,
        _cache_prefix_lengths=lengths,
    )
    if canonical_moe_json(actual) != canonical_moe_json(comparison):
        raise ValueError("cache evidence source comparison differs from actual replay")
    report = {
        "schema": CACHE_SCHEMA,
        "source_report": source_report,
        "prefix_count": prefix_count,
        "policy": CACHE_POLICY,
        "tolerance": {"absolute": ATOL, "relative": RTOL},
        "resource_admission": budget,
        "observations": observations,
        "summary": _summarize(plan, observations),
        "verification_scope": CACHE_SCOPE,
    }
    report["report_sha256"] = decoder_comparison_digest(report)
    verify_gqa_cache_evidence(report)
    return report


def verify_gqa_cache_evidence(report):
    """Recompute native storage/score inventories and every raw logit residual."""
    _fields(
        report,
        "schema source_report prefix_count policy tolerance resource_admission observations summary verification_scope report_sha256",
        "GQA cache report",
    )
    if (
        report["schema"] != CACHE_SCHEMA
        or report["policy"] != CACHE_POLICY
        or report["verification_scope"] != CACHE_SCOPE
    ):
        raise ValueError("GQA cache schema or protocol differs")
    _equal(
        report["report_sha256"], decoder_comparison_digest(report), "GQA cache digest"
    )
    _equal(
        report["tolerance"], {"absolute": ATOL, "relative": RTOL}, "GQA cache tolerance"
    )
    plan, comparison = _source(report["source_report"])
    prefixes = _prefixes(comparison, plan, report["prefix_count"])
    _equal(
        report["resource_admission"],
        require_gqa_experiment_budget(
            plan,
            evaluation_windows=comparison["evaluation_windows"],
            cache_prefix_lengths=tuple(row["length"] for row in prefixes),
        ),
        "GQA cache whole-job admission",
    )
    observations = report["observations"]
    if type(observations) is not list or len(observations) != len(plan.seeds) * len(
        plan.arms
    ) * len(prefixes):
        raise ValueError("GQA cache observation exposure differs")
    expected = [
        (trial, arm, config, prefix)
        for trial in comparison["trials"]
        for arm, config in zip(trial["arms"], plan.arms, strict=True)
        for prefix in prefixes
    ]
    prefix_hashes = {}
    for row, (trial, arm, config, prefix) in zip(observations, expected, strict=True):
        _fields(
            row,
            "seed kv_heads model_sha256 prefix full_execution steps",
            "GQA cache observation",
        )
        _equal(
            [row["seed"], row["kv_heads"], row["model_sha256"]],
            [trial["seed"], config.kv_heads, arm["final_model_sha256"]],
            "GQA cache final-model binding",
        )
        _fields(
            row["prefix"],
            "document_index document_sha256 length sha256",
            "GQA cache prefix",
        )
        _equal(
            {key: row["prefix"][key] for key in prefix},
            prefix,
            "GQA cache source prefix geometry",
        )
        _digest(row["prefix"]["sha256"])
        identity = prefix["document_index"]
        if (
            identity in prefix_hashes
            and prefix_hashes[identity] != row["prefix"]["sha256"]
        ):
            raise ValueError("GQA cache source prefix differs across treatments")
        prefix_hashes[identity] = row["prefix"]["sha256"]
        _equal(
            row["full_execution"],
            require_gqa_execution_budget(
                config, batch_size=1, sequence_length=prefix["length"]
            ),
            "GQA cache full-prefix score work",
        )
        if type(row["steps"]) is not list or len(row["steps"]) != prefix["length"]:
            raise ValueError("GQA cache growth omitted or duplicated prefix positions")
        for length, step in enumerate(row["steps"], 1):
            _fields(
                step,
                "length shape storage execution full_logits cached_logits parity",
                "GQA cache growth",
            )
            execution = require_gqa_execution_budget(
                config, batch_size=1, sequence_length=length, query_tokens=1
            )
            _equal(
                [step["length"], step["shape"], step["execution"]],
                [length, [1, config.kv_heads, length, config.gpt2.head_dim], execution],
                "GQA native cache geometry/score work",
            )
            _equal(
                step["storage"],
                {
                    "model_sha256": arm["final_model_sha256"],
                    "logical_bytes": execution["kv_cache_bytes"],
                    "retained_storage_bytes": execution["kv_cache_bytes"],
                    "unique_backing_stores": 2 * config.gpt2.layer_count,
                    "kv_heads": config.kv_heads,
                    "sequence_length": length,
                },
                "GQA actual native backing storage",
            )
            _equal(
                step["parity"],
                _parity(step["full_logits"], step["cached_logits"]),
                "GQA raw numerical cache parity",
            )
    _equal(report["summary"], _summarize(plan, observations), "GQA cache summary")
    return plan


def replay_gqa_cache_evidence(report, train_bytes, heldout_bytes):
    plan = verify_gqa_cache_evidence(report)
    actual = run_gqa_cache_evidence(
        report["source_report"],
        train_bytes,
        heldout_bytes,
        prefix_count=report["prefix_count"],
    )
    if canonical_moe_json(actual) != canonical_moe_json(report):
        raise ValueError("GQA cache evidence differs from actual source/model replay")
    return plan
