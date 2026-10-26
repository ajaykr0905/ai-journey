"""Once-per-target likelihood under the serving lab's truncated-prefix policy."""

from itertools import islice
import json
import math
import re

import torch
from torch.nn import functional as F

from .ddp_evaluation import Document
from .decoder_variant_comparison import decoder_comparison_digest
from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_runtime import isolated_decoder_runtime
from .moe_config import MoEConfig
from .moe_evaluation import summarize_moe_routing_batches
from .moe_report import canonical_moe_json
from .moe_resources import require_moe_job_budget
from .moe_route_audit import validate_moe_routing
from .training_models import (
    model_config_from_mapping,
    model_config_mapping,
    training_model_fingerprint,
)

ROLLING_SCHEMA = "ai-journey-moe-rolling-context-v1"
ROLLING_POLICY = "last T prefix bytes; reset positions within each truncated window; score only its final position"
ROLLING_SCOPE = "internal consistency, not measurement authentication; actual source and model replay is a separate check; no GPU or performance claim"


def _bounded_config(config):
    if type(config) not in (DecoderVariantConfig, MoEConfig):
        raise ValueError("rolling evaluation requires a dense or MoE decoder config")
    base = config.gpt2
    if (
        base.vocab_size != 256
        or not 2 <= base.context_length <= 32
        or not 4 <= base.embedding_dim <= 32
        or not 1 <= base.layer_count <= 2
        or (isinstance(config, MoEConfig) and config.expert_count > 8)
    ):
        raise ValueError("rolling evaluation exceeds the bounded byte-decoder protocol")


def _admission(config, target_tokens, batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("rolling evaluation batch_size must be in [1, 16]")
    if type(target_tokens) is not int or not 1 <= target_tokens <= 8192:
        raise ValueError("rolling evaluation requires one to 8192 targets")
    # Conservatively reuse the full-job gate: one rolling prefix per target,
    # not ceil(targets / context). Its extra training/two-pass charges are an
    # admission upper bound, never measured evaluation FLOPs or timing.
    return require_moe_job_budget(
        (config,),
        seeds=1,
        steps=1,
        batch_size=1,
        evaluation_windows=target_tokens,
        evaluation_batch_size=batch_size,
    )


def _document_evidence(documents):
    if not 1 <= len(documents) <= 2048 or any(
        not isinstance(document, Document) for document in documents
    ):
        raise ValueError("rolling evaluation requires bounded Document values")
    evidence = [{"sha256": doc.digest, "bytes": len(doc.content)} for doc in documents]
    if len({row["sha256"] for row in evidence}) != len(evidence):
        raise ValueError("rolling evaluation documents must be content-distinct")
    return evidence


def _rows(lengths, context):
    for index, length in enumerate(lengths):
        for offset in range(1, length):
            start = max(0, offset - context)
            yield {
                "document_index": index,
                "target_offset": offset,
                "context_start": start,
                "context_tokens": offset - start,
            }


def _routing(routing, config, mask):
    layers = list(config.moe_layers) if isinstance(config, MoEConfig) else []
    if (
        type(routing) is not list
        or any(type(row) is not dict for row in routing)
        or [row.get("layer_index") for row in routing] != layers
    ):
        raise ValueError("rolling routing layers differ from the architecture")
    indices = [index for index, valid in enumerate(sum(mask, [])) if valid]
    for row in routing:
        validate_moe_routing(row, expert_count=config.expert_count)
        if (
            row["capacity_applied"] is not False
            or row["valid_tokens"] != len(indices)
            or row["token_indices"] != indices
            or len(row["accepted_per_token"]) != len(mask)
            or any(len(values) != len(mask[0]) for values in row["accepted_per_token"])
        ):
            raise ValueError("rolling routing must be dropless on the full prefix mask")


@isolated_decoder_runtime()
def evaluate_moe_rolling_documents(model, documents, *, batch_size=4):
    """Score every document target once using its own last-T truncated prefix.

    Unlike nonoverlapping-window evaluation, a window boundary never discards
    available previous bytes. Positions reset when the prefix is truncated,
    matching full-context rebuild serving, not an indefinitely growing KV cache.
    Context routing exposure includes all prefix tokens, while loss exposure is
    exactly one final position per prefix. Padding is excluded from routing.
    """

    config = model.config
    _bounded_config(config)
    documents = tuple(islice(iter(documents), 2049))
    evidence = _document_evidence(documents)
    targets = sum(row["bytes"] - 1 for row in evidence)
    budget = _admission(config, targets, batch_size)
    if any(
        parameter.device.type != "cpu" or parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise ValueError("rolling evaluation requires a float32 CPU model")
    modes = [(module, module.training) for module in model.modules()]
    batches, losses = [], []
    iterator = iter(
        _rows([row["bytes"] for row in evidence], config.gpt2.context_length)
    )
    try:
        model.eval()
        with torch.inference_mode():
            while rows := list(islice(iterator, batch_size)):
                width = max(row["context_tokens"] for row in rows)
                inputs = torch.zeros((len(rows), width), dtype=torch.long)
                mask = torch.zeros_like(inputs, dtype=torch.bool)
                target = []
                for index, row in enumerate(rows):
                    document = documents[row["document_index"]].content
                    prefix = document[row["context_start"] : row["target_offset"]]
                    inputs[index, : len(prefix)] = torch.tensor(list(prefix))
                    mask[index, : len(prefix)] = True
                    target.append(document[row["target_offset"]])
                if isinstance(config, MoEConfig):
                    output = model.forward_training(inputs, token_mask=mask)
                    logits, routing = (
                        output.logits,
                        json.loads(json.dumps(output.routing)),
                    )
                else:
                    logits, routing = model(inputs), []
                _routing(routing, config, mask.tolist())
                if logits.shape != (*inputs.shape, 256):
                    raise ValueError("rolling logits do not match the prefix batch")
                selected = logits[
                    torch.arange(len(rows)),
                    torch.tensor([row["context_tokens"] - 1 for row in rows]),
                ]
                nll = F.cross_entropy(
                    selected, torch.tensor(target), reduction="none"
                ).double()
                if not bool(torch.isfinite(nll).all()):
                    raise FloatingPointError("rolling target loss is non-finite")
                losses.extend(nll.tolist())
                batches.append(
                    {
                        "rows": rows,
                        "shape": list(inputs.shape),
                        "context_mask": mask.tolist(),
                        "target_nll": nll.tolist(),
                        "routing": routing,
                    }
                )
    finally:
        for module, mode in modes:
            module.training = mode
    total = math.fsum(losses)
    mean = total / targets
    report = {
        "schema": ROLLING_SCHEMA,
        "architecture": model_config_mapping(config),
        "model_sha256": training_model_fingerprint(model),
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch_version": str(torch.__version__),
        },
        "policy": ROLLING_POLICY,
        "documents": evidence,
        "batch_size": batch_size,
        "resource_admission": budget,
        "target_tokens": targets,
        "context_tokens": sum(
            row["context_tokens"] for batch in batches for row in batch["rows"]
        ),
        "nll_sum": total,
        "mean_nll": mean,
        "bits_per_byte": mean / math.log(2),
        "perplexity": math.exp(mean) if mean < 700 else None,
        "batches": batches,
        "routing_summary": summarize_moe_routing_batches(
            batches, layers=config.moe_layers, expert_count=config.expert_count
        )
        if isinstance(config, MoEConfig)
        else [],
        "verification_scope": ROLLING_SCOPE,
    }
    report["report_sha256"] = decoder_comparison_digest(report)
    verify_moe_rolling_report(report)
    return report


def verify_moe_rolling_report(report):
    """Recompute coverage, mask/route work and additive scores without a model."""

    fields = {
        "schema",
        "architecture",
        "model_sha256",
        "runtime",
        "policy",
        "documents",
        "batch_size",
        "resource_admission",
        "target_tokens",
        "context_tokens",
        "nll_sum",
        "mean_nll",
        "bits_per_byte",
        "perplexity",
        "batches",
        "routing_summary",
        "verification_scope",
        "report_sha256",
    }
    if (
        type(report) is not dict
        or set(report) != fields
        or report["schema"] != ROLLING_SCHEMA
    ):
        raise ValueError("rolling report fields/schema differ")
    if report["report_sha256"] != decoder_comparison_digest(report):
        raise ValueError("rolling report seal differs")
    config = model_config_from_mapping(report["architecture"])
    _bounded_config(config)
    runtime = report["runtime"]
    if (
        type(runtime) is not dict
        or set(runtime) != {"device", "dtype", "threads", "torch_version"}
        or canonical_moe_json([runtime["device"], runtime["dtype"], runtime["threads"]])
        != canonical_moe_json(["cpu", "float32", 1])
        or type(runtime["torch_version"]) is not str
        or not 1 <= len(runtime["torch_version"]) <= 100
        or report["policy"] != ROLLING_POLICY
        or report["verification_scope"] != ROLLING_SCOPE
        or type(report["model_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", report["model_sha256"]) is None
    ):
        raise ValueError("rolling runtime, policy or model identity differs")
    evidence = report["documents"]
    if type(evidence) is not list or not 1 <= len(evidence) <= 2048:
        raise ValueError("rolling document evidence is unbounded")
    for row in evidence:
        if (
            type(row) is not dict
            or set(row) != {"sha256", "bytes"}
            or type(row["bytes"]) is not int
            or not 2 <= row["bytes"] <= 8193
            or type(row["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
        ):
            raise ValueError("rolling document identity/length is invalid")
    if len({row["sha256"] for row in evidence}) != len(evidence):
        raise ValueError("rolling document identities are not distinct")
    target_count = sum(row["bytes"] - 1 for row in evidence)
    budget = _admission(config, target_count, report["batch_size"])
    if canonical_moe_json(report["resource_admission"]) != canonical_moe_json(budget):
        raise ValueError("rolling admission differs from the full prefix workload")
    iterator = iter(
        _rows([row["bytes"] for row in evidence], config.gpt2.context_length)
    )
    losses, context_tokens = [], 0
    batches = report["batches"]
    if type(batches) is not list or len(batches) != math.ceil(
        target_count / report["batch_size"]
    ):
        raise ValueError("rolling batch count differs from target coverage")
    for batch in batches:
        rows = list(islice(iterator, report["batch_size"]))
        width = max(row["context_tokens"] for row in rows)
        mask = [
            [index < row["context_tokens"] for index in range(width)] for row in rows
        ]
        if (
            type(batch) is not dict
            or set(batch) != {"rows", "shape", "context_mask", "target_nll", "routing"}
            or canonical_moe_json(
                [batch["rows"], batch["shape"], batch["context_mask"]]
            )
            != canonical_moe_json([rows, [len(rows), width], mask])
        ):
            raise ValueError("rolling rows, score positions or context mask differ")
        values = batch["target_nll"]
        if (
            type(values) is not list
            or len(values) != len(rows)
            or any(
                type(value) is not float or not math.isfinite(value) or value < 0
                for value in values
            )
        ):
            raise ValueError("rolling raw target losses differ")
        _routing(batch["routing"], config, mask)
        context_tokens += sum(row["context_tokens"] for row in rows)
        losses.extend(values)
    total, mean = math.fsum(losses), math.fsum(losses) / target_count
    expected = [
        target_count,
        context_tokens,
        total,
        mean,
        mean / math.log(2),
        math.exp(mean) if mean < 700 else None,
    ]
    actual = [
        report[key]
        for key in (
            "target_tokens",
            "context_tokens",
            "nll_sum",
            "mean_nll",
            "bits_per_byte",
            "perplexity",
        )
    ]
    summary = (
        summarize_moe_routing_batches(
            batches, layers=config.moe_layers, expert_count=config.expert_count
        )
        if isinstance(config, MoEConfig)
        else []
    )
    if canonical_moe_json(actual) != canonical_moe_json(expected) or canonical_moe_json(
        report["routing_summary"]
    ) != canonical_moe_json(summary):
        raise ValueError("rolling additive loss or routing summary differs")
    return config


def replay_moe_rolling_report(report, model, documents):
    """Rerun every recorded prefix against the actual model and document bytes."""

    config = verify_moe_rolling_report(report)
    documents = tuple(islice(iter(documents), 2049))
    if (
        report["runtime"]["torch_version"] != str(torch.__version__)
        or canonical_moe_json(model_config_mapping(model.config))
        != canonical_moe_json(model_config_mapping(config))
        or training_model_fingerprint(model) != report["model_sha256"]
        or canonical_moe_json(_document_evidence(documents))
        != canonical_moe_json(report["documents"])
    ):
        raise ValueError(
            "rolling replay requires the recorded CPU runtime, model and source"
        )
    actual = evaluate_moe_rolling_documents(
        model, documents, batch_size=report["batch_size"]
    )
    if canonical_moe_json(actual) != canonical_moe_json(report):
        raise ValueError("rolling replay observations differ from actual execution")
    return config
