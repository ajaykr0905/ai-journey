"""Complete document likelihood with explicit dropless MoE routing evidence."""

from hashlib import sha256
import json
import math

import torch
from torch.nn import functional as F

from .ddp_evaluation import evaluation_batches
from .decoder_variant_runtime import isolated_decoder_runtime
from .moe_config import MoEConfig
from .moe_route_audit import validate_moe_routing


def summarize_moe_routing_batches(batches, *, layers, expert_count):
    """Aggregate aligned per-layer assignment counts without averaging fractions."""

    summaries = []
    for index in layers:
        rows = [
            row
            for batch in batches
            for row in batch["routing"]
            if row["layer_index"] == index
        ]
        summaries.append(
            {
                "layer_index": index,
                "valid_tokens": sum(row["valid_tokens"] for row in rows),
                **{
                    name: [
                        sum(row[name][expert] for row in rows)
                        for expert in range(expert_count)
                    ]
                    for name in (
                        "attempted_counts",
                        "accepted_counts",
                        "dropped_counts",
                    )
                },
                "probability_sums": [
                    math.fsum(row["probability_sums"][expert] for row in rows)
                    for expert in range(expert_count)
                ],
            }
        )
    return summaries


@isolated_decoder_runtime()
def evaluate_moe_documents(model, documents, *, batch_size=4):
    """Score every target once, retaining raw losses and the routing workload.

    Normal evaluation is explicitly dropless. Training-capacity probes use a
    separate function and may depend on the token grouping of their workload.
    """

    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("MoE document evaluation batch_size must be in [1, 16]")
    documents = tuple(documents)
    if not 1 <= len(documents) <= 2048 or len({doc.digest for doc in documents}) != len(
        documents
    ):
        raise ValueError("MoE evaluation requires distinct bounded documents")
    target_count = sum(len(doc.content) - 1 for doc in documents)
    if not 1 <= target_count <= 8192:
        raise ValueError("MoE evaluation requires one to 8192 targets")
    if any(
        parameter.device.type != "cpu" or parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise ValueError("MoE evaluation requires a float32 CPU model")
    config = model.config
    sparse = isinstance(config, MoEConfig)
    base = config.gpt2
    if (
        base.vocab_size != 256
        or not 2 <= base.context_length <= 32
        or not 4 <= base.embedding_dim <= 32
        or not 1 <= base.layer_count <= 2
        or (sparse and config.expert_count > 8)
    ):
        raise ValueError("MoE evaluation exceeds the bounded byte-decoder protocol")
    modes = [(module, module.training) for module in model.modules()]
    batches, losses = [], []
    schedule = sha256()
    try:
        model.eval()
        with torch.inference_mode():
            for inputs, targets, mask in evaluation_batches(
                documents, context_length=base.context_length, batch_size=batch_size
            ):
                if sparse:
                    result = model.forward_training(inputs, token_mask=mask)
                    logits = result.logits
                    routing = json.loads(json.dumps(result.routing, allow_nan=False))
                    if [row["layer_index"] for row in routing] != list(
                        config.moe_layers
                    ):
                        raise ValueError(
                            "MoE evaluation routing layers differ from the model"
                        )
                    for row in routing:
                        validate_moe_routing(
                            row, expert_count=config.expert_count, top_k=2
                        )
                        if row["capacity_applied"] is not False or row[
                            "valid_tokens"
                        ] != int(mask.sum()):
                            raise ValueError(
                                "normal MoE evaluation must be dropless and exclude padding"
                            )
                        if (
                            row["token_indices"]
                            != mask.flatten().nonzero().flatten().tolist()
                            or len(row["accepted_per_token"]) != inputs.shape[0]
                            or any(
                                len(values) != inputs.shape[1]
                                for values in row["accepted_per_token"]
                            )
                        ):
                            raise ValueError(
                                "MoE routes do not align with the actual token mask"
                            )
                else:
                    logits, routing = model(inputs), []
                if logits.shape != (*targets.shape, base.vocab_size):
                    raise ValueError("MoE evaluation logits do not match the targets")
                nll = (
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="none",
                    )
                    .reshape_as(targets)[mask]
                    .double()
                )
                if not bool(torch.isfinite(nll).all()):
                    raise FloatingPointError("MoE evaluation loss is non-finite")
                selected = nll.tolist()
                losses.extend(selected)
                schedule.update(
                    json.dumps(
                        {
                            "inputs": inputs.tolist(),
                            "targets": targets.tolist(),
                            "mask": mask.tolist(),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                )
                batches.append(
                    {
                        "shape": list(inputs.shape),
                        "token_mask": mask.tolist(),
                        "token_nll": selected,
                        "routing": routing,
                    }
                )
    finally:
        for module, training in modes:
            module.training = training
    if len(losses) != target_count:
        raise RuntimeError("MoE evaluation omitted or duplicated a document target")
    total = math.fsum(losses)
    mean = total / target_count
    return {
        "capacity_policy": "dropless_eval",
        "target_tokens": target_count,
        "nll_sum": total,
        "mean_nll": mean,
        "bits_per_byte": mean / math.log(2),
        "perplexity": math.exp(mean) if mean < 700 else None,
        "batch_schedule_sha256": schedule.hexdigest(),
        "batches": batches,
        "routing_summary": summarize_moe_routing_batches(
            batches, layers=config.moe_layers, expert_count=config.expert_count
        )
        if sparse
        else [],
    }
