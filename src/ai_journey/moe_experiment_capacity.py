"""Fixed-weight training-capacity probes, explicitly separate from inference."""

from hashlib import sha256
import json
import math

import torch

from .ddp_evaluation import evaluation_batches
from .decoder_variant_runtime import isolated_decoder_runtime
from .moe_config import MoEConfig
from .moe_evaluation import evaluate_moe_documents, summarize_moe_routing_batches
from .moe_evaluation_strata import STRATA_SCOPE, summarize_moe_token_losses
from .moe_resources import require_moe_job_budget
from .moe_route_audit import validate_moe_routing
from .training_models import training_model_fingerprint

CAPACITY_SCOPE = (
    "fixed-weight zero-dropout training-mode capacity probe; batch grouping may "
    "change admission and loss; not normal dropless inference or a performance benchmark"
)


def _combine_strata(rows):
    strata = []
    for count in range(3):
        selected = [row["strata"][count] for row in rows]
        tokens = sum(row["target_tokens"] for row in selected)
        total = math.fsum(row["nll_sum"] for row in selected)
        strata.append(
            {
                "accepted_routes": count,
                "target_tokens": tokens,
                "nll_sum": total,
                "mean_nll": total / tokens if tokens else None,
            }
        )
    return {
        "scope": STRATA_SCOPE,
        "target_tokens": sum(row["target_tokens"] for row in rows),
        "nll_sum": math.fsum(row["nll_sum"] for row in rows),
        "strata": strata,
    }


@isolated_decoder_runtime()
def probe_moe_training_capacity(model, documents, *, batch_sizes=(1, 4)):
    """Measure actual route drops with unchanged weights and declared grouping."""

    config = model.config
    if not isinstance(config, MoEConfig) or any(
        (
            config.gpt2.embedding_dropout,
            config.gpt2.attention_dropout,
            config.gpt2.residual_dropout,
        )
    ):
        raise ValueError("capacity probes require a zero-dropout MoE model")
    if (
        type(batch_sizes) is not tuple
        or not 1 <= len(batch_sizes) <= 4
        or len(set(batch_sizes)) != len(batch_sizes)
        or any(type(value) is not int or not 1 <= value <= 8 for value in batch_sizes)
    ):
        raise ValueError(
            "capacity probe requires one to four distinct batch sizes in [1, 8]"
        )
    documents = tuple(documents)
    targets = sum(len(doc.content) - 1 for doc in documents)
    if not 1 <= targets <= 8192 or len({doc.digest for doc in documents}) != len(
        documents
    ):
        raise ValueError("capacity probes require distinct bounded public documents")
    windows = sum(
        math.ceil((len(doc.content) - 1) / config.gpt2.context_length)
        for doc in documents
    )
    budget = require_moe_job_budget(
        (config,),
        seeds=len(batch_sizes) + 1,
        steps=1,
        batch_size=1,
        evaluation_windows=windows,
        evaluation_batch_size=max(batch_sizes),
    )
    identity = training_model_fingerprint(model)
    dropless = evaluate_moe_documents(model, documents, batch_size=max(batch_sizes))
    modes = [(module, module.training) for module in model.modules()]
    cases = []
    try:
        model.train()
        with torch.inference_mode():
            for batch_size in batch_sizes:
                batches, losses = [], []
                strata = {layer: [] for layer in config.moe_layers}
                schedule = sha256()
                for inputs, labels, mask in evaluation_batches(
                    documents,
                    context_length=config.gpt2.context_length,
                    batch_size=batch_size,
                ):
                    result = model.forward_training(inputs, token_mask=mask)
                    routing = json.loads(json.dumps(result.routing, allow_nan=False))
                    if [row["layer_index"] for row in routing] != list(
                        config.moe_layers
                    ):
                        raise ValueError("capacity probe routing layers differ")
                    for row in routing:
                        validate_moe_routing(row, expert_count=config.expert_count)
                        if (
                            row["capacity_applied"] is not True
                            or row["token_indices"]
                            != mask.flatten().nonzero().flatten().tolist()
                            or row["capacity"]
                            != max(
                                config.min_capacity,
                                math.ceil(
                                    config.capacity_factor
                                    * int(mask.sum())
                                    * 2
                                    / config.expert_count
                                ),
                            )
                        ):
                            raise ValueError(
                                "capacity probe did not apply the masked training policy"
                            )
                        layer_losses = summarize_moe_token_losses(
                            result.logits,
                            labels,
                            torch.tensor(row["accepted_per_token"], dtype=torch.long),
                            mask,
                        )
                        strata[row["layer_index"]].append(layer_losses)
                    selected = (
                        torch.nn.functional.cross_entropy(
                            result.logits.reshape(-1, config.gpt2.vocab_size),
                            labels.reshape(-1),
                            reduction="none",
                        )
                        .reshape_as(labels)[mask]
                        .double()
                        .tolist()
                    )
                    if any(not math.isfinite(value) for value in selected):
                        raise FloatingPointError("capacity probe loss is non-finite")
                    losses.extend(selected)
                    schedule.update(
                        json.dumps(
                            {"inputs": inputs.tolist(), "mask": mask.tolist()},
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
                if len(losses) != targets:
                    raise RuntimeError("capacity probe omitted a target")
                cases.append(
                    {
                        "batch_size": batch_size,
                        "target_tokens": targets,
                        "nll_sum": math.fsum(losses),
                        "mean_nll": math.fsum(losses) / targets,
                        "delta_from_dropless_nll": math.fsum(losses) / targets
                        - dropless["mean_nll"],
                        "batch_schedule_sha256": schedule.hexdigest(),
                        "batches": batches,
                        "routing_summary": summarize_moe_routing_batches(
                            batches,
                            layers=config.moe_layers,
                            expert_count=config.expert_count,
                        ),
                        "loss_strata": [
                            {"layer_index": layer, **_combine_strata(strata[layer])}
                            for layer in config.moe_layers
                        ],
                    }
                )
    finally:
        for module, training in modes:
            module.training = training
    if training_model_fingerprint(model) != identity:
        raise RuntimeError("capacity probe changed model state")
    return {
        "scope": CAPACITY_SCOPE,
        "model_sha256": identity,
        "documents": [doc.digest for doc in documents],
        "resource_admission": budget,
        "dropless": dropless,
        "cases": cases,
    }
