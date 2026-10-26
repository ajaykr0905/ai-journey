"""Bounded, once-per-target likelihood for opt-in decoder documents."""

import math

import torch
from torch.nn import functional as F

from .ddp_evaluation import evaluation_batches
from .decoder_variant_runtime import isolated_decoder_runtime


@isolated_decoder_runtime()
def evaluate_decoder_documents(model, documents, *, batch_size: int = 16) -> dict:
    """Score every within-document target once, preserving caller model state."""

    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("evaluation batch_size must be an integer in [1, 32]")
    documents = tuple(documents)
    if not documents or len({doc.digest for doc in documents}) != len(documents):
        raise ValueError("evaluation documents must be nonempty and content-distinct")
    target_count = sum(len(doc.content) - 1 for doc in documents)
    if not 1 <= target_count <= 8192:
        raise ValueError("evaluation requires between 1 and 8192 target tokens")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("decoder document evaluation requires a CPU model")
    modes = [(module, module.training) for module in model.modules()]
    losses = []
    tokens = 0
    try:
        model.eval()
        with torch.inference_mode():
            for inputs, targets, mask in evaluation_batches(
                documents,
                context_length=model.config.gpt2.context_length,
                batch_size=batch_size,
            ):
                logits = model(inputs)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
                selected = nll[mask].double()
                if not bool(torch.isfinite(selected).all()):
                    raise FloatingPointError(
                        "document evaluation produced non-finite loss"
                    )
                losses.extend(selected.tolist())
                tokens += int(mask.sum())
    finally:
        for module, training in modes:
            module.training = training
    if tokens != target_count:
        raise RuntimeError("document evaluation did not cover each target exactly once")
    total = math.fsum(losses)
    mean = total / tokens
    return {
        "target_tokens": tokens,
        "nll_sum": total,
        "mean_nll": mean,
        "bits_per_byte": mean / math.log(2),
        "perplexity": math.exp(mean) if mean < 700 else None,
    }
