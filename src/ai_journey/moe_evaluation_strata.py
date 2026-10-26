"""Token-aligned loss diagnostics for explicitly capacity-limited MoE probes."""

import math

import torch
from torch.nn import functional as F

STRATA_SCOPE = (
    "descriptive token loss grouped by accepted training-capacity routes; "
    "not a causal estimate or normal dropless inference score"
)


def summarize_moe_token_losses(logits, targets, accepted_routes, token_mask):
    """Conserve valid target exposure and NLL across zero/one/two-route strata.

    ``accepted_routes`` is a detached int64 [batch,time] count for one layer.
    Padded positions must have zero accepted routes and contribute no loss.
    No gradients, parameters, RNG state or caller tensors are modified.
    """

    tensors = (logits, targets, accepted_routes, token_mask)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("MoE loss strata require tensor inputs")
    if any(value.device.type != "cpu" for value in tensors):
        raise ValueError("MoE loss strata require CPU tensors")
    if (
        logits.ndim != 3
        or not logits.is_floating_point()
        or targets.dtype != torch.long
        or targets.shape != logits.shape[:2]
        or accepted_routes.dtype != torch.long
        or accepted_routes.shape != targets.shape
        or token_mask.dtype != torch.bool
        or token_mask.shape != targets.shape
        or not 2 <= logits.shape[-1] <= 65536
        or not 1 <= targets.numel() <= 8192
        or logits.numel() > 8_000_000
    ):
        raise ValueError("invalid bounded token-aligned MoE loss inputs")
    if (
        not bool(torch.isfinite(logits).all())
        or not bool(token_mask.any())
        or bool((targets < 0).any())
        or bool((targets >= logits.shape[-1]).any())
        or bool((accepted_routes < 0).any())
        or bool((accepted_routes > 2).any())
        or bool((accepted_routes[~token_mask] != 0).any())
    ):
        raise ValueError("invalid finite MoE losses, targets or accepted-route counts")
    with torch.no_grad():
        losses = F.cross_entropy(
            logits.detach().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
    if not bool(torch.isfinite(losses[token_mask]).all()):
        raise FloatingPointError("MoE token loss is non-finite")
    rows = []
    for count in range(3):
        selected = losses[token_mask & (accepted_routes == count)].double().tolist()
        total = math.fsum(selected)
        rows.append(
            {
                "accepted_routes": count,
                "target_tokens": len(selected),
                "nll_sum": total,
                "mean_nll": total / len(selected) if selected else None,
            }
        )
    all_losses = losses[token_mask].double().tolist()
    return {
        "scope": STRATA_SCOPE,
        "target_tokens": len(all_losses),
        "nll_sum": math.fsum(all_losses),
        "strata": rows,
    }
