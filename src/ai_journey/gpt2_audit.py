"""Deterministic parity audits for imported GPT-2 logits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import torch
from torch import Tensor

from .gpt2_model import GPT2LanguageModel


class GPT2AuditError(ValueError):
    """Raised when a GPT-2 parity probe is invalid."""


@dataclass(frozen=True)
class LogitParityResult:
    """Bounded comparison between independent and reference GPT-2 logits."""

    token_ids: tuple[int, ...]
    compared_values: int
    max_absolute_error: float
    mean_absolute_error: float
    top1_matches: int
    top1_total: int
    reference_digest: str
    candidate_digest: str


def _tensor_digest(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = sha256(metadata)
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def audit_logit_parity(
    model: GPT2LanguageModel,
    token_ids: Tensor,
    reference_logits: Tensor,
) -> LogitParityResult:
    """Compare all logits for one bounded token probe."""

    if not isinstance(model, GPT2LanguageModel):
        raise TypeError("model must be GPT2LanguageModel")
    if (
        not isinstance(token_ids, Tensor)
        or token_ids.dtype != torch.long
        or token_ids.ndim != 2
        or token_ids.shape[0] != 1
        or token_ids.numel() == 0
    ):
        raise TypeError(
            "token_ids must be a non-empty rank-two single-batch long tensor"
        )
    if (
        not isinstance(reference_logits, Tensor)
        or not reference_logits.is_floating_point()
    ):
        raise TypeError("reference_logits must be a floating-point tensor")
    if reference_logits.device.type != "cpu" or token_ids.device.type != "cpu":
        raise GPT2AuditError("parity audit inputs must be CPU tensors")
    if not bool(torch.isfinite(reference_logits).all()):
        raise GPT2AuditError("reference_logits must contain only finite values")

    was_training = model.training
    model.eval()
    try:
        candidate = model(token_ids)
    finally:
        model.train(was_training)
    if candidate.shape != reference_logits.shape:
        raise GPT2AuditError(
            f"reference logits shape {tuple(reference_logits.shape)} does not match "
            f"candidate shape {tuple(candidate.shape)}"
        )
    if not bool(torch.isfinite(candidate).all()):
        raise GPT2AuditError("candidate logits must contain only finite values")
    differences = (candidate.double() - reference_logits.double()).abs()
    max_error = float(differences.max())
    mean_error = float(differences.mean())
    if not math.isfinite(max_error) or not math.isfinite(mean_error):
        raise GPT2AuditError("parity metrics must be finite")
    candidate_top1 = candidate.argmax(dim=-1)
    reference_top1 = reference_logits.argmax(dim=-1)
    return LogitParityResult(
        token_ids=tuple(int(token) for token in token_ids[0]),
        compared_values=differences.numel(),
        max_absolute_error=max_error,
        mean_absolute_error=mean_error,
        top1_matches=int((candidate_top1 == reference_top1).sum()),
        top1_total=reference_top1.numel(),
        reference_digest=_tensor_digest(reference_logits),
        candidate_digest=_tensor_digest(candidate),
    )
