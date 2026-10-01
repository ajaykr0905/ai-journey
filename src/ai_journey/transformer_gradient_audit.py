"""Propagate manually derived loss gradients through the existing transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .cross_entropy import manual_cross_entropy
from .transformer_lab import DecoderLanguageModel, TransformerConfig


@dataclass(frozen=True)
class TransformerGradientAudit:
    seed: int
    loss_error: float
    parameter_errors: dict[str, float]
    tolerance: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def audit_transformer_loss_gradient(
    *, seed: int = 28, tolerance: float = 1e-10
) -> TransformerGradientAudit:
    """Check every parameter using manual dlogits versus native loss autograd.

    Both paths use PyTorch to differentiate the transformer. Only the categorical
    loss derivative is handwritten. A fixed synthetic token batch, CPU float64,
    and RNG isolation make this a small reproducible diagnostic, not training or
    evidence of GPU/distributed performance.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative and finite")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(seed)
        model = (
            DecoderLanguageModel(
                TransformerConfig(
                    vocab_size=7,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                    dropout=0.0,
                )
            )
            .double()
            .eval()
        )
        inputs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]], dtype=torch.long)
        targets = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 1]], dtype=torch.long)
        logits, native_loss = model(inputs, targets)
        trace = manual_cross_entropy(
            logits.detach().numpy().reshape(-1, 7), targets.numpy().reshape(-1)
        )
        named = list(model.named_parameters())
        parameters = tuple(parameter for _, parameter in named)
        reference = torch.autograd.grad(native_loss, parameters, retain_graph=True)
        supplied = torch.from_numpy(trace.dlogits).reshape_as(logits)
        manual = torch.autograd.grad(logits, parameters, grad_outputs=supplied)
        errors = {
            name: float(torch.max(torch.abs(actual - expected)))
            for (name, _), actual, expected in zip(
                named, manual, reference, strict=True
            )
        }
        loss_error = abs(trace.loss - float(native_loss.detach()))
    passed = all(
        np.isfinite(error) and error <= tolerance
        for error in (loss_error, *errors.values())
    )
    return TransformerGradientAudit(seed, loss_error, errors, tolerance, bool(passed))
