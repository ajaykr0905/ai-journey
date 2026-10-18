"""Length-normalized direct preference optimization for the tiny CPU decoder."""

from __future__ import annotations

from dataclasses import dataclass
import math
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from .preference_data import (
    PreferenceDataError,
    PreferencePair,
    candidate_targets,
)
from .transformer_lab import CharacterCodec, DecoderLanguageModel, model_fingerprint


def _finite_cpu_decoder(model):
    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be DecoderLanguageModel")
    if any(
        parameter.device.type != "cpu"
        or not parameter.is_floating_point()
        or not bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    ):
        raise PreferenceDataError(
            "preference scoring requires finite floating-point CPU parameters"
        )


def candidate_mean_logprob(
    model: DecoderLanguageModel,
    pair: PreferencePair,
    label: str,
    codec: CharacterCodec,
) -> torch.Tensor:
    """Return differentiable mean response log probability for one branch."""
    _finite_cpu_decoder(model)
    if not isinstance(codec, CharacterCodec):
        raise TypeError("codec must be CharacterCodec")
    if len(codec.tokens) != model.config.vocab_size:
        raise PreferenceDataError("codec vocabulary must match the decoder")
    targets = candidate_targets(pair, label, codec, block_size=model.config.block_size)
    values = []
    for width in range(1, model.config.block_size + 1):
        selected = [
            index
            for index, context in enumerate(targets.contexts)
            if len(context) == width
        ]
        if not selected:
            continue
        inputs = torch.tensor(
            [targets.contexts[index] for index in selected], dtype=torch.long
        )
        labels = torch.tensor(
            [targets.targets[index] for index in selected], dtype=torch.long
        )
        logits, _ = model(inputs)
        values.append(
            F.log_softmax(logits[:, -1], dim=-1).gather(1, labels[:, None]).squeeze(1)
        )
    if not values:
        raise PreferenceDataError("candidate has no scoreable response targets")
    score = torch.cat(values).mean()
    if not bool(torch.isfinite(score)):
        raise PreferenceDataError("candidate mean log probability must be finite")
    return score


def evaluate_candidate_logprob(model, pair, label, codec) -> float:
    """Score a branch without gradients and restore every caller module mode."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            value = float(candidate_mean_logprob(model, pair, label, codec))
    finally:
        for module, mode in modes:
            module.training = mode
    if not math.isfinite(value):
        raise PreferenceDataError("evaluated candidate score must be finite")
    return value


@dataclass(frozen=True)
class PreferenceObjective:
    loss: torch.Tensor
    policy_margin: torch.Tensor
    reference_margin: torch.Tensor
    reward_margin: torch.Tensor


def direct_preference_objective(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float,
) -> PreferenceObjective:
    """Compute a stable length-normalized DPO-style pair objective."""
    values = (
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
    )
    if any(
        not isinstance(value, torch.Tensor)
        or value.numel() != 1
        or not value.is_floating_point()
        or not bool(torch.isfinite(value))
        for value in values
    ):
        raise PreferenceDataError("preference objective requires finite scalar tensors")
    if len({value.device for value in values}) != 1:
        raise PreferenceDataError("preference objective tensors must share a device")
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(beta)
        or not 0 < beta <= 10
    ):
        raise PreferenceDataError("beta must be finite and in (0, 10]")
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen.detach() - reference_rejected.detach()
    reward_margin = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(reward_margin)
    if not bool(torch.isfinite(loss)):
        raise PreferenceDataError("preference loss must be finite")
    return PreferenceObjective(loss, policy_margin, reference_margin, reward_margin)


def _validate_batches(batches):
    if (
        not isinstance(batches, tuple)
        or not batches
        or any(
            not isinstance(batch, tuple)
            or not batch
            or any(not isinstance(pair, PreferencePair) for pair in batch)
            for batch in batches
        )
    ):
        raise PreferenceDataError("preference training requires nonempty pair batches")
    width = len(batches[0])
    if any(len(batch) != width for batch in batches):
        raise PreferenceDataError("preference training batches must be rectangular")


def train_preference_batches(
    policy: DecoderLanguageModel,
    reference: DecoderLanguageModel,
    batches: tuple[tuple[PreferencePair, ...], ...],
    codec: CharacterCodec,
    *,
    learning_rate: float,
    beta: float,
) -> dict:
    """Optimize exactly one fixed pair schedule against an immutable reference."""
    _finite_cpu_decoder(policy)
    _finite_cpu_decoder(reference)
    _validate_batches(batches)
    if policy is reference or {id(parameter) for parameter in policy.parameters()} & {
        id(parameter) for parameter in reference.parameters()
    }:
        raise PreferenceDataError(
            "policy and reference must own independent parameters"
        )
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise PreferenceDataError("reference parameters must be frozen")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 0.1
    ):
        raise PreferenceDataError("learning_rate must be finite and in (0, 0.1]")
    trainable = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    optimizer = (
        torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
        if trainable
        else None
    )
    policy_modes = [(module, module.training) for module in policy.modules()]
    reference_modes = [(module, module.training) for module in reference.modules()]
    reference_before = model_fingerprint(reference)
    trace = []
    try:
        policy.train()
        reference.eval()
        for batch in batches:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            losses = []
            margins = []
            for pair in batch:
                with torch.no_grad():
                    reference_chosen = candidate_mean_logprob(
                        reference, pair, "chosen", codec
                    )
                    reference_rejected = candidate_mean_logprob(
                        reference, pair, "rejected", codec
                    )
                with nullcontext() if optimizer is not None else torch.no_grad():
                    policy_chosen = candidate_mean_logprob(
                        policy, pair, "chosen", codec
                    )
                    policy_rejected = candidate_mean_logprob(
                        policy, pair, "rejected", codec
                    )
                    objective = direct_preference_objective(
                        policy_chosen,
                        policy_rejected,
                        reference_chosen,
                        reference_rejected,
                        beta=beta,
                    )
                losses.append(objective.loss)
                margins.append(
                    (
                        float(objective.policy_margin.detach()),
                        float(objective.reference_margin.detach()),
                        float(objective.reward_margin.detach()),
                    )
                )
            loss = torch.stack(losses).mean()
            if optimizer is not None:
                loss.backward()
                if any(
                    parameter.grad is None
                    or not bool(torch.isfinite(parameter.grad).all())
                    for parameter in trainable
                ):
                    raise PreferenceDataError(
                        "trainable preference gradients must be present and finite"
                    )
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable, 1.0, error_if_nonfinite=True
                    )
                )
                optimizer.step()
                _finite_cpu_decoder(policy)
            else:
                gradient_norm = 0.0
            trace.append(
                {
                    "loss": float(loss.detach()),
                    "policy_margin": math.fsum(row[0] for row in margins)
                    / len(margins),
                    "reference_margin": math.fsum(row[1] for row in margins)
                    / len(margins),
                    "reward_margin": math.fsum(row[2] for row in margins)
                    / len(margins),
                    "preference_accuracy": sum(row[0] > 0 for row in margins)
                    / len(margins),
                    "gradient_norm": gradient_norm,
                }
            )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
    finally:
        for module, mode in policy_modes:
            module.training = mode
        for module, mode in reference_modes:
            module.training = mode
    reference_after = model_fingerprint(reference)
    if reference_after != reference_before:
        raise PreferenceDataError("reference model changed during preference training")
    return {
        "steps": len(batches),
        "comparisons": sum(len(batch) for batch in batches),
        "optimizer_steps": len(batches) if optimizer is not None else 0,
        "reference_before_sha256": reference_before,
        "reference_after_sha256": reference_after,
        "trace": trace,
    }
