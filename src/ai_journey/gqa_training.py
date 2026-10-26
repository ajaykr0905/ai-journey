"""Explicit grouped-query CPU updates reuse the audited transactional protocol."""

from .gqa_model import GroupedQueryModel
from .moe_training import build_moe_optimizer, train_moe_step


def build_gqa_optimizer(model, *, learning_rate, config=None, weight_decay=0.01):
    if not isinstance(model, GroupedQueryModel):
        raise TypeError("model must be GroupedQueryModel")
    return build_moe_optimizer(
        model, learning_rate=learning_rate, config=config, weight_decay=weight_decay
    )


def train_gqa_step(
    model,
    optimizer,
    inputs,
    targets,
    *,
    token_mask=None,
    balance_weight=0.01,
    z_weight=0.0,
    max_gradient_norm=1.0,
):
    """Train dense or expert GQA with identical controls and no kernel-speed claim.

    Dense models have exact zero auxiliary terms; expert models retain the
    explicit balance/z objective and idle-expert dense AdamW semantics.
    """
    if not isinstance(model, GroupedQueryModel):
        raise TypeError("model must be GroupedQueryModel")
    return train_moe_step(
        model,
        optimizer,
        inputs,
        targets,
        token_mask=token_mask,
        balance_weight=balance_weight,
        z_weight=z_weight,
        max_gradient_norm=max_gradient_norm,
    )
