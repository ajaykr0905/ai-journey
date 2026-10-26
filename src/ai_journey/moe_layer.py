"""A capacity-aware expert FFN with live auxiliary losses and detached evidence."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .gpt2_model import GPT2MLP
from .moe_capacity import plan_capacity
from .moe_config import MoEConfig
from .moe_dispatch import dispatch_experts
from .moe_routing import Top2Router
from .swiglu import SwiGLU


@dataclass(frozen=True)
class MoELayerOutput:
    hidden: Tensor
    balance_loss: Tensor
    router_z_loss: Tensor
    stats: dict
    idle_experts: tuple[int, ...]


class MoEFeedForward(nn.Module):
    """CPU reference: sparse forward compute, training-only capacity drops."""

    def __init__(self, config: MoEConfig, *, layer_index: int = 0):
        super().__init__()
        if not isinstance(config, MoEConfig):
            raise TypeError("config must be MoEConfig")
        if type(layer_index) is not int or layer_index not in config.moe_layers:
            raise ValueError("layer_index must name a configured MoE layer")
        self.config = config
        self.layer_index = layer_index
        self.router = Top2Router(config.gpt2.embedding_dim, config.expert_count)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    SwiGLU(
                        config.gpt2.embedding_dim, config.resolved_swiglu_hidden_dim
                    ),
                    nn.Dropout(config.gpt2.residual_dropout),
                )
                if config.mlp_kind == "swiglu"
                else GPT2MLP(config.gpt2)
                for _ in range(config.expert_count)
            ]
        )

    def forward(self, inputs: Tensor, *, token_mask: Tensor | None = None) -> Tensor:
        return self.forward_training(inputs, token_mask=token_mask).hidden

    def forward_training(
        self, inputs: Tensor, *, token_mask: Tensor | None = None
    ) -> MoELayerOutput:
        routes = self.router(inputs, token_mask=token_mask)
        plan = plan_capacity(
            routes,
            capacity_factor=self.config.capacity_factor,
            min_capacity=self.config.min_capacity,
            apply_capacity=self.training,
        )
        hidden = dispatch_experts(inputs, self.experts, routes, plan)
        count = len(routes.token_indices)
        if count:
            frequency = plan.attempted_counts.to(routes.probabilities.dtype) / (
                count * 2
            )
            balance = (
                self.config.expert_count
                * (frequency * routes.probabilities.mean(0)).sum()
            )
            z_loss = torch.logsumexp(routes.logits.double(), dim=-1).square().mean()
        else:
            balance = routes.logits.sum() * 0
            z_loss = balance.double()
        if not bool(torch.isfinite(balance)) or not bool(torch.isfinite(z_loss)):
            raise ValueError("router auxiliary losses must be finite")
        accepted = torch.zeros(math.prod(routes.token_shape), dtype=torch.long)
        accepted[routes.token_indices] = plan.accepted.sum(-1)
        stats = {
            "layer_index": self.layer_index,
            "valid_tokens": count,
            "capacity": plan.capacity,
            "attempted_counts": tuple(plan.attempted_counts.tolist()),
            "accepted_counts": tuple(plan.accepted_counts.tolist()),
            "dropped_counts": tuple(plan.dropped_counts.tolist()),
            "probability_sums": tuple(
                routes.probabilities.detach().double().sum(0).tolist()
            ),
            "capacity_applied": plan.capacity_applied,
            "accepted_per_token": tuple(
                tuple(row) for row in accepted.reshape(routes.token_shape).tolist()
            )
            if len(routes.token_shape) == 2
            else tuple(accepted.tolist()),
            "token_indices": tuple(routes.token_indices.tolist()),
            "selected_experts": tuple(
                tuple(row) for row in routes.expert_indices.tolist()
            ),
            "accepted_routes": tuple(tuple(row) for row in plan.accepted.tolist()),
        }
        idle = tuple(
            index
            for index, accepted_count in enumerate(stats["accepted_counts"])
            if accepted_count == 0
        )
        return MoELayerOutput(hidden, balance, z_loss, stats, idle)
