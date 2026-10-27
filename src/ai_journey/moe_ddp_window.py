"""One DDP-visible forward owns the entire bounded optimizer-window graph."""

import torch
from torch import nn
from torch.nn import functional as F

from .moe_model import MoEDecoderModel
from .moe_route_audit import validate_moe_routing
from .moe_ddp_resources import require_moe_ddp_budget


def validate_moe_ddp_window(window, config):
    require_moe_ddp_budget(config)
    if type(window) is not tuple or len(window) != config.accumulation_steps:
        raise ValueError("MoE DDP window must contain every declared microbatch")
    count = 0
    for batch in window:
        if type(batch) is not tuple or len(batch) != 3:
            raise ValueError("MoE DDP microbatch requires inputs, targets and mask")
        inputs, targets, mask = batch
        if (
            any(
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or tuple(value.shape)
                != (config.micro_batch_size, config.sequence_length)
                for value in batch
            )
            or inputs.dtype != torch.long
            or targets.dtype != torch.long
            or mask.dtype != torch.bool
        ):
            raise ValueError("MoE DDP microbatch shape/dtype/device differs")
        if inputs.min() < 0 or inputs.max() >= config.model.gpt2.vocab_size:
            raise ValueError("MoE DDP input token is out of range")
        if mask.any() and (
            targets[mask].min() < 0
            or targets[mask].max() >= config.model.gpt2.vocab_size
        ):
            raise ValueError("MoE DDP eligible target is out of range")
        count += int(mask.sum())
    return count


class MoEOptimizerWindow(nn.Module):
    def __init__(self, decoder, config):
        super().__init__()
        if type(decoder) is not MoEDecoderModel or decoder.config != config.model:
            raise ValueError("MoE DDP window requires its declared expert model")
        self.decoder, self.config = decoder, config

    def forward(self, window):
        count = validate_moe_ddp_window(window, self.config)
        layers = self.config.model.moe_layers
        captured = {layer: [] for layer in layers}
        handles = []
        ce, routing = [], []
        try:
            for layer in layers:

                def capture(module, args, output, layer=layer):
                    captured[layer].append(output)

                handles.append(
                    self.decoder.h[layer].mlp.router.register_forward_hook(capture)
                )
            for inputs, targets, mask in window:
                output = self.decoder.forward_training(inputs, token_mask=mask)
                ce.append(
                    F.cross_entropy(output.logits[mask], targets[mask], reduction="sum")
                    if mask.any()
                    else output.logits[0, 0, 0] * 0
                )
                for stats in output.routing:
                    validate_moe_routing(
                        stats, expert_count=self.config.model.expert_count
                    )
                routing.append(output.routing)
        finally:
            for handle in handles:
                handle.remove()
        if any(
            len(captured[layer]) != self.config.accumulation_steps for layer in layers
        ):
            raise ValueError("MoE DDP did not observe every live router forward")
        probabilities, z_sums = [], []
        for layer in layers:
            probabilities.append(
                torch.stack(
                    [row.probabilities.double().sum(0) for row in captured[layer]]
                ).sum(0)
            )
            z_sums.append(
                torch.stack(
                    [
                        torch.logsumexp(row.logits.double(), -1).square().sum()
                        for row in captured[layer]
                    ]
                ).sum()
            )
        return {
            "ce_sum": torch.stack(ce).sum(),
            "probability_sums": torch.stack(probabilities),
            "z_sums": torch.stack(z_sums),
            "attempted_counts": torch.tensor(
                [
                    [
                        sum(
                            batch[index]["attempted_counts"][expert]
                            for batch in routing
                        )
                        for expert in range(self.config.model.expert_count)
                    ]
                    for index in range(len(layers))
                ],
                dtype=torch.long,
            ),
            "accepted_counts": torch.tensor(
                [
                    [
                        sum(
                            batch[index]["accepted_counts"][expert] for batch in routing
                        )
                        for expert in range(self.config.model.expert_count)
                    ]
                    for index in range(len(layers))
                ],
                dtype=torch.long,
            ),
            "target_tokens": count,
            "routing": tuple(routing),
        }
