"""Small, explicit low-rank adapters for the bounded CPU decoder lab."""

from dataclasses import asdict, dataclass, replace
import copy
from hashlib import sha256
import math
import json

import torch
from torch import nn
from torch.nn import functional as F


class LoRAError(ValueError):
    """An adapter, its target, or its saved state violates the contract."""


@dataclass(frozen=True)
class LoRAConfig:
    """Predeclared adapter capacity, scaling, dropout, and local RNG seed."""

    rank: int = 2
    alpha: float = 4.0
    dropout: float = 0.0
    seed: int = 42

    def __post_init__(self):
        if type(self.rank) is not int or not 1 <= self.rank <= 256:
            raise LoRAError("rank must be an integer in [1, 256]")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(self.alpha)
            or not 0 < self.alpha <= 1024
        ):
            raise LoRAError("alpha must be finite and in (0, 1024]")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise LoRAError("dropout must be finite and in [0, 1)")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise LoRAError("seed must be an integer in [0, 2**32)")

    @property
    def scaling(self):
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Frozen dense projection plus ``(alpha/rank) * B @ A``.

    B starts at zero, so installing an adapter preserves the base function.
    A uses a local generator: constructing adapters does not consume the
    caller's training RNG. The supplied dense layer is never modified.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        if type(base) is not nn.Linear or not isinstance(config, LoRAConfig):
            raise TypeError("expected a plain nn.Linear and LoRAConfig")
        if base.weight.device.type != "cpu" or not base.weight.is_floating_point():
            raise LoRAError("this adapter lab requires floating-point CPU weights")
        if config.rank > min(base.in_features, base.out_features):
            raise LoRAError("adapter rank exceeds the projection dimensions")
        if any(not bool(torch.isfinite(p).all()) for p in base.parameters()):
            raise LoRAError("base projection parameters must be finite")
        self.config = config
        self.base = copy.deepcopy(base)
        self.base.requires_grad_(False)
        for parameter in self.base.parameters():
            parameter.grad = None
        self.in_features = base.in_features
        self.out_features = base.out_features
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        a = torch.empty(config.rank, base.in_features, dtype=base.weight.dtype)
        a.normal_(mean=0, std=1 / math.sqrt(base.in_features), generator=generator)
        self.lora_a = nn.Parameter(a)
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, config.rank, dtype=base.weight.dtype)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.train(base.training)

    @property
    def weight(self):
        """Expose the base dtype/device contract used by decoder validation."""
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def delta_weight(self):
        return (self.lora_b @ self.lora_a) * self.config.scaling

    def forward(self, inputs):
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.config.scaling


def inject_adapters(model: nn.Module, targets: tuple[str, ...], config: LoRAConfig):
    """Return an independent model adapting only explicitly named projections.

    Resolve every target before cloning or freezing. Aliased modules and nested
    adapters are rejected rather than silently adapting only some references.
    Target-specific local seeds make installation order irrelevant.
    """
    if not isinstance(model, nn.Module) or not isinstance(config, LoRAConfig):
        raise TypeError("expected nn.Module and LoRAConfig")
    if (
        not isinstance(targets, tuple)
        or not 1 <= len(targets) <= 128
        or any(not isinstance(name, str) or not name for name in targets)
        or len(set(targets)) != len(targets)
    ):
        raise LoRAError("targets must be a nonempty tuple of distinct module paths")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise LoRAError("model already contains adapters")
    all_modules = list(model.named_modules(remove_duplicate=False))
    for name in targets:
        try:
            target = model.get_submodule(name)
        except AttributeError as exc:
            raise LoRAError(f"unknown adapter target: {name}") from exc
        if type(target) is not nn.Linear:
            raise LoRAError(f"adapter target must be a plain linear layer: {name}")
        if sum(module is target for _, module in all_modules) != 1:
            raise LoRAError(f"adapter target is shared by multiple paths: {name}")
        if config.rank > min(target.in_features, target.out_features):
            raise LoRAError(f"adapter rank exceeds target dimensions: {name}")
    result = copy.deepcopy(model)
    result.requires_grad_(False)
    for parameter in result.parameters():
        parameter.grad = None
    for name in targets:
        seed = int.from_bytes(sha256(f"{config.seed}:{name}".encode()).digest()[:4])
        parent_name, _, child_name = name.rpartition(".")
        parent = result.get_submodule(parent_name)
        parent.add_module(
            child_name,
            LoRALinear(result.get_submodule(name), replace(config, seed=seed)),
        )
    return result


def adapter_parameters(model: nn.Module):
    """Return exactly the low-rank factors, in stable module traversal order."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")
    return tuple(
        parameter
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_a, module.lora_b)
    )


def parameter_summary(model: nn.Module):
    """Count unique parameters without equating savings with measured RAM."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _tensor_digest(items):
    digest = sha256()
    for name, parameter in sorted(items):
        value = parameter.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_fingerprint(model: nn.Module):
    """Bind names, shapes, dtypes and bytes of every frozen parameter/buffer."""
    items = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    items.extend(model.named_buffers())
    return _tensor_digest(items)


def build_adapter_optimizer(model: nn.Module, *, learning_rate=3e-3, weight_decay=0.0):
    """Build AdamW only after checking the complete trainability boundary."""
    for name, value in (
        ("learning_rate", learning_rate),
        ("weight_decay", weight_decay),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (name == "learning_rate" and value == 0)
        ):
            raise LoRAError(
                f"{name} must be finite and {'positive' if name == 'learning_rate' else 'nonnegative'}"
            )
    parameters = adapter_parameters(model)
    expected = {id(p) for p in parameters}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if not expected or actual != expected:
        raise LoRAError("only adapter factors may be trainable")
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise LoRAError("adapter parameters must be finite")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def adapter_state(model: nn.Module):
    """Export only factors and the exact frozen-model identity they belong to.

    Returned tensors own their storage. This is an adapter checkpoint, not an
    optimizer or training-resume checkpoint, and contains no base-model weights.
    """
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if any(
                not bool(torch.isfinite(p).all())
                for p in (module.lora_a, module.lora_b)
            ):
                raise LoRAError("cannot export nonfinite adapter factors")
            layers[name] = {
                "config": asdict(module.config),
                "a": module.lora_a.detach().clone(),
                "b": module.lora_b.detach().clone(),
            }
    if not layers:
        raise LoRAError("model contains no adapters")
    return {
        "schema": "lora-state-v1",
        "base_fingerprint": frozen_fingerprint(model),
        "layers": layers,
    }
