"""Independent primitive-level WaveNet rebuild for curriculum Day 33."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .wavenet import (
    HierarchicalLanguageModel,
    ShapeTraceStep,
    WaveNetConfig,
    WaveNetError,
)


@dataclass(frozen=True)
class RebuildStageSpec:
    """One exact temporal grouping and projection contract."""

    index: int
    input_length: int
    output_length: int
    factor: int
    input_dim: int
    output_dim: int

    @property
    def weight_shape(self) -> tuple[int, int]:
        return self.output_dim, self.factor * self.input_dim


@dataclass(frozen=True)
class RebuildPlan:
    """Immutable parameter and shape plan compiled from a WaveNet config."""

    config_fingerprint: str
    vocab_size: int
    context_size: int
    embedding_dim: int
    hidden_dim: int
    dropout: float
    stages: tuple[RebuildStageSpec, ...]

    @property
    def parameter_count(self) -> int:
        embedding = self.vocab_size * self.embedding_dim
        stages = sum(
            stage.output_dim * stage.factor * stage.input_dim + 2 * stage.output_dim
            for stage in self.stages
        )
        output = self.vocab_size * self.hidden_dim + self.vocab_size
        return embedding + stages + output

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


def compile_rebuild_plan(config: WaveNetConfig) -> RebuildPlan:
    """Compile every rebuild tensor shape without constructing a model."""

    if not isinstance(config, WaveNetConfig):
        raise TypeError("config must be WaveNetConfig")
    stages: list[RebuildStageSpec] = []
    length = config.context_size
    input_dim = config.embedding_dim
    for index, factor in enumerate(config.group_factors):
        output_length = length // factor
        stages.append(
            RebuildStageSpec(
                index=index,
                input_length=length,
                output_length=output_length,
                factor=factor,
                input_dim=input_dim,
                output_dim=config.hidden_dim,
            )
        )
        length = output_length
        input_dim = config.hidden_dim
    if length != 1:
        raise WaveNetError("rebuild plan must reduce the context to one position")
    return RebuildPlan(
        config_fingerprint=config.fingerprint(),
        vocab_size=config.vocab_size,
        context_size=config.context_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=float(config.dropout),
        stages=tuple(stages),
    )


class RebuiltWaveNet(nn.Module):
    """WaveNet parameters registered without high-level layer modules."""

    def __init__(self, config: WaveNetConfig) -> None:
        super().__init__()
        self.config = config
        self.plan = compile_rebuild_plan(config)
        self.embedding_weight = nn.Parameter(
            torch.empty(config.vocab_size, config.embedding_dim)
        )
        self.stage_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(spec.weight_shape)) for spec in self.plan.stages]
        )
        self.stage_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(spec.output_dim)) for spec in self.plan.stages]
        )
        self.stage_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(spec.output_dim)) for spec in self.plan.stages]
        )
        self.output_weight = nn.Parameter(
            torch.empty(config.vocab_size, config.hidden_dim)
        )
        self.output_bias = nn.Parameter(torch.empty(config.vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding_weight)
        for weight in self.stage_weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.output_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.config.hidden_dim)
        nn.init.uniform_(self.output_bias, -bound, bound)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_manifest(self) -> dict[str, tuple[int, ...]]:
        """Return the exact shape of every registered trainable tensor."""

        return {
            name: tuple(parameter.shape) for name, parameter in self.named_parameters()
        }

    def forward(
        self, token_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        self._validate_inputs(token_ids, targets)
        hidden = self.embedding_weight[token_ids]
        for index, spec in enumerate(self.plan.stages):
            hidden = self._stage_forward(hidden, index=index, spec=spec)
        logits = hidden[:, 0, :] @ self.output_weight.transpose(0, 1)
        logits = logits + self.output_bias
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    def _stage_forward(
        self, hidden: Tensor, *, index: int, spec: RebuildStageSpec
    ) -> Tensor:
        hidden = hidden.reshape(
            hidden.shape[0],
            spec.output_length,
            spec.factor * spec.input_dim,
        )
        hidden = hidden @ self.stage_weights[index].transpose(0, 1)
        mean = hidden.mean(dim=-1, keepdim=True)
        variance = (hidden - mean).square().mean(dim=-1, keepdim=True)
        hidden = (hidden - mean) * torch.rsqrt(variance + 1e-5)
        hidden = hidden * self.stage_scales[index] + self.stage_biases[index]
        hidden = torch.tanh(hidden)
        return F.dropout(hidden, p=self.config.dropout, training=self.training)

    def _validate_inputs(self, token_ids: Tensor, targets: Tensor | None) -> None:
        if not isinstance(token_ids, Tensor) or token_ids.ndim != 2:
            raise TypeError("token_ids must be a two-dimensional tensor")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long dtype")
        if token_ids.shape[1] != self.config.context_size:
            raise WaveNetError("token_ids must match the configured context_size")
        if token_ids.numel() and (
            int(token_ids.min()) < 0 or int(token_ids.max()) >= self.config.vocab_size
        ):
            raise WaveNetError("token id is outside the vocabulary")
        if targets is None:
            return
        if not isinstance(targets, Tensor) or targets.shape != token_ids.shape[:1]:
            raise TypeError("targets must contain one token id per context")
        if targets.dtype != torch.long:
            raise TypeError("targets must use torch.long dtype")
        if targets.numel() and (
            int(targets.min()) < 0 or int(targets.max()) >= self.config.vocab_size
        ):
            raise WaveNetError("target token id is outside the vocabulary")


def initialize_rebuilt_wavenet(
    config: WaveNetConfig, *, seed: int = 33
) -> RebuiltWaveNet:
    """Initialize the rebuild repeatably without advancing caller RNG state."""

    if not isinstance(config, WaveNetConfig):
        raise TypeError("config must be WaveNetConfig")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return RebuiltWaveNet(config)


def trace_rebuild_shapes(
    model: RebuiltWaveNet, *, batch_size: int = 2
) -> tuple[ShapeTraceStep, ...]:
    """Run the primitive implementation and record every tensor boundary."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise WaveNetError("batch_size must be positive")
    mode = model.training
    steps: list[ShapeTraceStep] = []
    try:
        model.eval()
        with torch.no_grad():
            token_ids = torch.zeros(
                batch_size, model.config.context_size, dtype=torch.long
            )
            hidden = model.embedding_weight[token_ids]
            steps.append(
                ShapeTraceStep(
                    "embedding",
                    tuple(token_ids.shape),
                    tuple(hidden.shape),
                    model.embedding_weight.numel(),
                )
            )
            for index, spec in enumerate(model.plan.stages):
                inputs = hidden
                hidden = model._stage_forward(hidden, index=index, spec=spec)
                parameter_count = (
                    model.stage_weights[index].numel()
                    + model.stage_scales[index].numel()
                    + model.stage_biases[index].numel()
                )
                steps.append(
                    ShapeTraceStep(
                        f"stage_{index + 1}",
                        tuple(inputs.shape),
                        tuple(hidden.shape),
                        parameter_count,
                    )
                )
            head_inputs = hidden[:, 0, :]
            logits = head_inputs @ model.output_weight.transpose(0, 1)
            logits = logits + model.output_bias
            steps.append(
                ShapeTraceStep(
                    "output",
                    tuple(head_inputs.shape),
                    tuple(logits.shape),
                    model.output_weight.numel() + model.output_bias.numel(),
                )
            )
    finally:
        model.train(mode)
    return tuple(steps)


def load_reference_parameters(
    model: RebuiltWaveNet, reference: HierarchicalLanguageModel
) -> None:
    """Copy one reference model into the independently structured rebuild."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    if not isinstance(reference, HierarchicalLanguageModel):
        raise TypeError("reference must be HierarchicalLanguageModel")
    if model.config != reference.config:
        raise WaveNetError("reference and rebuild configurations must match")
    with torch.no_grad():
        model.embedding_weight.copy_(reference.embedding.weight)
        for index, stage in enumerate(reference.stages):
            linear = stage.network[1]
            normalization = stage.network[2]
            assert isinstance(linear, nn.Linear)
            assert isinstance(normalization, nn.LayerNorm)
            model.stage_weights[index].copy_(linear.weight)
            model.stage_scales[index].copy_(normalization.weight)
            model.stage_biases[index].copy_(normalization.bias)
        model.output_weight.copy_(reference.output.weight)
        model.output_bias.copy_(reference.output.bias)


def rebuild_model_fingerprint(model: RebuiltWaveNet) -> str:
    """Hash the rebuild architecture binding and every parameter byte."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    digest = sha256(model.plan.fingerprint().encode())
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
