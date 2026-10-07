"""Independent primitive-level WaveNet rebuild for curriculum Day 33."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

from .wavenet import WaveNetConfig, WaveNetError


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
