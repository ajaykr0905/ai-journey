"""Independent primitive-level WaveNet rebuild for curriculum Day 33."""

from __future__ import annotations

import copy
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
    WaveNetBatchCursor,
    WaveNetConfig,
    WaveNetDataset,
    WaveNetDatasetSplit,
    WaveNetError,
    WaveNetMetrics,
    WaveNetTrainingConfig,
    WaveNetTrainingStep,
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


@dataclass(frozen=True)
class RebuildForwardAudit:
    """Numerical agreement between reference and primitive forward passes."""

    examples: int
    max_abs_logit_error: float
    loss_abs_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.max_abs_logit_error <= self.tolerance
            and self.loss_abs_error <= self.tolerance
        )


def audit_rebuild_forward(
    reference: HierarchicalLanguageModel,
    rebuilt: RebuiltWaveNet,
    contexts: Tensor,
    targets: Tensor,
    *,
    tolerance: float = 1e-6,
) -> RebuildForwardAudit:
    """Compare logits and loss after mapping the same reference parameters."""

    if not isinstance(reference, HierarchicalLanguageModel):
        raise TypeError("reference must be HierarchicalLanguageModel")
    if not isinstance(rebuilt, RebuiltWaveNet):
        raise TypeError("rebuilt must be RebuiltWaveNet")
    if reference.config != rebuilt.config:
        raise WaveNetError("reference and rebuild configurations must match")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise WaveNetError("tolerance must be non-negative and finite")
    rebuilt._validate_inputs(contexts, targets)
    reference_mode = reference.training
    rebuilt_mode = rebuilt.training
    try:
        reference.eval()
        rebuilt.eval()
        with torch.no_grad():
            reference_logits, reference_loss = reference(contexts, targets)
            rebuilt_logits, rebuilt_loss = rebuilt(contexts, targets)
            assert reference_loss is not None and rebuilt_loss is not None
            return RebuildForwardAudit(
                examples=int(targets.numel()),
                max_abs_logit_error=float(
                    (reference_logits - rebuilt_logits).abs().max()
                ),
                loss_abs_error=abs(float(reference_loss) - float(rebuilt_loss)),
                tolerance=float(tolerance),
            )
    finally:
        reference.train(reference_mode)
        rebuilt.train(rebuilt_mode)


@dataclass(frozen=True)
class RebuildGradientAudit:
    """Gradient agreement across every mapped reference and rebuild tensor."""

    parameter_tensors: int
    max_abs_error: float
    mismatched_parameters: tuple[str, ...]
    nonfinite_parameters: tuple[str, ...]
    tolerance: float

    @property
    def passed(self) -> bool:
        return not self.mismatched_parameters and not self.nonfinite_parameters


def _reference_parameter_pairs(
    reference: HierarchicalLanguageModel, rebuilt: RebuiltWaveNet
) -> tuple[tuple[str, nn.Parameter, nn.Parameter], ...]:
    pairs: list[tuple[str, nn.Parameter, nn.Parameter]] = [
        ("embedding", reference.embedding.weight, rebuilt.embedding_weight)
    ]
    for index, stage in enumerate(reference.stages):
        linear = stage.network[1]
        normalization = stage.network[2]
        assert isinstance(linear, nn.Linear)
        assert isinstance(normalization, nn.LayerNorm)
        pairs.extend(
            (
                (
                    f"stage_{index + 1}.weight",
                    linear.weight,
                    rebuilt.stage_weights[index],
                ),
                (
                    f"stage_{index + 1}.scale",
                    normalization.weight,
                    rebuilt.stage_scales[index],
                ),
                (
                    f"stage_{index + 1}.bias",
                    normalization.bias,
                    rebuilt.stage_biases[index],
                ),
            )
        )
    pairs.extend(
        (
            ("output.weight", reference.output.weight, rebuilt.output_weight),
            ("output.bias", reference.output.bias, rebuilt.output_bias),
        )
    )
    return tuple(pairs)


def audit_rebuild_gradients(
    reference: HierarchicalLanguageModel,
    rebuilt: RebuiltWaveNet,
    contexts: Tensor,
    targets: Tensor,
    *,
    tolerance: float = 1e-6,
) -> RebuildGradientAudit:
    """Backpropagate identical losses and compare every mapped gradient."""

    if not isinstance(reference, HierarchicalLanguageModel):
        raise TypeError("reference must be HierarchicalLanguageModel")
    if not isinstance(rebuilt, RebuiltWaveNet):
        raise TypeError("rebuilt must be RebuiltWaveNet")
    if reference.config != rebuilt.config:
        raise WaveNetError("reference and rebuild configurations must match")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise WaveNetError("tolerance must be non-negative and finite")
    rebuilt._validate_inputs(contexts, targets)
    reference_mode = reference.training
    rebuilt_mode = rebuilt.training
    try:
        reference.eval()
        rebuilt.eval()
        reference.zero_grad(set_to_none=True)
        rebuilt.zero_grad(set_to_none=True)
        _, reference_loss = reference(contexts, targets)
        _, rebuilt_loss = rebuilt(contexts, targets)
        assert reference_loss is not None and rebuilt_loss is not None
        reference_loss.backward()
        rebuilt_loss.backward()
        maximum = 0.0
        mismatched: list[str] = []
        nonfinite: list[str] = []
        pairs = _reference_parameter_pairs(reference, rebuilt)
        for name, reference_parameter, rebuilt_parameter in pairs:
            reference_gradient = reference_parameter.grad
            rebuilt_gradient = rebuilt_parameter.grad
            if reference_gradient is None or rebuilt_gradient is None:
                mismatched.append(name)
                continue
            if not bool(torch.isfinite(reference_gradient).all()) or not bool(
                torch.isfinite(rebuilt_gradient).all()
            ):
                nonfinite.append(name)
                continue
            error = float((reference_gradient - rebuilt_gradient).abs().max())
            maximum = max(maximum, error)
            if error > tolerance:
                mismatched.append(name)
        return RebuildGradientAudit(
            parameter_tensors=len(pairs),
            max_abs_error=maximum,
            mismatched_parameters=tuple(mismatched),
            nonfinite_parameters=tuple(nonfinite),
            tolerance=float(tolerance),
        )
    finally:
        reference.train(reference_mode)
        rebuilt.train(rebuilt_mode)


@dataclass(frozen=True)
class RebuildFiniteDifferenceAudit:
    """Independent central-difference check for one rebuild parameter value."""

    parameter: str
    index: tuple[int, ...]
    analytic_gradient: float
    numerical_gradient: float
    absolute_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.absolute_error <= self.tolerance


def audit_rebuild_finite_difference(
    model: RebuiltWaveNet,
    contexts: Tensor,
    targets: Tensor,
    *,
    parameter: str = "output_bias",
    index: tuple[int, ...] = (0,),
    epsilon: float = 1e-3,
    tolerance: float = 1e-3,
) -> RebuildFiniteDifferenceAudit:
    """Compare autograd with a central difference at one selected coordinate."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    model._validate_inputs(contexts, targets)
    named_parameters = dict(model.named_parameters())
    if parameter not in named_parameters:
        raise WaveNetError(f"unknown rebuild parameter: {parameter}")
    selected = named_parameters[parameter]
    if (
        not isinstance(index, tuple)
        or len(index) != selected.ndim
        or any(isinstance(value, bool) or not isinstance(value, int) for value in index)
    ):
        raise TypeError("index must contain one integer per parameter dimension")
    if any(
        value < 0 or value >= selected.shape[axis] for axis, value in enumerate(index)
    ):
        raise WaveNetError("parameter index is out of bounds")
    for name, value, positive in (
        ("epsilon", epsilon, True),
        ("tolerance", tolerance, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)
        ):
            qualifier = "positive" if positive else "non-negative"
            raise WaveNetError(f"{name} must be {qualifier} and finite")
    mode = model.training
    original = float(selected[index].detach())
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        _, loss = model(contexts, targets)
        assert loss is not None
        loss.backward()
        assert selected.grad is not None
        analytic = float(selected.grad[index])
        with torch.no_grad():
            selected[index] = original + epsilon
            _, plus_loss = model(contexts, targets)
            selected[index] = original - epsilon
            _, minus_loss = model(contexts, targets)
            selected[index] = original
        assert plus_loss is not None and minus_loss is not None
        numerical = (float(plus_loss) - float(minus_loss)) / (2 * epsilon)
        return RebuildFiniteDifferenceAudit(
            parameter=parameter,
            index=index,
            analytic_gradient=analytic,
            numerical_gradient=numerical,
            absolute_error=abs(analytic - numerical),
            tolerance=float(tolerance),
        )
    finally:
        with torch.no_grad():
            selected[index] = original
        model.train(mode)


def evaluate_rebuild(
    model: RebuiltWaveNet,
    dataset: WaveNetDataset,
    *,
    batch_size: int = 256,
) -> WaveNetMetrics:
    """Evaluate every dataset example once while preserving model mode."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    if not isinstance(dataset, WaveNetDataset):
        raise TypeError("dataset must be WaveNetDataset")
    if dataset.context_size != model.config.context_size:
        raise WaveNetError("dataset context_size does not match the rebuild")
    if dataset.vocab_size != model.config.vocab_size:
        raise WaveNetError("dataset vocabulary does not match the rebuild")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise WaveNetError("batch_size must be positive")
    mode = model.training
    total_loss = 0.0
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, dataset.sample_count, batch_size):
                stop = min(start + batch_size, dataset.sample_count)
                _, loss = model(
                    dataset.contexts[start:stop], dataset.targets[start:stop]
                )
                assert loss is not None
                total_loss += float(loss) * (stop - start)
    finally:
        model.train(mode)
    nll = total_loss / dataset.sample_count
    return WaveNetMetrics(
        nll=nll, perplexity=math.exp(nll), sample_count=dataset.sample_count
    )


def train_rebuild_steps(
    model: RebuiltWaveNet,
    cursor: WaveNetBatchCursor,
    optimizer: torch.optim.Optimizer,
    config: WaveNetTrainingConfig,
    *,
    start_step: int = 0,
    step_count: int | None = None,
) -> tuple[WaveNetTrainingStep, ...]:
    """Train a bounded rebuild interval with auditable loss and gradient norms."""

    if not isinstance(model, RebuiltWaveNet):
        raise TypeError("model must be RebuiltWaveNet")
    if not isinstance(cursor, WaveNetBatchCursor):
        raise TypeError("cursor must be WaveNetBatchCursor")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(config, WaveNetTrainingConfig):
        raise TypeError("config must be WaveNetTrainingConfig")
    if isinstance(start_step, bool) or not isinstance(start_step, int):
        raise TypeError("start_step must be an integer")
    if start_step < 0:
        raise WaveNetError("start_step must be non-negative")
    count = config.steps if step_count is None else step_count
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("step_count must be an integer")
    if count <= 0:
        raise WaveNetError("step_count must be positive")
    trace: list[WaveNetTrainingStep] = []
    for step in range(start_step, start_step + count):
        model.train()
        contexts, targets = cursor.next()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(contexts, targets)
        assert loss is not None
        if not torch.isfinite(loss):
            raise WaveNetError("rebuild training produced a nonfinite loss")
        loss.backward()
        gradient_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        )
        if not math.isfinite(gradient_norm):
            raise WaveNetError("rebuild training produced a nonfinite gradient norm")
        optimizer.step()
        trace.append(
            WaveNetTrainingStep(
                step=step, loss=float(loss.detach()), gradient_norm=gradient_norm
            )
        )
    return tuple(trace)


@dataclass(frozen=True)
class RebuildTrainingResult:
    """Deterministic rebuild training state and full-dataset metrics."""

    model: RebuiltWaveNet
    initial_train: WaveNetMetrics
    initial_validation: WaveNetMetrics
    final_train: WaveNetMetrics
    final_validation: WaveNetMetrics
    trace: tuple[WaveNetTrainingStep, ...]
    optimizer_state: dict[str, object]
    cursor_state: dict[str, int]


def fit_rebuild(
    datasets: WaveNetDatasetSplit,
    *,
    model_config: WaveNetConfig,
    training_config: WaveNetTrainingConfig,
) -> RebuildTrainingResult:
    """Train the primitive rebuild deterministically without caller RNG effects."""

    if not isinstance(datasets, WaveNetDatasetSplit):
        raise TypeError("datasets must be WaveNetDatasetSplit")
    if not isinstance(model_config, WaveNetConfig):
        raise TypeError("model_config must be WaveNetConfig")
    if not isinstance(training_config, WaveNetTrainingConfig):
        raise TypeError("training_config must be WaveNetTrainingConfig")
    if datasets.train.context_size != model_config.context_size:
        raise WaveNetError("dataset context_size does not match the rebuild")
    if datasets.train.vocab_size != model_config.vocab_size:
        raise WaveNetError("dataset vocabulary does not match the rebuild")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(training_config.seed)
        model = RebuiltWaveNet(model_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        cursor = WaveNetBatchCursor(
            datasets.train,
            batch_size=training_config.batch_size,
            seed=training_config.seed,
        )
        initial_train = evaluate_rebuild(model, datasets.train)
        initial_validation = evaluate_rebuild(model, datasets.validation)
        trace = train_rebuild_steps(model, cursor, optimizer, training_config)
        return RebuildTrainingResult(
            model=model,
            initial_train=initial_train,
            initial_validation=initial_validation,
            final_train=evaluate_rebuild(model, datasets.train),
            final_validation=evaluate_rebuild(model, datasets.validation),
            trace=trace,
            optimizer_state=copy.deepcopy(optimizer.state_dict()),
            cursor_state=cursor.state_dict(),
        )
