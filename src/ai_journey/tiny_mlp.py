"""Day 17: a deterministic multilayer perceptron built on scalar autodiff.

The module composes the independently written :class:`Value` engine from Day
15 into neurons, dense layers, and an MLP.  It keeps the training experiment
small enough to inspect: four examples, full-batch gradient descent, explicit
gradient clearing, optional clipping, and finite-difference checks against
selected parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite, sqrt
from numbers import Real
from random import Random

from .scalar_autodiff import Value


class MLPValidationError(ValueError):
    """Raised when model, data, training, or experiment invariants fail."""


@dataclass(frozen=True)
class TrainingExample:
    """One finite feature vector and a binary target in ``{-1, +1}``."""

    inputs: tuple[float, ...]
    target: float


@dataclass(frozen=True)
class TrainingConfig:
    """Deterministic full-batch training settings."""

    layer_widths: tuple[int, ...] = (4, 4, 1)
    steps: int = 120
    learning_rate: float = 0.05
    l2_coefficient: float = 1e-4
    gradient_clip_norm: float | None = 5.0
    record_every: int = 10
    seed: int = 1709


@dataclass(frozen=True)
class ParameterSnapshot:
    """Stable name and value for one trainable scalar."""

    name: str
    value: float


@dataclass(frozen=True)
class GradientProbe:
    """Autodiff and centered-finite-difference gradients for one parameter."""

    name: str
    analytic: float
    numerical: float
    absolute_error: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class TrainingPoint:
    """Loss, predictions, and gradient statistics at one optimizer step."""

    step: int
    total_loss: float
    data_loss: float
    l2_penalty: float
    gradient_l2_norm: float
    gradient_scale: float
    accuracy: float
    predictions: tuple[float, ...]


@dataclass(frozen=True)
class TrainingResult:
    """Complete deterministic training trace and final parameter state."""

    config: TrainingConfig
    parameter_count: int
    history: tuple[TrainingPoint, ...]
    final_parameters: tuple[ParameterSnapshot, ...]


@dataclass(frozen=True)
class MLPExperiment:
    """The fixed Day 17 dataset, gradient probes, and training trace."""

    input_width: int
    dataset: tuple[TrainingExample, ...]
    gradient_probes: tuple[GradientProbe, ...]
    training: TrainingResult


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def validate_layer_widths(widths: Iterable[int]) -> tuple[int, ...]:
    """Require one or more positive, non-boolean layer widths."""

    normalized = tuple(widths)
    if not normalized:
        raise MLPValidationError("layer widths must not be empty")
    if any(
        isinstance(width, bool) or not isinstance(width, int) for width in normalized
    ):
        raise TypeError("layer widths must contain integers")
    if any(width <= 0 for width in normalized):
        raise MLPValidationError("layer widths must be positive")
    return normalized


def validate_training_config(config: TrainingConfig) -> TrainingConfig:
    """Validate every optimizer and recording control."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    validate_layer_widths(config.layer_widths)
    if isinstance(config.steps, bool) or not isinstance(config.steps, int):
        raise TypeError("steps must be an integer")
    if config.steps <= 0:
        raise MLPValidationError("steps must be positive")
    if isinstance(config.record_every, bool) or not isinstance(
        config.record_every, int
    ):
        raise TypeError("record_every must be an integer")
    if config.record_every <= 0:
        raise MLPValidationError("record_every must be positive")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise TypeError("seed must be an integer")
    learning_rate = _finite_scalar("learning_rate", config.learning_rate)
    l2_coefficient = _finite_scalar("l2_coefficient", config.l2_coefficient)
    if learning_rate <= 0.0:
        raise MLPValidationError("learning_rate must be positive")
    if l2_coefficient < 0.0:
        raise MLPValidationError("l2_coefficient must be non-negative")
    if config.gradient_clip_norm is not None:
        clip = _finite_scalar("gradient_clip_norm", config.gradient_clip_norm)
        if clip <= 0.0:
            raise MLPValidationError("gradient_clip_norm must be positive")
    return config


def validate_dataset(
    examples: Iterable[TrainingExample], input_width: int
) -> tuple[TrainingExample, ...]:
    """Normalize a non-empty, fixed-width binary classification dataset."""

    if isinstance(input_width, bool) or not isinstance(input_width, int):
        raise TypeError("input_width must be an integer")
    if input_width <= 0:
        raise MLPValidationError("input_width must be positive")
    normalized: list[TrainingExample] = []
    for index, example in enumerate(examples):
        if not isinstance(example, TrainingExample):
            raise TypeError("dataset must contain TrainingExample objects")
        if len(example.inputs) != input_width:
            raise MLPValidationError(
                f"example {index} has {len(example.inputs)} inputs; expected {input_width}"
            )
        inputs = tuple(
            _finite_scalar(f"example {index} input {column}", value)
            for column, value in enumerate(example.inputs)
        )
        target = _finite_scalar(f"example {index} target", example.target)
        if target not in {-1.0, 1.0}:
            raise MLPValidationError("classification targets must be -1 or +1")
        normalized.append(TrainingExample(inputs, target))
    if not normalized:
        raise MLPValidationError("dataset must not be empty")
    return tuple(normalized)


class Neuron:
    """A scalar affine transform followed by optional ``tanh``."""

    def __init__(
        self,
        input_width: int,
        rng: Random,
        *,
        name: str,
        nonlinear: bool = True,
    ) -> None:
        if isinstance(input_width, bool) or not isinstance(input_width, int):
            raise TypeError("input_width must be an integer")
        if input_width <= 0:
            raise MLPValidationError("input_width must be positive")
        if not isinstance(rng, Random):
            raise TypeError("rng must be random.Random")
        if not isinstance(name, str) or not name:
            raise MLPValidationError("neuron name must be non-empty")
        scale = 1.0 / sqrt(input_width)
        self.weights = tuple(
            Value(rng.uniform(-scale, scale), label=f"{name}.weight{index}")
            for index in range(input_width)
        )
        self.bias = Value(0.0, label=f"{name}.bias")
        self.nonlinear = nonlinear

    def __call__(self, inputs: Sequence[Value]) -> Value:
        if len(inputs) != len(self.weights):
            raise MLPValidationError(
                f"neuron expected {len(self.weights)} inputs; received {len(inputs)}"
            )
        if any(not isinstance(value, Value) for value in inputs):
            raise TypeError("neuron inputs must be Value objects")
        activation = self.bias
        for weight, value in zip(self.weights, inputs):
            activation = activation + weight * value
        return activation.tanh() if self.nonlinear else activation

    def named_parameters(self) -> tuple[tuple[str, Value], ...]:
        """Return stable parameter names paired with their scalar values."""

        return tuple((value.label, value) for value in (*self.weights, self.bias))


class DenseLayer:
    """A collection of neurons sharing the same input vector."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        rng: Random,
        *,
        name: str,
        nonlinear: bool = True,
    ) -> None:
        if isinstance(output_width, bool) or not isinstance(output_width, int):
            raise TypeError("output_width must be an integer")
        if output_width <= 0:
            raise MLPValidationError("output_width must be positive")
        self.input_width = input_width
        self.output_width = output_width
        self.neurons = tuple(
            Neuron(
                input_width,
                rng,
                name=f"{name}.neuron{index}",
                nonlinear=nonlinear,
            )
            for index in range(output_width)
        )

    def __call__(self, inputs: Sequence[Value]) -> tuple[Value, ...]:
        return tuple(neuron(inputs) for neuron in self.neurons)

    def named_parameters(self) -> tuple[tuple[str, Value], ...]:
        return tuple(
            parameter
            for neuron in self.neurons
            for parameter in neuron.named_parameters()
        )


class MLP:
    """A deterministic sequence of dense ``tanh`` layers."""

    def __init__(
        self, input_width: int, layer_widths: Iterable[int], *, seed: int = 1709
    ) -> None:
        if isinstance(input_width, bool) or not isinstance(input_width, int):
            raise TypeError("input_width must be an integer")
        if input_width <= 0:
            raise MLPValidationError("input_width must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        widths = validate_layer_widths(layer_widths)
        rng = Random(seed)
        layer_inputs = (input_width, *widths[:-1])
        self.input_width = input_width
        self.layer_widths = widths
        self.layers = tuple(
            DenseLayer(
                previous_width,
                width,
                rng,
                name=f"layer{index}",
                nonlinear=True,
            )
            for index, (previous_width, width) in enumerate(zip(layer_inputs, widths))
        )

    def __call__(self, inputs: Sequence[Value]) -> tuple[Value, ...]:
        if len(inputs) != self.input_width:
            raise MLPValidationError(
                f"MLP expected {self.input_width} inputs; received {len(inputs)}"
            )
        values = tuple(inputs)
        for layer in self.layers:
            values = layer(values)
        return values

    def named_parameters(self) -> tuple[tuple[str, Value], ...]:
        return tuple(
            parameter for layer in self.layers for parameter in layer.named_parameters()
        )

    def parameters(self) -> tuple[Value, ...]:
        return tuple(value for _, value in self.named_parameters())


def parameter_count(input_width: int, layer_widths: Iterable[int]) -> int:
    """Return the affine parameter count implied by an architecture."""

    if isinstance(input_width, bool) or not isinstance(input_width, int):
        raise TypeError("input_width must be an integer")
    if input_width <= 0:
        raise MLPValidationError("input_width must be positive")
    widths = validate_layer_widths(layer_widths)
    return sum(
        output_width * (previous_width + 1)
        for previous_width, output_width in zip((input_width, *widths[:-1]), widths)
    )


def snapshot_parameters(model: MLP) -> tuple[ParameterSnapshot, ...]:
    """Capture a stable, serializable model state."""

    if not isinstance(model, MLP):
        raise TypeError("model must be an MLP")
    return tuple(
        ParameterSnapshot(name, parameter.data)
        for name, parameter in model.named_parameters()
    )


def restore_parameters(model: MLP, snapshot: Iterable[ParameterSnapshot]) -> None:
    """Restore a complete named state, rejecting missing or reordered values."""

    if not isinstance(model, MLP):
        raise TypeError("model must be an MLP")
    state = tuple(snapshot)
    current = model.named_parameters()
    if len(state) != len(current):
        raise MLPValidationError("snapshot parameter count does not match model")
    if [item.name for item in state] != [name for name, _ in current]:
        raise MLPValidationError("snapshot parameter names do not match model")
    for item, (_, parameter) in zip(state, current):
        parameter.data = _finite_scalar(f"snapshot {item.name}", item.value)


def zero_parameter_gradients(parameters: Iterable[Value]) -> None:
    """Clear trainable gradients before each new backward pass."""

    for parameter in parameters:
        if not isinstance(parameter, Value):
            raise TypeError("parameters must contain Value objects")
        parameter.grad = 0.0


def prediction(model: MLP, inputs: Sequence[Real]) -> Value:
    """Run one scalar-output prediction from finite Python inputs."""

    if model.layer_widths[-1] != 1:
        raise MLPValidationError("prediction requires a one-output MLP")
    values = tuple(
        Value(_finite_scalar(f"input {index}", value), label=f"input{index}")
        for index, value in enumerate(inputs)
    )
    return model(values)[0]


def loss_components(
    model: MLP,
    examples: Iterable[TrainingExample],
    *,
    l2_coefficient: Real = 0.0,
) -> tuple[Value, Value, Value, tuple[Value, ...]]:
    """Build mean squared error, L2 penalty, and total scalar loss."""

    dataset = validate_dataset(examples, model.input_width)
    coefficient = _finite_scalar("l2_coefficient", l2_coefficient)
    if coefficient < 0.0:
        raise MLPValidationError("l2_coefficient must be non-negative")
    predictions = tuple(prediction(model, example.inputs) for example in dataset)
    squared_errors = tuple(
        (output - example.target) ** 2.0
        for output, example in zip(predictions, dataset)
    )
    data_loss = squared_errors[0]
    for error in squared_errors[1:]:
        data_loss = data_loss + error
    data_loss = data_loss / len(squared_errors)
    data_loss.label = "data_loss"

    parameters = model.parameters()
    squared_norm = parameters[0] * parameters[0]
    for parameter in parameters[1:]:
        squared_norm = squared_norm + parameter * parameter
    penalty = squared_norm * coefficient
    penalty.label = "l2_penalty"
    total = data_loss + penalty
    total.label = "total_loss"
    return total, data_loss, penalty, predictions


def binary_accuracy(
    predictions: Sequence[Real], examples: Sequence[TrainingExample]
) -> float:
    """Score sign-based binary predictions, treating zero as positive."""

    if len(predictions) != len(examples):
        raise MLPValidationError("prediction count must match example count")
    if not examples:
        raise MLPValidationError("examples must not be empty")
    matches = 0
    for index, (output, example) in enumerate(zip(predictions, examples)):
        value = _finite_scalar(f"prediction {index}", output)
        predicted_label = 1.0 if value >= 0.0 else -1.0
        matches += predicted_label == example.target
    return matches / len(examples)


def gradient_l2_norm(parameters: Iterable[Value]) -> float:
    """Return the Euclidean norm of the current parameter-gradient vector."""

    gradients: list[float] = []
    for parameter in parameters:
        if not isinstance(parameter, Value):
            raise TypeError("parameters must contain Value objects")
        gradients.append(_finite_scalar("parameter gradient", parameter.grad))
    if not gradients:
        raise MLPValidationError("parameter collection must not be empty")
    return sqrt(sum(gradient * gradient for gradient in gradients))


def clipping_scale(gradient_norm: Real, maximum_norm: Real | None) -> float:
    """Return a global gradient scale in ``(0, 1]``."""

    norm = _finite_scalar("gradient_norm", gradient_norm)
    if norm < 0.0:
        raise MLPValidationError("gradient_norm must be non-negative")
    if maximum_norm is None or norm == 0.0:
        return 1.0
    maximum = _finite_scalar("maximum_norm", maximum_norm)
    if maximum <= 0.0:
        raise MLPValidationError("maximum_norm must be positive")
    return min(1.0, maximum / norm)


def sgd_step(
    parameters: Iterable[Value], learning_rate: Real, *, gradient_scale: Real = 1.0
) -> None:
    """Apply one finite simultaneous SGD update."""

    rate = _finite_scalar("learning_rate", learning_rate)
    scale = _finite_scalar("gradient_scale", gradient_scale)
    if rate <= 0.0:
        raise MLPValidationError("learning_rate must be positive")
    if not 0.0 < scale <= 1.0:
        raise MLPValidationError("gradient_scale must be in (0, 1]")
    normalized = tuple(parameters)
    if not normalized:
        raise MLPValidationError("parameter collection must not be empty")
    candidates: list[float] = []
    for parameter in normalized:
        if not isinstance(parameter, Value):
            raise TypeError("parameters must contain Value objects")
        candidate = parameter.data - rate * scale * parameter.grad
        candidates.append(_finite_scalar("updated parameter", candidate))
    for parameter, candidate in zip(normalized, candidates):
        parameter.data = candidate


def _training_point(
    step: int,
    total_loss: Value,
    data_loss: Value,
    penalty: Value,
    predictions: Sequence[Value],
    examples: Sequence[TrainingExample],
    parameters: Sequence[Value],
    clip_norm: float | None,
) -> TrainingPoint:
    norm = gradient_l2_norm(parameters)
    scale = clipping_scale(norm, clip_norm)
    values = tuple(output.data for output in predictions)
    return TrainingPoint(
        step=step,
        total_loss=total_loss.data,
        data_loss=data_loss.data,
        l2_penalty=penalty.data,
        gradient_l2_norm=norm,
        gradient_scale=scale,
        accuracy=binary_accuracy(values, examples),
        predictions=values,
    )


def train_mlp(
    model: MLP,
    examples: Iterable[TrainingExample],
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train with deterministic full-batch SGD and record an auditable trace."""

    if not isinstance(model, MLP):
        raise TypeError("model must be an MLP")
    settings = validate_training_config(config or TrainingConfig())
    if model.layer_widths != settings.layer_widths:
        raise MLPValidationError("config layer widths must match the model")
    dataset = validate_dataset(examples, model.input_width)
    parameters = model.parameters()
    expected = parameter_count(model.input_width, model.layer_widths)
    if len(parameters) != expected:
        raise MLPValidationError("model parameter count does not match architecture")

    history: list[TrainingPoint] = []
    for step in range(settings.steps + 1):
        total, data_loss, penalty, outputs = loss_components(
            model, dataset, l2_coefficient=settings.l2_coefficient
        )
        zero_parameter_gradients(parameters)
        total.backward()
        point = _training_point(
            step,
            total,
            data_loss,
            penalty,
            outputs,
            dataset,
            parameters,
            settings.gradient_clip_norm,
        )
        if step == 0 or step == settings.steps or step % settings.record_every == 0:
            history.append(point)
        if step < settings.steps:
            sgd_step(
                parameters,
                settings.learning_rate,
                gradient_scale=point.gradient_scale,
            )

    return TrainingResult(
        config=settings,
        parameter_count=len(parameters),
        history=tuple(history),
        final_parameters=snapshot_parameters(model),
    )


def check_parameter_gradients(
    model: MLP,
    examples: Iterable[TrainingExample],
    *,
    l2_coefficient: Real = 0.0,
    epsilon: Real = 1e-6,
    maximum_checks: int = 7,
    absolute_tolerance: Real = 1e-7,
    relative_tolerance: Real = 1e-5,
) -> tuple[GradientProbe, ...]:
    """Compare selected parameter gradients with centered finite differences."""

    dataset = validate_dataset(examples, model.input_width)
    step = _finite_scalar("epsilon", epsilon)
    atol = _finite_scalar("absolute_tolerance", absolute_tolerance)
    rtol = _finite_scalar("relative_tolerance", relative_tolerance)
    coefficient = _finite_scalar("l2_coefficient", l2_coefficient)
    if step <= 0.0 or atol < 0.0 or rtol < 0.0 or coefficient < 0.0:
        raise MLPValidationError("gradient-check controls are out of range")
    if isinstance(maximum_checks, bool) or not isinstance(maximum_checks, int):
        raise TypeError("maximum_checks must be an integer")
    if maximum_checks <= 0:
        raise MLPValidationError("maximum_checks must be positive")

    named = model.named_parameters()
    total, _, _, _ = loss_components(model, dataset, l2_coefficient=coefficient)
    zero_parameter_gradients(value for _, value in named)
    total.backward()
    analytic = {name: value.grad for name, value in named}
    original = snapshot_parameters(model)
    check_count = min(maximum_checks, len(named))
    selected_indices = tuple(
        round(index * (len(named) - 1) / max(1, check_count - 1))
        for index in range(check_count)
    )
    probes: list[GradientProbe] = []
    try:
        for index in selected_indices:
            name, parameter = named[index]
            center = parameter.data
            parameter.data = center + step
            high = loss_components(model, dataset, l2_coefficient=coefficient)[0].data
            parameter.data = center - step
            low = loss_components(model, dataset, l2_coefficient=coefficient)[0].data
            parameter.data = center
            numerical = (high - low) / (2.0 * step)
            error = abs(analytic[name] - numerical)
            tolerance = atol + rtol * max(abs(analytic[name]), abs(numerical))
            probes.append(
                GradientProbe(
                    name=name,
                    analytic=analytic[name],
                    numerical=numerical,
                    absolute_error=error,
                    tolerance=tolerance,
                    passed=error <= tolerance,
                )
            )
    finally:
        restore_parameters(model, original)
    return tuple(probes)


def tiny_dataset() -> tuple[TrainingExample, ...]:
    """Return the fixed four-example, three-feature Day 17 dataset."""

    return (
        TrainingExample((2.0, 3.0, -1.0), 1.0),
        TrainingExample((3.0, -1.0, 0.5), -1.0),
        TrainingExample((0.5, 1.0, 1.0), -1.0),
        TrainingExample((1.0, 1.0, -1.0), 1.0),
    )


def run_mlp_experiment(config: TrainingConfig | None = None) -> MLPExperiment:
    """Gradient-check and train the fixed public-safe Day 17 MLP."""

    settings = validate_training_config(config or TrainingConfig())
    dataset = tiny_dataset()
    model = MLP(3, settings.layer_widths, seed=settings.seed)
    probes = check_parameter_gradients(
        model,
        dataset,
        l2_coefficient=settings.l2_coefficient,
    )
    training = train_mlp(model, dataset, settings)
    return MLPExperiment(3, dataset, probes, training)


def experiment_metrics(
    experiment: MLPExperiment,
) -> dict[str, int | float | bool]:
    """Summarize loss reduction, classification, clipping, and gradient checks."""

    first = experiment.training.history[0]
    last = experiment.training.history[-1]
    max_probe_error = max(probe.absolute_error for probe in experiment.gradient_probes)
    return {
        "example_count": len(experiment.dataset),
        "parameter_count": experiment.training.parameter_count,
        "record_count": len(experiment.training.history),
        "initial_loss": first.total_loss,
        "final_loss": last.total_loss,
        "loss_reduction_fraction": 1.0 - last.total_loss / first.total_loss,
        "initial_accuracy": first.accuracy,
        "final_accuracy": last.accuracy,
        "clipped_record_count": sum(
            point.gradient_scale < 1.0 for point in experiment.training.history
        ),
        "gradient_probe_count": len(experiment.gradient_probes),
        "max_gradient_probe_error": max_probe_error,
        "all_gradient_probes_pass": all(
            probe.passed for probe in experiment.gradient_probes
        ),
    }


def validate_experiment(experiment: MLPExperiment) -> None:
    """Reject incomplete, inconsistent, or unsuccessful training evidence."""

    dataset = validate_dataset(experiment.dataset, experiment.input_width)
    result = experiment.training
    validate_training_config(result.config)
    if result.parameter_count != parameter_count(
        experiment.input_width, result.config.layer_widths
    ):
        raise MLPValidationError("experiment parameter count is inconsistent")
    if len(result.final_parameters) != result.parameter_count:
        raise MLPValidationError("final parameter snapshot is incomplete")
    if not experiment.gradient_probes:
        raise MLPValidationError("experiment must include gradient probes")
    failures = [probe.name for probe in experiment.gradient_probes if not probe.passed]
    if failures:
        raise MLPValidationError(f"gradient probes failed: {', '.join(failures)}")
    if not result.history or result.history[0].step != 0:
        raise MLPValidationError("training history must begin at step zero")
    if result.history[-1].step != result.config.steps:
        raise MLPValidationError("training history must include the final step")
    if any(
        left.step >= right.step
        for left, right in zip(result.history, result.history[1:])
    ):
        raise MLPValidationError("training history steps must increase")
    first, last = result.history[0], result.history[-1]
    if last.total_loss >= first.total_loss:
        raise MLPValidationError("final loss must be lower than initial loss")
    if last.total_loss > first.total_loss * 0.25:
        raise MLPValidationError("training must reduce loss by at least 75%")
    if last.accuracy != 1.0:
        raise MLPValidationError("final training accuracy must be 100%")
    if len(last.predictions) != len(dataset):
        raise MLPValidationError("final predictions must cover every example")
    for point in result.history:
        values = (
            point.total_loss,
            point.data_loss,
            point.l2_penalty,
            point.gradient_l2_norm,
            point.gradient_scale,
            point.accuracy,
            *point.predictions,
        )
        if any(not isfinite(value) for value in values):
            raise MLPValidationError("training history must contain finite values")


def render_architecture_mermaid(input_width: int, widths: Iterable[int]) -> str:
    """Render layer widths and trainable affine parameter counts."""

    normalized = validate_layer_widths(widths)
    nodes = (
        ("input", input_width),
        *tuple((f"layer{index}", width) for index, width in enumerate(normalized)),
    )
    lines = ["flowchart LR"]
    for name, width in nodes:
        lines.append(f'    {name}["{name}<br/>{width} scalars"]')
    for (left_name, left_width), (right_name, right_width) in pairwise(nodes):
        count = right_width * (left_width + 1)
        lines.append(f'    {left_name} -->|"{count} parameters"| {right_name}')
    return "\n".join(lines)


def render_mlp_markdown(experiment: MLPExperiment) -> str:
    """Render deterministic Day 17 evidence as a compact Markdown report."""

    validate_experiment(experiment)
    metrics = experiment_metrics(experiment)
    lines = [
        "# Day 17 scalar MLP training report",
        "",
        (
            "A 3→4→4→1 tanh network is trained with full-batch gradient descent "
            "on four fixed examples. Every parameter is a scalar `Value` in the "
            "Day 15 autodiff graph."
        ),
        "",
        "## Result",
        "",
        f"- parameters: `{metrics['parameter_count']}`",
        f"- initial loss: `{float(metrics['initial_loss']):.9f}`",
        f"- final loss: `{float(metrics['final_loss']):.9f}`",
        f"- loss reduction: `{100.0 * float(metrics['loss_reduction_fraction']):.2f}%`",
        f"- final sign accuracy: `{100.0 * float(metrics['final_accuracy']):.1f}%`",
        (
            f"- maximum gradient-probe error: "
            f"`{float(metrics['max_gradient_probe_error']):.3e}`"
        ),
        "",
        "## Training trace",
        "",
        "| Step | Total loss | Data loss | L2 penalty | Gradient norm | Scale | Accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in experiment.training.history:
        lines.append(
            f"| {point.step} | {point.total_loss:.9f} | {point.data_loss:.9f} | "
            f"{point.l2_penalty:.3e} | {point.gradient_l2_norm:.6f} | "
            f"{point.gradient_scale:.6f} | {100.0 * point.accuracy:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Final predictions",
            "",
            "| Example | Inputs | Target | Prediction | Sign correct |",
            "|---:|---|---:|---:|:---:|",
        ]
    )
    final_predictions = experiment.training.history[-1].predictions
    for index, (example, output) in enumerate(
        zip(experiment.dataset, final_predictions), start=1
    ):
        sign = 1.0 if output >= 0.0 else -1.0
        lines.append(
            f"| {index} | `{example.inputs}` | {example.target:+.0f} | "
            f"{output:+.6f} | {'yes' if sign == example.target else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Parameter gradient probes",
            "",
            "| Parameter | Autodiff | Finite difference | Error | Tolerance | Pass |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for probe in experiment.gradient_probes:
        lines.append(
            f"| `{probe.name}` | {probe.analytic:.9f} | {probe.numerical:.9f} | "
            f"{probe.absolute_error:.3e} | {probe.tolerance:.3e} | "
            f"{'yes' if probe.passed else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Architecture",
            "",
            "```mermaid",
            render_architecture_mermaid(
                experiment.input_width, experiment.training.config.layer_widths
            ),
            "```",
            "",
            "## Scope",
            "",
            (
                "This deterministic run verifies scalar graph composition, "
                "parameter collection, explicit gradient clearing, "
                "finite-difference agreement, and decreasing training loss. It "
                "does not claim lecture completion, a learner-typed "
                "implementation, generalization, or production use."
            ),
            "",
        ]
    )
    return "\n".join(lines)
