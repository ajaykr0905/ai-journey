"""Day 15: a deterministic scalar reverse-mode automatic differentiation engine.

Each :class:`Value` stores a scalar and the operation that produced it.  The
resulting parent links form a directed acyclic computation graph.  Reverse-mode
autodiff visits that graph in reverse topological order and applies the chain
rule one local derivative at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp as math_exp
from math import isfinite, tanh as math_tanh
from numbers import Real
from typing import Callable, Mapping


class GraphCycleError(ValueError):
    """Raised when a malformed computation graph contains a cycle."""


@dataclass(frozen=True)
class NodeSnapshot:
    """Stable, serializable description of one scalar graph node."""

    node_id: str
    data: float
    grad: float
    label: str
    operation: str


@dataclass(frozen=True)
class GraphSnapshot:
    """Nodes and parent-to-result edges in stable topological order."""

    nodes: tuple[NodeSnapshot, ...]
    edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GradientCheck:
    """Analytic and centered-finite-difference gradients for one input."""

    name: str
    analytic: float
    numerical: float
    absolute_error: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class AutodiffExperiment:
    """Inspectable outputs from the deterministic Day 15 experiment."""

    output: float
    graph: GraphSnapshot
    gradient_checks: tuple[GradientCheck, ...]


def _as_finite_scalar(name: str, value: object) -> float:
    """Normalize one real, finite, non-boolean scalar."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class Value:
    """A scalar value plus the local rule needed for reverse-mode autodiff."""

    def __init__(
        self,
        data: Real,
        _children: tuple[Value, ...] = (),
        _operation: str = "",
        *,
        label: str = "",
    ) -> None:
        self.data = _as_finite_scalar("data", data)
        self.grad = 0.0
        if not isinstance(label, str):
            raise TypeError("label must be a string")
        if not isinstance(_operation, str):
            raise TypeError("operation must be a string")
        if not all(isinstance(child, Value) for child in _children):
            raise TypeError("children must contain only Value objects")
        self.label = label
        self.operation = _operation
        self._parents = tuple(_children)
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        details = f"data={self.data:g}, grad={self.grad:g}"
        if self.label:
            details += f", label={self.label!r}"
        return f"Value({details})"

    @staticmethod
    def _coerce(other: Real | Value) -> Value:
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: Real | Value) -> Value:
        right = self._coerce(other)
        result = Value(self.data + right.data, (self, right), "+")

        def backward() -> None:
            self.grad += result.grad
            right.grad += result.grad

        result._backward = backward
        return result

    def __radd__(self, other: Real | Value) -> Value:
        return self + other

    def __mul__(self, other: Real | Value) -> Value:
        right = self._coerce(other)
        result = Value(self.data * right.data, (self, right), "*")

        def backward() -> None:
            self.grad += right.data * result.grad
            right.grad += self.data * result.grad

        result._backward = backward
        return result

    def __rmul__(self, other: Real | Value) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Real | Value) -> Value:
        return self + (-self._coerce(other))

    def __rsub__(self, other: Real | Value) -> Value:
        return self._coerce(other) - self

    def __pow__(self, exponent: Real) -> Value:
        power = _as_finite_scalar("exponent", exponent)
        try:
            data = self.data**power
            local_derivative = (
                0.0 if power == 0.0 else power * self.data ** (power - 1.0)
            )
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError(
                "power must have a finite real value and derivative"
            ) from exc
        if isinstance(data, complex) or not isfinite(float(data)):
            raise ValueError("power must produce a finite real value")
        if isinstance(local_derivative, complex) or not isfinite(
            float(local_derivative)
        ):
            raise ValueError("power derivative must be finite and real")
        result = Value(float(data), (self,), f"**{power:g}")

        def backward() -> None:
            self.grad += float(local_derivative) * result.grad

        result._backward = backward
        return result

    def __truediv__(self, other: Real | Value) -> Value:
        right = self._coerce(other)
        if right.data == 0.0:
            raise ZeroDivisionError("cannot divide by zero")
        return self * right**-1.0

    def __rtruediv__(self, other: Real | Value) -> Value:
        return self._coerce(other) / self

    def exp(self) -> Value:
        """Return ``e**self`` with its local derivative."""

        try:
            data = math_exp(self.data)
        except OverflowError as exc:
            raise ValueError("exp result must be finite") from exc
        if not isfinite(data):
            raise ValueError("exp result must be finite")
        result = Value(data, (self,), "exp")

        def backward() -> None:
            self.grad += data * result.grad

        result._backward = backward
        return result

    def tanh(self) -> Value:
        """Apply hyperbolic tangent with derivative ``1 - tanh(x)^2``."""

        data = math_tanh(self.data)
        result = Value(data, (self,), "tanh")

        def backward() -> None:
            self.grad += (1.0 - data * data) * result.grad

        result._backward = backward
        return result

    def relu(self) -> Value:
        """Apply ReLU, using derivative zero at the non-differentiable origin."""

        data = max(0.0, self.data)
        result = Value(data, (self,), "ReLU")

        def backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * result.grad

        result._backward = backward
        return result

    def backward(self, gradient: Real = 1.0) -> None:
        """Propagate a seed gradient through the graph in reverse topological order."""

        seed = _as_finite_scalar("gradient", gradient)
        ordered = topological_sort(self)
        for node in ordered:
            node.grad = 0.0
        self.grad = seed
        for node in reversed(ordered):
            node._backward()


def topological_sort(root: Value) -> tuple[Value, ...]:
    """Return every reachable node once, with parents before their result."""

    if not isinstance(root, Value):
        raise TypeError("root must be a Value")
    order: list[Value] = []
    state: dict[Value, int] = {}

    def visit(node: Value) -> None:
        status = state.get(node, 0)
        if status == 1:
            raise GraphCycleError("computation graph must be acyclic")
        if status == 2:
            return
        state[node] = 1
        for parent in node._parents:
            visit(parent)
        state[node] = 2
        order.append(node)

    visit(root)
    return tuple(order)


def zero_grad(root: Value) -> None:
    """Clear gradients for every node reachable from ``root``."""

    for node in topological_sort(root):
        node.grad = 0.0


def snapshot_graph(root: Value) -> GraphSnapshot:
    """Capture values, gradients, operations, and edges with stable node IDs."""

    ordered = topological_sort(root)
    identifiers = {node: f"n{index}" for index, node in enumerate(ordered)}
    nodes = tuple(
        NodeSnapshot(
            node_id=identifiers[node],
            data=node.data,
            grad=node.grad,
            label=node.label,
            operation=node.operation,
        )
        for node in ordered
    )
    edges = tuple(
        (identifiers[parent], identifiers[node])
        for node in ordered
        for parent in node._parents
    )
    return GraphSnapshot(nodes=nodes, edges=edges)


def render_mermaid(graph: GraphSnapshot) -> str:
    """Render a graph snapshot as a Mermaid flowchart."""

    lines = ["flowchart LR"]
    for node in graph.nodes:
        name = node.label or node.operation or "leaf"
        safe_name = name.replace('"', "'")
        lines.append(
            f'    {node.node_id}["{safe_name}<br/>value={node.data:.6f}<br/>'
            f'grad={node.grad:.6f}"]'
        )
    for parent, result in graph.edges:
        lines.append(f"    {parent} --> {result}")
    return "\n".join(lines)


def numerical_derivative(
    function: Callable[[float], float], value: Real, *, epsilon: Real = 1e-6
) -> float:
    """Estimate one derivative with a centered finite difference."""

    point = _as_finite_scalar("value", value)
    step = _as_finite_scalar("epsilon", epsilon)
    if step <= 0.0:
        raise ValueError("epsilon must be positive")
    high = _as_finite_scalar("function result", function(point + step))
    low = _as_finite_scalar("function result", function(point - step))
    return (high - low) / (2.0 * step)


Expression = Callable[[Mapping[str, Value]], Value]


def gradient_check(
    expression: Expression,
    inputs: Mapping[str, Real],
    *,
    epsilon: Real = 1e-6,
    absolute_tolerance: Real = 1e-7,
    relative_tolerance: Real = 1e-5,
) -> tuple[GradientCheck, ...]:
    """Compare autodiff gradients with independently rebuilt finite differences."""

    if not callable(expression):
        raise TypeError("expression must be callable")
    if not inputs:
        raise ValueError("inputs must not be empty")
    normalized: dict[str, float] = {}
    for name, value in inputs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("input names must be non-empty strings")
        normalized[name] = _as_finite_scalar(f"input {name}", value)
    step = _as_finite_scalar("epsilon", epsilon)
    atol = _as_finite_scalar("absolute_tolerance", absolute_tolerance)
    rtol = _as_finite_scalar("relative_tolerance", relative_tolerance)
    if step <= 0.0 or atol < 0.0 or rtol < 0.0:
        raise ValueError("epsilon must be positive and tolerances must be non-negative")

    variables = {name: Value(value, label=name) for name, value in normalized.items()}
    output = expression(variables)
    if not isinstance(output, Value):
        raise TypeError("expression must return a Value")
    output.label = output.label or "output"
    output.backward()

    checks: list[GradientCheck] = []
    for name in normalized:

        def evaluate(candidate: float, *, selected: str = name) -> float:
            points = {
                key: Value(candidate if key == selected else value, label=key)
                for key, value in normalized.items()
            }
            result = expression(points)
            if not isinstance(result, Value):
                raise TypeError("expression must return a Value")
            return result.data

        numerical = numerical_derivative(evaluate, normalized[name], epsilon=step)
        analytic = variables[name].grad
        error = abs(analytic - numerical)
        tolerance = atol + rtol * max(abs(analytic), abs(numerical))
        checks.append(
            GradientCheck(
                name=name,
                analytic=analytic,
                numerical=numerical,
                absolute_error=error,
                tolerance=tolerance,
                passed=error <= tolerance,
            )
        )
    return tuple(checks)


def tiny_neuron_loss(values: Mapping[str, Value]) -> Value:
    """Build a two-input tanh neuron followed by squared-error loss."""

    required = ("x1", "x2", "w1", "w2", "bias", "target")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"missing values: {', '.join(missing)}")
    prediction = (
        values["x1"] * values["w1"] + values["x2"] * values["w2"] + values["bias"]
    ).tanh()
    prediction.label = "prediction"
    loss = (prediction - values["target"]) ** 2.0
    loss.label = "loss"
    return loss


def default_experiment_inputs() -> dict[str, float]:
    """Return the fixed public-safe inputs used by the Day 15 experiment."""

    return {
        "x1": 0.75,
        "x2": -1.25,
        "w1": 0.40,
        "w2": -0.20,
        "bias": 0.10,
        "target": 0.80,
    }


def run_autodiff_experiment() -> AutodiffExperiment:
    """Run the tiny neuron, backpropagate, snapshot it, and check every input."""

    raw_inputs = default_experiment_inputs()
    variables = {name: Value(value, label=name) for name, value in raw_inputs.items()}
    loss = tiny_neuron_loss(variables)
    loss.backward()
    graph = snapshot_graph(loss)
    checks = gradient_check(tiny_neuron_loss, raw_inputs)
    return AutodiffExperiment(loss.data, graph, checks)


def experiment_metrics(experiment: AutodiffExperiment) -> dict[str, int | float | bool]:
    """Summarize graph size and gradient-check accuracy."""

    max_error = max(check.absolute_error for check in experiment.gradient_checks)
    return {
        "output": experiment.output,
        "node_count": len(experiment.graph.nodes),
        "edge_count": len(experiment.graph.edges),
        "gradient_check_count": len(experiment.gradient_checks),
        "max_gradient_error": max_error,
        "all_gradients_pass": all(check.passed for check in experiment.gradient_checks),
    }


def validate_experiment(experiment: AutodiffExperiment) -> None:
    """Fail loudly when the graph or any independent gradient check is invalid."""

    if not experiment.graph.nodes or not experiment.graph.edges:
        raise ValueError("experiment graph must contain nodes and edges")
    node_ids = {node.node_id for node in experiment.graph.nodes}
    if len(node_ids) != len(experiment.graph.nodes):
        raise ValueError("graph node IDs must be unique")
    if any(
        left not in node_ids or right not in node_ids
        for left, right in experiment.graph.edges
    ):
        raise ValueError("every graph edge must reference known nodes")
    if not experiment.gradient_checks:
        raise ValueError("experiment must contain gradient checks")
    failures = [check.name for check in experiment.gradient_checks if not check.passed]
    if failures:
        raise ValueError(f"gradient checks failed: {', '.join(failures)}")


def render_autodiff_markdown(experiment: AutodiffExperiment) -> str:
    """Render the deterministic graph and gradient comparison as Markdown."""

    validate_experiment(experiment)
    metrics = experiment_metrics(experiment)
    lines = [
        "# Day 15 scalar autodiff report",
        "",
        "A two-input tanh neuron feeds a squared-error loss. The graph is built "
        "from scalar operations and differentiated in reverse topological order.",
        "",
        "## Deterministic result",
        "",
        f"- loss: `{experiment.output:.12f}`",
        f"- graph nodes: `{metrics['node_count']}`",
        f"- graph edges: `{metrics['edge_count']}`",
        f"- maximum gradient-check error: `{float(metrics['max_gradient_error']):.3e}`",
        f"- all gradient checks passed: `{metrics['all_gradients_pass']}`",
        "",
        "## Gradient checks",
        "",
        "| Input | Autodiff | Finite difference | Absolute error | Tolerance | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for check in experiment.gradient_checks:
        lines.append(
            f"| `{check.name}` | {check.analytic:.9f} | {check.numerical:.9f} | "
            f"{check.absolute_error:.3e} | {check.tolerance:.3e} | "
            f"{'yes' if check.passed else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Computation graph",
            "",
            "```mermaid",
            render_mermaid(experiment.graph),
            "```",
            "",
            "## Scope",
            "",
            "This report verifies a small scalar engine against centered finite "
            "differences. It does not claim lecture completion, tensor support, "
            "production performance, or a completed MLP milestone.",
            "",
        ]
    )
    return "\n".join(lines)
