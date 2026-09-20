"""Day 16: an inspectable manual reverse pass for scalar add/multiply graphs.

The implementation intentionally does not call :meth:`Value.backward` to
compute its manual gradients.  It evaluates an operation table, writes down
each local derivative, and accumulates one chain-rule contribution per edge.
The Day 15 autodiff engine is used only as an independent comparison oracle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .scalar_autodiff import Value


class ManualGraphError(ValueError):
    """Raised when a manual computation graph is invalid."""


@dataclass(frozen=True)
class NodeSpec:
    """Declarative description of one scalar graph node."""

    name: str
    operation: str
    parents: tuple[str, ...] = ()
    value: float | None = None


@dataclass(frozen=True)
class ForwardStep:
    """One evaluated node in topological order."""

    name: str
    operation: str
    parents: tuple[str, ...]
    parent_values: tuple[float, ...]
    value: float


@dataclass(frozen=True)
class ReverseStep:
    """One edge-level chain-rule contribution during the reverse pass."""

    result: str
    parent: str
    parent_index: int
    upstream_gradient: float
    local_derivative: float
    contribution: float
    gradient_before: float
    gradient_after: float


@dataclass(frozen=True)
class ManualTrace:
    """Complete forward values, reverse steps, and accumulated gradients."""

    root: str
    seed: float
    forward_steps: tuple[ForwardStep, ...]
    reverse_steps: tuple[ReverseStep, ...]
    gradients: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class GradientComparison:
    """Manual, autodiff, and closed-form gradients for one input."""

    name: str
    manual: float
    autodiff: float
    closed_form: float
    manual_autodiff_error: float
    manual_closed_form_error: float
    passed: bool


@dataclass(frozen=True)
class ManualBackpropExperiment:
    """Deterministic Day 16 graph, trace, and independent comparisons."""

    nodes: tuple[NodeSpec, ...]
    trace: ManualTrace
    comparisons: tuple[GradientComparison, ...]


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_graph(nodes: Iterable[NodeSpec], root: str) -> tuple[NodeSpec, ...]:
    """Validate and normalize an add/multiply computation graph."""

    normalized = tuple(nodes)
    if not normalized:
        raise ManualGraphError("graph must contain at least one node")
    if not isinstance(root, str) or not root:
        raise ManualGraphError("root must be a non-empty string")

    by_name: dict[str, NodeSpec] = {}
    for node in normalized:
        if not isinstance(node, NodeSpec):
            raise TypeError("nodes must contain only NodeSpec objects")
        if not node.name:
            raise ManualGraphError("node names must be non-empty")
        if node.name in by_name:
            raise ManualGraphError(f"duplicate node name: {node.name}")
        if node.operation not in {"input", "add", "mul"}:
            raise ManualGraphError(
                f"unsupported operation for {node.name}: {node.operation}"
            )
        if node.operation == "input":
            if node.parents:
                raise ManualGraphError(f"input {node.name} cannot have parents")
            if node.value is None:
                raise ManualGraphError(f"input {node.name} must have a value")
            _finite_scalar(f"value for {node.name}", node.value)
        else:
            if len(node.parents) != 2:
                raise ManualGraphError(
                    f"{node.operation} node {node.name} must have two parents"
                )
            if node.value is not None:
                raise ManualGraphError(
                    f"operation node {node.name} cannot declare a value"
                )
        by_name[node.name] = node

    if root not in by_name:
        raise ManualGraphError(f"unknown root: {root}")
    for node in normalized:
        missing = [parent for parent in node.parents if parent not in by_name]
        if missing:
            raise ManualGraphError(
                f"node {node.name} has unknown parents: {', '.join(missing)}"
            )
    topological_order(normalized, root)
    return normalized


def topological_order(nodes: Iterable[NodeSpec], root: str) -> tuple[str, ...]:
    """Return reachable node names with every parent before its result."""

    normalized = tuple(nodes)
    by_name = {node.name: node for node in normalized}
    if len(by_name) != len(normalized):
        raise ManualGraphError("node names must be unique")
    if root not in by_name:
        raise ManualGraphError(f"unknown root: {root}")

    ordered: list[str] = []
    state: dict[str, int] = {}

    def visit(name: str) -> None:
        if name not in by_name:
            raise ManualGraphError(f"unknown parent: {name}")
        status = state.get(name, 0)
        if status == 1:
            raise ManualGraphError("manual computation graph must be acyclic")
        if status == 2:
            return
        state[name] = 1
        for parent in by_name[name].parents:
            visit(parent)
        state[name] = 2
        ordered.append(name)

    visit(root)
    return tuple(ordered)


def apply_operation(operation: str, parent_values: tuple[float, ...]) -> float:
    """Evaluate a supported scalar operation from already computed inputs."""

    if len(parent_values) != 2:
        raise ManualGraphError("add and mul operations require two parent values")
    left = _finite_scalar("left parent", parent_values[0])
    right = _finite_scalar("right parent", parent_values[1])
    if operation == "add":
        result = left + right
    elif operation == "mul":
        result = left * right
    else:
        raise ManualGraphError(f"unsupported operation: {operation}")
    return _finite_scalar("operation result", result)


def local_derivatives(
    operation: str, parent_values: tuple[float, ...]
) -> tuple[float, float]:
    """Return partial derivatives with respect to the left and right inputs."""

    if len(parent_values) != 2:
        raise ManualGraphError("local derivatives require two parent values")
    left = _finite_scalar("left parent", parent_values[0])
    right = _finite_scalar("right parent", parent_values[1])
    if operation == "add":
        return (1.0, 1.0)
    if operation == "mul":
        return (right, left)
    raise ManualGraphError(f"unsupported operation: {operation}")


def evaluate_forward(nodes: Iterable[NodeSpec], root: str) -> tuple[ForwardStep, ...]:
    """Evaluate reachable graph nodes and retain every intermediate value."""

    normalized = validate_graph(nodes, root)
    by_name = {node.name: node for node in normalized}
    values: dict[str, float] = {}
    steps: list[ForwardStep] = []
    for name in topological_order(normalized, root):
        node = by_name[name]
        parent_values = tuple(values[parent] for parent in node.parents)
        value = (
            _finite_scalar(f"value for {name}", node.value)
            if node.operation == "input"
            else apply_operation(node.operation, parent_values)
        )
        values[name] = value
        steps.append(
            ForwardStep(name, node.operation, node.parents, parent_values, value)
        )
    return tuple(steps)


def manual_backward(
    nodes: Iterable[NodeSpec], root: str, *, seed: Real = 1.0
) -> ManualTrace:
    """Apply the chain rule edge by edge in reverse topological order."""

    normalized = validate_graph(nodes, root)
    forward = evaluate_forward(normalized, root)
    by_name = {node.name: node for node in normalized}
    values = {step.name: step.value for step in forward}
    order = tuple(step.name for step in forward)
    seed_value = _finite_scalar("seed", seed)
    gradients = {name: 0.0 for name in order}
    gradients[root] = seed_value
    reverse_steps: list[ReverseStep] = []

    for result_name in reversed(order):
        node = by_name[result_name]
        if node.operation == "input":
            continue
        parent_values = tuple(values[parent] for parent in node.parents)
        partials = local_derivatives(node.operation, parent_values)
        upstream = gradients[result_name]
        for index, (parent, local) in enumerate(zip(node.parents, partials)):
            before = gradients[parent]
            contribution = upstream * local
            after = before + contribution
            gradients[parent] = after
            reverse_steps.append(
                ReverseStep(
                    result=result_name,
                    parent=parent,
                    parent_index=index,
                    upstream_gradient=upstream,
                    local_derivative=local,
                    contribution=contribution,
                    gradient_before=before,
                    gradient_after=after,
                )
            )

    return ManualTrace(
        root=root,
        seed=seed_value,
        forward_steps=forward,
        reverse_steps=tuple(reverse_steps),
        gradients=tuple((name, gradients[name]) for name in order),
    )


def gradient_map(trace: ManualTrace) -> dict[str, float]:
    """Return a convenient name-to-gradient mapping from a manual trace."""

    return dict(trace.gradients)


def build_tiny_graph(
    inputs: Mapping[str, Real] | None = None,
) -> tuple[NodeSpec, ...]:
    """Build ``y = ((a * b) + c) * d + a`` with fixed or supplied inputs."""

    raw = {"a": 2.0, "b": -3.0, "c": 10.0, "d": -2.0}
    if inputs is not None:
        if set(inputs) != set(raw):
            raise ManualGraphError("inputs must contain exactly a, b, c, and d")
        raw = {name: _finite_scalar(name, inputs[name]) for name in raw}
    return (
        *(NodeSpec(name, "input", value=value) for name, value in raw.items()),
        NodeSpec("product", "mul", ("a", "b")),
        NodeSpec("shifted", "add", ("product", "c")),
        NodeSpec("scaled", "mul", ("shifted", "d")),
        NodeSpec("y", "add", ("scaled", "a")),
    )


def closed_form_gradients(inputs: Mapping[str, Real]) -> dict[str, float]:
    """Evaluate derivatives of the tiny graph from its expanded formula."""

    if set(inputs) != {"a", "b", "c", "d"}:
        raise ManualGraphError("inputs must contain exactly a, b, c, and d")
    a = _finite_scalar("a", inputs["a"])
    b = _finite_scalar("b", inputs["b"])
    c = _finite_scalar("c", inputs["c"])
    d = _finite_scalar("d", inputs["d"])
    return {"a": b * d + 1.0, "b": a * d, "c": d, "d": a * b + c}


def autodiff_gradients(inputs: Mapping[str, Real]) -> dict[str, float]:
    """Differentiate the tiny graph with the separate Day 15 engine."""

    if set(inputs) != {"a", "b", "c", "d"}:
        raise ManualGraphError("inputs must contain exactly a, b, c, and d")
    values = {
        name: Value(_finite_scalar(name, value), label=name)
        for name, value in inputs.items()
    }
    output = ((values["a"] * values["b"] + values["c"]) * values["d"]) + values["a"]
    output.backward()
    return {name: values[name].grad for name in ("a", "b", "c", "d")}


def run_manual_backprop_experiment() -> ManualBackpropExperiment:
    """Run the fixed graph and compare three independent gradient paths."""

    inputs = {"a": 2.0, "b": -3.0, "c": 10.0, "d": -2.0}
    nodes = build_tiny_graph(inputs)
    trace = manual_backward(nodes, "y")
    manual = gradient_map(trace)
    autodiff = autodiff_gradients(inputs)
    closed = closed_form_gradients(inputs)
    comparisons = tuple(
        GradientComparison(
            name=name,
            manual=manual[name],
            autodiff=autodiff[name],
            closed_form=closed[name],
            manual_autodiff_error=abs(manual[name] - autodiff[name]),
            manual_closed_form_error=abs(manual[name] - closed[name]),
            passed=(
                abs(manual[name] - autodiff[name]) <= 1e-12
                and abs(manual[name] - closed[name]) <= 1e-12
            ),
        )
        for name in ("a", "b", "c", "d")
    )
    return ManualBackpropExperiment(nodes, trace, comparisons)


def experiment_metrics(
    experiment: ManualBackpropExperiment,
) -> dict[str, int | float | bool]:
    """Summarize trace size, output, accumulation, and comparison accuracy."""

    forward_values = {step.name: step.value for step in experiment.trace.forward_steps}
    max_error = max(
        max(item.manual_autodiff_error, item.manual_closed_form_error)
        for item in experiment.comparisons
    )
    accumulation_steps = sum(
        step.gradient_before != 0.0 for step in experiment.trace.reverse_steps
    )
    return {
        "node_count": len(experiment.trace.forward_steps),
        "reverse_step_count": len(experiment.trace.reverse_steps),
        "comparison_count": len(experiment.comparisons),
        "accumulation_step_count": accumulation_steps,
        "output": forward_values[experiment.trace.root],
        "max_comparison_error": max_error,
        "all_comparisons_pass": all(item.passed for item in experiment.comparisons),
    }


def validate_experiment(experiment: ManualBackpropExperiment) -> None:
    """Reject incomplete traces, bad accumulation, or mismatched gradients."""

    validate_graph(experiment.nodes, experiment.trace.root)
    if not experiment.trace.forward_steps:
        raise ManualGraphError("experiment must contain forward steps")
    if not experiment.trace.reverse_steps:
        raise ManualGraphError("experiment must contain reverse steps")
    if {item.name for item in experiment.comparisons} != {"a", "b", "c", "d"}:
        raise ManualGraphError("experiment must compare all four inputs")
    failures = [item.name for item in experiment.comparisons if not item.passed]
    if failures:
        raise ManualGraphError(f"gradient comparisons failed: {', '.join(failures)}")
    gradients = gradient_map(experiment.trace)
    if gradients[experiment.trace.root] != experiment.trace.seed:
        raise ManualGraphError("root gradient must equal the reverse-pass seed")
    a_steps = [step for step in experiment.trace.reverse_steps if step.parent == "a"]
    if len(a_steps) != 2 or a_steps[-1].gradient_before == 0.0:
        raise ManualGraphError("shared input a must accumulate two gradient paths")


def render_trace_mermaid(experiment: ManualBackpropExperiment) -> str:
    """Render forward values and final gradients as a Mermaid graph."""

    values = {step.name: step.value for step in experiment.trace.forward_steps}
    gradients = gradient_map(experiment.trace)
    lines = ["flowchart LR"]
    for node in experiment.nodes:
        if node.name not in values:
            continue
        lines.append(
            f'    {node.name}["{node.name}<br/>value={values[node.name]:.3f}'
            f'<br/>grad={gradients[node.name]:.3f}"]'
        )
    for node in experiment.nodes:
        for index, parent in enumerate(node.parents):
            if parent in values and node.name in values:
                local = local_derivatives(
                    node.operation, tuple(values[item] for item in node.parents)
                )[index]
                lines.append(f'    {parent} -->|"local={local:.3f}"| {node.name}')
    return "\n".join(lines)


def render_manual_backprop_markdown(experiment: ManualBackpropExperiment) -> str:
    """Render the complete Day 16 manual chain-rule audit as Markdown."""

    validate_experiment(experiment)
    metrics = experiment_metrics(experiment)
    lines = [
        "# Day 16 manual backpropagation report",
        "",
        (
            "The fixed graph is `y = ((a * b) + c) * d + a`. The manual pass "
            "uses explicit local derivatives and one accumulation step per edge."
        ),
        "",
        "## Deterministic result",
        "",
        f"- output: `{float(metrics['output']):.6f}`",
        f"- graph nodes: `{metrics['node_count']}`",
        f"- reverse edge steps: `{metrics['reverse_step_count']}`",
        f"- non-zero accumulation steps: `{metrics['accumulation_step_count']}`",
        f"- maximum comparison error: `{float(metrics['max_comparison_error']):.3e}`",
        f"- all comparisons passed: `{metrics['all_comparisons_pass']}`",
        "",
        "## Forward pass",
        "",
        "| Node | Operation | Parents | Value |",
        "|---|---|---|---:|",
    ]
    for step in experiment.trace.forward_steps:
        parents = ", ".join(step.parents) or "—"
        lines.append(
            f"| `{step.name}` | `{step.operation}` | {parents} | {step.value:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Reverse chain-rule trace",
            "",
            (
                "Each contribution is `upstream gradient × local derivative`. "
                "The before/after columns make shared-path accumulation visible."
            ),
            "",
            (
                "| Result → parent | Upstream | Local derivative | Contribution "
                "| Before | After |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for step in experiment.trace.reverse_steps:
        lines.append(
            f"| `{step.result}` → `{step.parent}` | "
            f"{step.upstream_gradient:.6f} | {step.local_derivative:.6f} | "
            f"{step.contribution:.6f} | {step.gradient_before:.6f} | "
            f"{step.gradient_after:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Independent gradient comparisons",
            "",
            "| Input | Manual trace | Day 15 autodiff | Closed form | Pass |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for item in experiment.comparisons:
        lines.append(
            f"| `{item.name}` | {item.manual:.6f} | {item.autodiff:.6f} | "
            f"{item.closed_form:.6f} | {'yes' if item.passed else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Computation graph",
            "",
            "```mermaid",
            render_trace_mermaid(experiment),
            "```",
            "",
            "## Scope",
            "",
            (
                "This report verifies add/multiply local derivatives, reverse "
                "ordering, and shared-path accumulation for one deterministic "
                "scalar graph. It does not claim lecture completion, a "
                "learner-written implementation, tensor support, or completion of "
                "the Week 3 MLP gate."
            ),
            "",
        ]
    )
    return "\n".join(lines)
