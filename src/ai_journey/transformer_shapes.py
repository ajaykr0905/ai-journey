"""Day 12: a deterministic, framework-free GPT tensor-shape model.

The module models dataflow rather than numerical values.  It makes every tensor
shape in a decoder-only transformer explicit, validates the graph, and renders a
reviewable Mermaid diagram and Markdown shape table.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

Shape = tuple[str, ...]


@dataclass(frozen=True)
class TransformerConfig:
    """Dimensions needed to describe a small decoder-only transformer."""

    batch_size: int = 2
    context_length: int = 8
    vocab_size: int = 32_000
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1_024
    n_layers: int = 4

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""

        return self.d_model // self.n_heads

    def dimensions(self) -> dict[str, int]:
        """Resolve the symbolic dimensions used throughout the shape graph."""

        return {
            "B": self.batch_size,
            "T": self.context_length,
            "V": self.vocab_size,
            "C": self.d_model,
            "H": self.n_heads,
            "D": self.head_dim,
            "F": self.d_ff,
            "L": self.n_layers,
            "3C": 3 * self.d_model,
            "1": 1,
        }


@dataclass(frozen=True)
class TensorSpec:
    """One named tensor in the GPT forward pass."""

    key: str
    label: str
    shape: Shape
    stage: str
    meaning: str


@dataclass(frozen=True)
class FlowEdge:
    """A directed operation connecting two tensors."""

    source: str
    target: str
    operation: str


@dataclass(frozen=True)
class GPTShapeFlow:
    """Ordered tensor specifications and their dataflow edges."""

    config: TransformerConfig
    tensors: tuple[TensorSpec, ...]
    edges: tuple[FlowEdge, ...]

    def tensor(self, key: str) -> TensorSpec:
        """Look up one tensor by stable key."""

        for tensor in self.tensors:
            if tensor.key == key:
                return tensor
        raise KeyError(key)


def format_shape(shape: Shape) -> str:
    """Format a symbolic shape as a compact mathematical tuple."""

    return "(" + ", ".join(shape) + ")"


def resolve_shape(shape: Shape, config: TransformerConfig) -> tuple[int, ...]:
    """Convert a symbolic shape into concrete dimensions for a configuration."""

    dimensions = config.dimensions()
    try:
        return tuple(dimensions[dimension] for dimension in shape)
    except KeyError as exc:
        raise ValueError(f"unknown shape dimension: {exc.args[0]}") from exc


def tensor_elements(tensor: TensorSpec, config: TransformerConfig) -> int:
    """Return the number of scalar elements in a tensor specification."""

    return reduce(mul, resolve_shape(tensor.shape, config), 1)


def build_gpt_shape_flow(config: TransformerConfig | None = None) -> GPTShapeFlow:
    """Build the full decoder-only GPT forward-pass shape graph."""

    cfg = config or TransformerConfig()
    tensors = (
        TensorSpec(
            "input_ids", "Token IDs", ("B", "T"), "Input", "Integer token indices"
        ),
        TensorSpec(
            "token_embeddings",
            "Token embeddings",
            ("B", "T", "C"),
            "Embedding",
            "Token lookup output",
        ),
        TensorSpec(
            "position_embeddings",
            "Position embeddings",
            ("T", "C"),
            "Embedding",
            "Learned position vectors",
        ),
        TensorSpec(
            "residual_input",
            "Residual stream",
            ("B", "T", "C"),
            "Embedding",
            "Token and position sum",
        ),
        TensorSpec(
            "attention_norm",
            "Pre-attention norm",
            ("B", "T", "C"),
            "Transformer block",
            "Normalized residual stream",
        ),
        TensorSpec(
            "qkv_projection",
            "QKV projection",
            ("B", "T", "3C"),
            "Attention",
            "Packed query, key, and value vectors",
        ),
        TensorSpec(
            "queries",
            "Queries",
            ("B", "H", "T", "D"),
            "Attention",
            "Per-head query vectors",
        ),
        TensorSpec(
            "keys",
            "Keys",
            ("B", "H", "T", "D"),
            "Attention",
            "Per-head key vectors",
        ),
        TensorSpec(
            "values",
            "Values",
            ("B", "H", "T", "D"),
            "Attention",
            "Per-head value vectors",
        ),
        TensorSpec(
            "attention_scores",
            "Scaled QK scores",
            ("B", "H", "T", "T"),
            "Attention",
            "Pairwise token compatibility",
        ),
        TensorSpec(
            "causal_mask",
            "Causal mask",
            ("1", "1", "T", "T"),
            "Attention",
            "Broadcast mask hiding future tokens",
        ),
        TensorSpec(
            "attention_weights",
            "Attention weights",
            ("B", "H", "T", "T"),
            "Attention",
            "Masked row-wise softmax",
        ),
        TensorSpec(
            "head_outputs",
            "Weighted values",
            ("B", "H", "T", "D"),
            "Attention",
            "Per-head contextualized vectors",
        ),
        TensorSpec(
            "concatenated_heads",
            "Concatenated heads",
            ("B", "T", "C"),
            "Attention",
            "Heads reassembled along channels",
        ),
        TensorSpec(
            "attention_output",
            "Attention output",
            ("B", "T", "C"),
            "Attention",
            "Output projection",
        ),
        TensorSpec(
            "post_attention",
            "Post-attention residual",
            ("B", "T", "C"),
            "Transformer block",
            "Residual plus attention output",
        ),
        TensorSpec(
            "mlp_norm",
            "Pre-MLP norm",
            ("B", "T", "C"),
            "Transformer block",
            "Normalized attention residual",
        ),
        TensorSpec(
            "mlp_hidden",
            "MLP expansion",
            ("B", "T", "F"),
            "MLP",
            "Expanded hidden representation",
        ),
        TensorSpec(
            "mlp_activation",
            "MLP activation",
            ("B", "T", "F"),
            "MLP",
            "Elementwise nonlinearity",
        ),
        TensorSpec(
            "mlp_output",
            "MLP projection",
            ("B", "T", "C"),
            "MLP",
            "Projection back to residual width",
        ),
        TensorSpec(
            "post_mlp",
            "Post-MLP residual",
            ("B", "T", "C"),
            "Transformer block",
            "Residual plus MLP output",
        ),
        TensorSpec(
            "final_norm",
            "Final norm",
            ("B", "T", "C"),
            "Output",
            "Normalized final residual stream",
        ),
        TensorSpec(
            "logits",
            "Vocabulary logits",
            ("B", "T", "V"),
            "Output",
            "Unnormalized next-token scores",
        ),
        TensorSpec(
            "token_probabilities",
            "Token probabilities",
            ("B", "T", "V"),
            "Output",
            "Softmax distribution over vocabulary",
        ),
    )
    edges = (
        FlowEdge("input_ids", "token_embeddings", "embedding lookup"),
        FlowEdge("token_embeddings", "residual_input", "add"),
        FlowEdge("position_embeddings", "residual_input", "add"),
        FlowEdge("residual_input", "attention_norm", "layer norm"),
        FlowEdge("attention_norm", "qkv_projection", "linear C to 3C"),
        FlowEdge("qkv_projection", "queries", "split and reshape"),
        FlowEdge("qkv_projection", "keys", "split and reshape"),
        FlowEdge("qkv_projection", "values", "split and reshape"),
        FlowEdge("queries", "attention_scores", "Q times K transpose / sqrt(D)"),
        FlowEdge("keys", "attention_scores", "Q times K transpose / sqrt(D)"),
        FlowEdge("attention_scores", "attention_weights", "mask then softmax"),
        FlowEdge("causal_mask", "attention_weights", "broadcast add"),
        FlowEdge("attention_weights", "head_outputs", "weights times V"),
        FlowEdge("values", "head_outputs", "weights times V"),
        FlowEdge("head_outputs", "concatenated_heads", "transpose and concatenate"),
        FlowEdge("concatenated_heads", "attention_output", "linear C to C"),
        FlowEdge("residual_input", "post_attention", "residual add"),
        FlowEdge("attention_output", "post_attention", "residual add"),
        FlowEdge("post_attention", "mlp_norm", "layer norm"),
        FlowEdge("mlp_norm", "mlp_hidden", "linear C to F"),
        FlowEdge("mlp_hidden", "mlp_activation", "GELU"),
        FlowEdge("mlp_activation", "mlp_output", "linear F to C"),
        FlowEdge("post_attention", "post_mlp", "residual add"),
        FlowEdge("mlp_output", "post_mlp", "residual add"),
        FlowEdge("post_mlp", "final_norm", "repeat block L times, then norm"),
        FlowEdge("final_norm", "logits", "unembedding C to V"),
        FlowEdge("logits", "token_probabilities", "softmax over V"),
    )
    flow = GPTShapeFlow(cfg, tensors, edges)
    validate_gpt_shape_flow(flow)
    return flow


def validate_gpt_shape_flow(flow: GPTShapeFlow) -> None:
    """Reject missing nodes, cycles, and shape-breaking transformer operations."""

    keys = [tensor.key for tensor in flow.tensors]
    if len(keys) != len(set(keys)):
        raise ValueError("tensor keys must be unique")

    known = set(keys)
    for edge in flow.edges:
        if edge.source not in known or edge.target not in known:
            raise ValueError(f"edge references unknown tensor: {edge}")

    expected_shapes: dict[str, Shape] = {
        "input_ids": ("B", "T"),
        "token_embeddings": ("B", "T", "C"),
        "position_embeddings": ("T", "C"),
        "residual_input": ("B", "T", "C"),
        "attention_norm": ("B", "T", "C"),
        "qkv_projection": ("B", "T", "3C"),
        "queries": ("B", "H", "T", "D"),
        "keys": ("B", "H", "T", "D"),
        "values": ("B", "H", "T", "D"),
        "attention_scores": ("B", "H", "T", "T"),
        "causal_mask": ("1", "1", "T", "T"),
        "attention_weights": ("B", "H", "T", "T"),
        "head_outputs": ("B", "H", "T", "D"),
        "concatenated_heads": ("B", "T", "C"),
        "attention_output": ("B", "T", "C"),
        "post_attention": ("B", "T", "C"),
        "mlp_norm": ("B", "T", "C"),
        "mlp_hidden": ("B", "T", "F"),
        "mlp_activation": ("B", "T", "F"),
        "mlp_output": ("B", "T", "C"),
        "post_mlp": ("B", "T", "C"),
        "final_norm": ("B", "T", "C"),
        "logits": ("B", "T", "V"),
        "token_probabilities": ("B", "T", "V"),
    }
    for key, expected in expected_shapes.items():
        if flow.tensor(key).shape != expected:
            raise ValueError(f"{key} must have shape {format_shape(expected)}")

    if flow.config.n_heads * flow.config.head_dim != flow.config.d_model:
        raise ValueError("attention heads must concatenate to d_model")

    adjacency = {key: [] for key in keys}
    indegree = {key: 0 for key in keys}
    for edge in flow.edges:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1
    ready = [key for key, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for target in adjacency[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(keys):
        raise ValueError("shape flow must be acyclic")


def parameter_count_breakdown(
    config: TransformerConfig, *, tie_embeddings: bool = True
) -> dict[str, int]:
    """Count learned scalar parameters by GPT component.

    Linear layers include biases.  Layer normalization contributes one scale and
    one bias per channel.  Tied unembedding reuses the token-embedding matrix.
    """

    c, f, v, t, layers = (
        config.d_model,
        config.d_ff,
        config.vocab_size,
        config.context_length,
        config.n_layers,
    )
    per_block = (
        2 * c
        + c * (3 * c)
        + 3 * c
        + c * c
        + c
        + 2 * c
        + c * f
        + f
        + f * c
        + c
    )
    breakdown = {
        "token_embeddings": v * c,
        "position_embeddings": t * c,
        "transformer_blocks": layers * per_block,
        "final_layer_norm": 2 * c,
        "unembedding": 0 if tie_embeddings else c * v,
    }
    breakdown["total"] = sum(breakdown.values())
    return breakdown


def render_shape_table(flow: GPTShapeFlow) -> str:
    """Render every tensor as a Markdown table."""

    rows = [
        "| Stage | Tensor | Symbolic shape | Concrete shape | Meaning |",
        "|---|---|---|---|---|",
    ]
    for tensor in flow.tensors:
        concrete = str(resolve_shape(tensor.shape, flow.config))
        rows.append(
            f"| {tensor.stage} | {tensor.label} | `{format_shape(tensor.shape)}` "
            f"| `{concrete}` | {tensor.meaning} |"
        )
    return "\n".join(rows)


def render_mermaid(flow: GPTShapeFlow) -> str:
    """Render the validated shape graph as a Mermaid flowchart."""

    lines = ["flowchart TD"]
    for tensor in flow.tensors:
        label = f"{tensor.label}<br/>{format_shape(tensor.shape)}"
        lines.append(f'    {tensor.key}["{label}"]')
    for edge in flow.edges:
        lines.append(
            f'    {edge.source} -->|"{edge.operation}"| {edge.target}'
        )
    return "\n".join(lines)


def render_reference_markdown(flow: GPTShapeFlow) -> str:
    """Render a self-contained reference diagram and shape ledger."""

    counts = parameter_count_breakdown(flow.config)
    return "\n".join(
        [
            "# Decoder-only GPT tensor-shape reference",
            "",
            "The graph is generated from a validated symbolic dataflow model.",
            "",
            "```mermaid",
            render_mermaid(flow),
            "```",
            "",
            "## Shape ledger",
            "",
            render_shape_table(flow),
            "",
            "## Configuration",
            "",
            "```text",
            *[f"{name}={value}" for name, value in flow.config.__dict__.items()],
            f"head_dim={flow.config.head_dim}",
            f"parameter_count={counts['total']}",
            "```",
            "",
        ]
    )
