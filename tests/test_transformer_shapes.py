from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.transformer_shapes import (
    FlowEdge,
    GPTShapeFlow,
    TransformerConfig,
    build_gpt_shape_flow,
    parameter_count_breakdown,
    render_mermaid,
    render_reference_markdown,
    render_shape_table,
    resolve_shape,
    tensor_elements,
    validate_gpt_shape_flow,
)


class TransformerShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TransformerConfig(
            batch_size=2,
            context_length=8,
            vocab_size=100,
            d_model=16,
            n_heads=4,
            d_ff=64,
            n_layers=2,
        )
        self.flow = build_gpt_shape_flow(self.config)

    def test_config_derives_head_dimension(self) -> None:
        self.assertEqual(self.config.head_dim, 4)
        self.assertEqual(self.config.dimensions()["3C"], 48)

    def test_config_rejects_non_positive_and_non_divisible_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            TransformerConfig(batch_size=0)
        with self.assertRaises(ValueError):
            TransformerConfig(d_model=10, n_heads=4)
        with self.assertRaises(ValueError):
            TransformerConfig(n_layers=True)  # type: ignore[arg-type]

    def test_core_attention_shapes_are_explicit(self) -> None:
        self.assertEqual(
            resolve_shape(self.flow.tensor("queries").shape, self.config),
            (2, 4, 8, 4),
        )
        self.assertEqual(
            resolve_shape(self.flow.tensor("attention_scores").shape, self.config),
            (2, 4, 8, 8),
        )
        self.assertEqual(
            resolve_shape(self.flow.tensor("causal_mask").shape, self.config),
            (1, 1, 8, 8),
        )

    def test_residual_stream_width_is_preserved(self) -> None:
        expected = (2, 8, 16)
        for key in ("residual_input", "post_attention", "post_mlp", "final_norm"):
            self.assertEqual(resolve_shape(self.flow.tensor(key).shape, self.config), expected)

    def test_logits_and_probabilities_cover_vocabulary(self) -> None:
        expected = (2, 8, 100)
        self.assertEqual(resolve_shape(self.flow.tensor("logits").shape, self.config), expected)
        self.assertEqual(
            resolve_shape(self.flow.tensor("token_probabilities").shape, self.config),
            expected,
        )

    def test_tensor_element_count_uses_concrete_dimensions(self) -> None:
        self.assertEqual(tensor_elements(self.flow.tensor("attention_scores"), self.config), 512)

    def test_unknown_symbol_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown shape dimension"):
            resolve_shape(("B", "UNKNOWN"), self.config)

    def test_every_edge_references_a_known_tensor(self) -> None:
        known = {tensor.key for tensor in self.flow.tensors}
        for edge in self.flow.edges:
            self.assertIn(edge.source, known)
            self.assertIn(edge.target, known)

    def test_validator_rejects_unknown_edge_endpoint(self) -> None:
        broken = replace(
            self.flow,
            edges=self.flow.edges + (FlowEdge("logits", "missing", "bad edge"),),
        )
        with self.assertRaisesRegex(ValueError, "unknown tensor"):
            validate_gpt_shape_flow(broken)

    def test_validator_rejects_a_cycle(self) -> None:
        broken = GPTShapeFlow(
            self.flow.config,
            self.flow.tensors,
            self.flow.edges + (FlowEdge("token_probabilities", "input_ids", "cycle"),),
        )
        with self.assertRaisesRegex(ValueError, "acyclic"):
            validate_gpt_shape_flow(broken)

    def test_parameter_counts_handle_tied_and_untied_unembedding(self) -> None:
        tied = parameter_count_breakdown(self.config, tie_embeddings=True)
        untied = parameter_count_breakdown(self.config, tie_embeddings=False)
        self.assertEqual(tied["unembedding"], 0)
        self.assertEqual(untied["total"] - tied["total"], 16 * 100)
        self.assertEqual(
            tied["total"],
            sum(value for key, value in tied.items() if key != "total"),
        )

    def test_markdown_table_labels_symbolic_and_concrete_shapes(self) -> None:
        table = render_shape_table(self.flow)
        self.assertIn("| Queries | `(B, H, T, D)` | `(2, 4, 8, 4)` |", table)
        self.assertEqual(table.count("| Attention |"), 10)

    def test_mermaid_contains_all_nodes_and_key_operations(self) -> None:
        diagram = render_mermaid(self.flow)
        for tensor in self.flow.tensors:
            self.assertIn(f"    {tensor.key}[", diagram)
        self.assertIn('queries -->|"Q times K transpose / sqrt(D)"| attention_scores', diagram)
        self.assertIn('logits -->|"softmax over V"| token_probabilities', diagram)

    def test_reference_markdown_is_deterministic(self) -> None:
        first = render_reference_markdown(self.flow)
        second = render_reference_markdown(build_gpt_shape_flow(self.config))
        self.assertEqual(first, second)
        self.assertIn("parameter_count=", first)


if __name__ == "__main__":
    unittest.main()
