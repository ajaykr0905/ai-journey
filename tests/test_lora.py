import unittest
import copy

import torch
from torch import nn

from ai_journey.lora import (
    LoRAConfig,
    LoRAError,
    LoRALinear,
    inject_adapters,
    adapter_parameters,
    parameter_summary,
    frozen_fingerprint,
    build_adapter_optimizer,
    adapter_state,
    load_adapter_state,
    merge_adapters,
    audit_adapter_gradients,
)
from ai_journey.transformer_lab import DecoderLanguageModel, TransformerConfig


class LoRAConfigTests(unittest.TestCase):
    def test_capacity_and_scaling_are_predeclared(self):
        self.assertEqual(LoRAConfig(rank=4, alpha=8).scaling, 2)

    def test_invalid_controls_cannot_silently_change_the_experiment(self):
        for name, values in {
            "rank": (True, 0, 257, 1.5),
            "alpha": (True, 0, -1, float("nan"), float("inf"), 1025),
            "dropout": (True, -0.1, 1, float("nan")),
            "seed": (True, -1, 2**32, 1.5),
        }.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(LoRAError):
                    LoRAConfig(**{name: value})


class LoRALinearTests(unittest.TestCase):
    def test_zero_update_preserves_base_output_and_constructor_rng(self):
        base = nn.Linear(4, 6).double().eval()
        inputs = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        rng = torch.get_rng_state().clone()
        layer = LoRALinear(base, LoRAConfig())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(base(inputs), layer(inputs)))
        self.assertTrue(all(p.requires_grad for p in base.parameters()))
        self.assertTrue(all(not p.requires_grad for p in layer.base.parameters()))
        self.assertFalse(layer.training)
        self.assertNotEqual(base.weight.data_ptr(), layer.base.weight.data_ptr())

    def test_low_rank_update_matches_dense_matrix_and_input_gradients(self):
        base = nn.Linear(4, 6).double()
        layer = LoRALinear(base, LoRAConfig(rank=2, alpha=3))
        with torch.no_grad():
            layer.lora_b.copy_(torch.arange(12).reshape(6, 2) / 10)
        x = torch.arange(8, dtype=torch.float64).reshape(2, 4).requires_grad_()
        expected = nn.functional.linear(
            x, base.weight + layer.delta_weight(), base.bias
        )
        actual = layer(x)
        torch.testing.assert_close(actual, expected)
        expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
        actual.sum().backward()
        torch.testing.assert_close(x.grad, expected_gradient)
        self.assertTrue(all(p.grad is None for p in layer.base.parameters()))
        self.assertGreater(float(layer.lora_a.grad.abs().sum()), 0)
        self.assertGreater(float(layer.lora_b.grad.abs().sum()), 0)

    def test_rank_and_finite_base_are_checked_before_allocation(self):
        with self.assertRaises(LoRAError):
            LoRALinear(nn.Linear(1, 3), LoRAConfig(rank=2))
        base = nn.Linear(4, 4)
        with torch.no_grad():
            base.weight[0, 0] = float("nan")
        with self.assertRaises(LoRAError):
            LoRALinear(base, LoRAConfig())

    def test_dropout_obeys_train_eval_modes(self):
        layer = LoRALinear(nn.Linear(4, 4), LoRAConfig(dropout=0.5))
        with torch.no_grad():
            layer.lora_b.fill_(1)
        inputs = torch.ones(20, 4)
        layer.eval()
        self.assertTrue(torch.equal(layer(inputs), layer(inputs)))
        expected = nn.functional.linear(
            inputs, layer.weight + layer.delta_weight(), layer.bias
        )
        torch.testing.assert_close(layer(inputs), expected)
        layer.train()
        self.assertFalse(torch.equal(layer(inputs), layer(inputs)))


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=8, block_size=4, embedding_dim=8, head_count=2, layer_count=1
            )
        )
        self.targets = (
            "blocks.0.attention.query_key_value",
            "blocks.0.attention.projection",
        )

    def test_decoder_injection_preserves_logits_and_original_trainability(self):
        tokens = torch.tensor([[1, 2, 3, 4]])
        logits = self.model(tokens)[0].detach().clone()
        adapted = inject_adapters(self.model, self.targets, LoRAConfig())
        self.assertTrue(torch.equal(logits, adapted(tokens)[0]))
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        for name, parameter in adapted.named_parameters():
            self.assertEqual(
                parameter.requires_grad, name.endswith(("lora_a", "lora_b"))
            )
        loss = adapted(tokens, tokens.roll(-1, 1))[1]
        loss.backward()
        self.assertTrue(
            all(p.grad is None for n, p in adapted.named_parameters() if ".base." in n)
        )

    def test_target_order_has_no_effect_on_local_initialization(self):
        first = inject_adapters(self.model, self.targets, LoRAConfig())
        second = inject_adapters(self.model, self.targets[::-1], LoRAConfig())
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]))

    def test_bad_targets_do_not_partially_mutate_original(self):
        for targets in ((), ("missing",), ("blocks",), (self.targets[0],) * 2):
            with self.subTest(targets=targets), self.assertRaises(LoRAError):
                inject_adapters(self.model, targets, LoRAConfig())
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        self.assertFalse(any(isinstance(m, LoRALinear) for m in self.model.modules()))

    def test_nested_and_aliased_adapters_are_rejected(self):
        adapted = inject_adapters(self.model, self.targets, LoRAConfig())
        with self.assertRaises(LoRAError):
            inject_adapters(adapted, self.targets, LoRAConfig())
        shared = nn.Linear(4, 4)
        alias = nn.ModuleDict({"left": shared, "right": shared})
        with self.assertRaises(LoRAError):
            inject_adapters(alias, ("left",), LoRAConfig())


class ParameterAccountingTests(unittest.TestCase):
    def test_trainable_census_is_exact_for_a_frozen_projection(self):
        model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
        adapted = inject_adapters(model, ("0",), LoRAConfig(rank=2))
        summary = parameter_summary(adapted)
        self.assertEqual(summary["trainable"], 2 * (4 + 6))
        self.assertEqual(summary["frozen"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(summary["total"], summary["trainable"] + summary["frozen"])
        self.assertEqual(sum(p.numel() for p in adapter_parameters(adapted)), 20)
        self.assertEqual(adapter_parameters(model), ())

    def test_frozen_identity_ignores_updates_but_detects_base_corruption(self):
        adapted = inject_adapters(nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig())
        original = frozen_fingerprint(adapted)
        with torch.no_grad():
            adapted[0].lora_b.fill_(1)
        self.assertEqual(frozen_fingerprint(adapted), original)
        with torch.no_grad():
            adapted[0].base.weight[0, 0] += 1
        self.assertNotEqual(frozen_fingerprint(adapted), original)

    def test_frozen_identity_includes_dtype_and_persistent_model_buffers(self):
        adapted = inject_adapters(nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig())
        original = frozen_fingerprint(adapted)
        self.assertNotEqual(frozen_fingerprint(adapted.double()), original)
        adapted.register_buffer("running_value", torch.tensor(1.0))
        before = frozen_fingerprint(adapted)
        adapted.running_value.fill_(2)
        self.assertNotEqual(frozen_fingerprint(adapted), before)


class AdapterOptimizerTests(unittest.TestCase):
    def test_optimizer_step_changes_factors_and_never_frozen_weights(self):
        model = inject_adapters(nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig())
        optimizer = build_adapter_optimizer(model)
        before = frozen_fingerprint(model)
        factor = model[0].lora_b.detach().clone()
        model(torch.ones(3, 4)).square().sum().backward()
        optimizer.step()
        self.assertEqual(before, frozen_fingerprint(model))
        self.assertFalse(torch.equal(factor, model[0].lora_b))
        self.assertEqual(len(optimizer.state), 2)
        self.assertEqual(
            {id(p) for group in optimizer.param_groups for p in group["params"]},
            {id(p) for p in adapter_parameters(model)},
        )

    def test_accidental_unfreeze_or_missing_adapter_factor_fails_closed(self):
        model = inject_adapters(nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig())
        model[0].base.weight.requires_grad_(True)
        with self.assertRaises(LoRAError):
            build_adapter_optimizer(model)
        model[0].base.weight.requires_grad_(False)
        model[0].lora_a.requires_grad_(False)
        with self.assertRaises(LoRAError):
            build_adapter_optimizer(model)

    def test_optimizer_controls_and_dense_models_are_rejected(self):
        model = inject_adapters(nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig())
        for controls in (
            {"learning_rate": True},
            {"learning_rate": 0},
            {"weight_decay": -1},
            {"weight_decay": float("nan")},
        ):
            with self.subTest(controls=controls), self.assertRaises(LoRAError):
                build_adapter_optimizer(model, **controls)
        with self.assertRaises(LoRAError):
            build_adapter_optimizer(nn.Linear(4, 4))


class AdapterStateTests(unittest.TestCase):
    def test_export_contains_no_base_weights_and_owns_its_tensors(self):
        model = inject_adapters(nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig())
        state = adapter_state(model)
        self.assertEqual(state["schema"], "lora-state-v1")
        self.assertEqual(state["base_fingerprint"], frozen_fingerprint(model))
        self.assertEqual(set(state["layers"]), {"0"})
        layer = state["layers"]["0"]
        self.assertEqual(set(layer), {"config", "a", "b"})
        self.assertEqual(layer["config"]["rank"], 2)
        layer["b"].fill_(5)
        self.assertEqual(float(model[0].lora_b.detach().abs().sum()), 0)

    def test_nonfinite_factors_and_models_without_adapters_cannot_be_saved(self):
        with self.assertRaises(LoRAError):
            adapter_state(nn.Linear(4, 4))
        model = inject_adapters(nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig())
        with torch.no_grad():
            model[0].lora_b[0, 0] = float("inf")
        with self.assertRaises(LoRAError):
            adapter_state(model)


class AdapterLoadTests(unittest.TestCase):
    def setUp(self):
        self.model = inject_adapters(
            nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)), ("0", "1"), LoRAConfig()
        )

    def test_load_reproduces_logits_without_aliasing_source_tensors(self):
        state = adapter_state(self.model)
        state["layers"]["0"]["b"].fill_(0.5)
        load_adapter_state(self.model, state)
        self.assertTrue(torch.equal(self.model[0].lora_b, state["layers"]["0"]["b"]))
        state["layers"]["0"]["b"].zero_()
        self.assertEqual(float(self.model[0].lora_b.detach()[0, 0]), 0.5)

    def test_late_invalid_tensor_never_partially_updates_earlier_layers(self):
        state = adapter_state(self.model)
        state["layers"]["0"]["b"].fill_(1)
        state["layers"]["1"]["b"][0, 0] = float("nan")
        with self.assertRaises(LoRAError):
            load_adapter_state(self.model, state)

        self.assertEqual(float(self.model[0].lora_b.detach().abs().sum()), 0)

    def test_base_identity_target_set_controls_dtype_and_shape_are_bound(self):
        original = adapter_state(self.model)
        for mutation in (
            lambda s: s.update(base_fingerprint="0" * 64),
            lambda s: s["layers"].pop("1"),
            lambda s: s["layers"]["0"]["config"].update(alpha=8),
            lambda s: s["layers"]["0"].update(a=s["layers"]["0"]["a"].double()),
            lambda s: s["layers"]["0"].update(a=torch.ones(1, 4)),
            lambda s: s.update(extra=True),
        ):
            state = copy.deepcopy(original)
            mutation(state)
            with self.assertRaises(LoRAError):
                load_adapter_state(self.model, state)

    def test_finite_factors_that_overflow_the_update_are_rejected(self):
        state = adapter_state(self.model)
        state["layers"]["0"]["a"].fill_(1e30)
        state["layers"]["0"]["b"].fill_(1e30)
        with self.assertRaises(LoRAError):
            load_adapter_state(self.model, state)


class AdapterMergeTests(unittest.TestCase):
    def test_merge_matches_decoder_logits_without_changing_training_model(self):
        base = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=8, block_size=4, embedding_dim=8, head_count=2, layer_count=1
            )
        ).double()
        adapted = inject_adapters(
            base,
            ("blocks.0.attention.query_key_value", "blocks.0.attention.projection"),
            LoRAConfig(),
        ).eval()
        with torch.no_grad():
            for module in adapted.modules():
                if isinstance(module, LoRALinear):
                    module.lora_b.fill_(0.1)
        before = {name: value.clone() for name, value in adapted.state_dict().items()}
        tokens = torch.tensor([[1, 2, 3, 4]])
        merged = merge_adapters(adapted)
        torch.testing.assert_close(
            merged(tokens)[0], adapted(tokens)[0], rtol=1e-10, atol=1e-12
        )
        self.assertFalse(any(isinstance(m, LoRALinear) for m in merged.modules()))
        self.assertEqual(set(merged.state_dict()), set(base.state_dict()))
        for name, value in adapted.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_root_layer_merge_and_dropout_contract(self):
        layer = LoRALinear(nn.Linear(4, 4), LoRAConfig(dropout=0.5))
        with self.assertRaises(LoRAError):
            merge_adapters(layer)
        layer.eval()
        self.assertIs(type(merge_adapters(layer)), nn.Linear)

    def test_nonfinite_merge_fails_without_modifying_base(self):
        layer = LoRALinear(nn.Linear(4, 4), LoRAConfig())
        before = layer.base.weight.detach().clone()
        with torch.no_grad():
            layer.lora_b.fill_(float("inf"))
        with self.assertRaises(LoRAError):
            merge_adapters(layer)
        self.assertTrue(torch.equal(before, layer.base.weight))


class ArchitectureIdentityTests(unittest.TestCase):
    def test_identical_tensors_with_different_attention_heads_cannot_share_adapters(
        self,
    ):
        common = dict(vocab_size=8, block_size=4, embedding_dim=8, layer_count=1)
        first = DecoderLanguageModel(TransformerConfig(**common, head_count=2))
        second = DecoderLanguageModel(TransformerConfig(**common, head_count=1))
        second.load_state_dict(first.state_dict())
        targets = ("blocks.0.attention.query_key_value",)
        source = inject_adapters(first, targets, LoRAConfig())
        destination = inject_adapters(second, targets, LoRAConfig())
        self.assertNotEqual(frozen_fingerprint(source), frozen_fingerprint(destination))
        with self.assertRaises(LoRAError):
            load_adapter_state(destination, adapter_state(source))

    def test_registered_activation_semantics_and_norm_epsilon_are_bound(self):
        first = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.LayerNorm(4))
        second = copy.deepcopy(first)
        second[1] = nn.ReLU()
        for model in (first, second):
            model.requires_grad_(False)
        self.assertNotEqual(frozen_fingerprint(first), frozen_fingerprint(second))
        second = copy.deepcopy(first)
        second[2].eps = 1e-2
        self.assertNotEqual(frozen_fingerprint(first), frozen_fingerprint(second))
        before = frozen_fingerprint(first)
        first.eval()
        self.assertEqual(frozen_fingerprint(first), before)


class FrozenBufferTests(unittest.TestCase):
    def test_scratch_batchnorm_decoder_is_rejected_before_installation(self):
        base = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=8,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
                normalization_mode="scratch_batch_norm",
            )
        )
        before = {name: value.clone() for name, value in base.state_dict().items()}
        with self.assertRaisesRegex(LoRAError, "stateless normalization"):
            inject_adapters(base, ("blocks.0.attention.query_key_value",), LoRAConfig())
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_torch_batchnorm_is_also_rejected_and_layernorm_is_supported(self):
        with self.assertRaisesRegex(LoRAError, "stateless normalization"):
            inject_adapters(
                nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)), ("0",), LoRAConfig()
            )
        model = inject_adapters(
            nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)), ("0",), LoRAConfig()
        )
        before = frozen_fingerprint(model)
        model(torch.ones(2, 4)).sum().backward()
        audit_adapter_gradients(model)
        self.assertEqual(frozen_fingerprint(model), before)


class DamagedFactorRecoveryTests(unittest.TestCase):
    def test_valid_state_can_replace_nonfinite_destination_factors(self):
        model = inject_adapters(nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig())
        state = adapter_state(model)
        x = torch.ones(2, 4)
        expected = model(x).detach().clone()
        with torch.no_grad():
            model[0].lora_a.fill_(float("nan"))
            model[0].lora_b.fill_(float("inf"))
        load_adapter_state(model, state)
        self.assertTrue(torch.equal(model(x), expected))

    def test_bad_incoming_state_still_does_not_partially_change_a_damaged_model(self):
        model = inject_adapters(nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig())
        state = adapter_state(model)
        with torch.no_grad():
            model[0].lora_a.fill_(float("nan"))
        state["layers"]["0"]["b"].fill_(float("inf"))
        with self.assertRaises(LoRAError):
            load_adapter_state(model, state)
        self.assertTrue(bool(torch.isnan(model[0].lora_a).all()))
        self.assertEqual(float(model[0].lora_b.detach().abs().sum()), 0)


class AdapterGradientAuditTests(unittest.TestCase):
    def setUp(self):
        self.model = inject_adapters(
            nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig()
        )

    def test_first_backward_has_zero_a_gradient_and_live_b_gradient(self):
        self.model(torch.ones(2, 4)).square().sum().backward()
        audit = audit_adapter_gradients(self.model)
        self.assertEqual(audit["factor_count"], 2)
        self.assertEqual(audit["gradient_norms"]["0.lora_a"], 0)
        self.assertGreater(audit["gradient_norms"]["0.lora_b"], 0)

    def test_missing_factor_gradients_cannot_masquerade_as_frozen_savings(self):
        with self.assertRaises(LoRAError):
            audit_adapter_gradients(self.model)

    def test_stale_base_gradient_and_nonfinite_factor_gradient_are_rejected(self):
        self.model(torch.ones(2, 4)).sum().backward()
        self.model[0].base.weight.grad = torch.zeros_like(self.model[0].base.weight)
        with self.assertRaises(LoRAError):
            audit_adapter_gradients(self.model)
        self.model[0].base.weight.grad = None
        self.model[0].lora_b.grad[0, 0] = float("nan")
        with self.assertRaises(LoRAError):
            audit_adapter_gradients(self.model)
