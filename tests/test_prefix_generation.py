from itertools import product
from contextlib import contextmanager
import weakref

import pytest
import torch

from ai_journey.ddp_generation import SamplingConfig, generate_tokens
from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gqa_config import GQAConfig
from ai_journey.moe_config import MoEConfig
from ai_journey.prefix_cache import PrefixKVStore
from ai_journey.prefix_generation import prefix_prefill
from ai_journey.training_models import build_training_model


def configurations():
    base = GPT2Config(
        8, 6, 8, 2, 2, embedding_dropout=0, attention_dropout=0, residual_dropout=0
    )
    result = [pytest.param(base, id="baseline")]
    for norm, mlp, position in product(
        ("layernorm", "rmsnorm"), ("gelu", "swiglu"), ("learned", "rope")
    ):
        result.append(
            pytest.param(
                DecoderVariantConfig(base, norm, mlp, position),
                id=f"{norm}-{mlp}-{position}",
            )
        )
    grouped_base = GPT2Config(
        8, 6, 8, 4, 2, embedding_dropout=0, attention_dropout=0, residual_dropout=0
    )
    for heads, sparse in product((1, 2, 4), (False, True)):
        decoder = (MoEConfig if sparse else DecoderVariantConfig)(
            grouped_base, position_kind="rope"
        )
        result.append(
            pytest.param(GQAConfig(decoder, heads), id=f"gqa-{heads}-moe-{sparse}")
        )
    return result


@pytest.fixture(params=configurations())
def model(request):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        return build_training_model(request.param).double().eval()


def test_actual_branch_and_exact_prefix_logits_match_full_forward(model):
    store = PrefixKVStore(max_bytes=64 * 1024)
    for prompt, reused in [
        ((1, 2, 3, 4), 0),
        ((1, 2, 3, 5), 3),
        ((1, 2, 3, 5), 4),
        ((1, 2), 1),
    ]:
        receipt = {}
        with prefix_prefill(model, prompt, store=store, observation=receipt) as (
            logits,
            cache,
        ):
            torch.testing.assert_close(
                logits[:, -1, :],
                model(torch.tensor([prompt]))[:, -1, :],
                atol=1e-10,
                rtol=1e-10,
            )
            assert cache.length == len(prompt)
            assert receipt["reused_prompt_tokens"] == reused
            assert receipt["forwarded_prefill_tokens"] == len(prompt) - reused
        assert store.snapshot()["active_leases"] == 0


@pytest.mark.parametrize("temperature", [0, 0.8])
def test_prefix_generation_matches_uncached_sampling_through_context_resets(
    model, temperature
):
    config = SamplingConfig(max_new_tokens=8, temperature=temperature, seed=53)
    store = PrefixKVStore(max_bytes=64 * 1024)
    prompt = (1, 2, 3, 4)
    expected = generate_tokens(model, prompt, config)
    for _ in range(2):
        receipt = {}
        actual = generate_tokens(
            model,
            prompt,
            config,
            use_cache=True,
            prefix_store=store,
            prefix_observation=receipt,
        )
        assert actual == expected
        assert receipt["forwarded_prefill_tokens"] == (4 if _ == 0 else 0)
        assert store.snapshot()["active_leases"] == 0


def test_overlong_prompts_bind_the_actual_reset_context_window(model):
    store = PrefixKVStore(max_bytes=64 * 1024)
    config = SamplingConfig(max_new_tokens=2, temperature=0)
    first = (7, 7, 1, 2, 3, 4, 5, 6)
    second = (0, 0, 1, 2, 3, 4, 5, 6)
    generate_tokens(model, first, config, use_cache=True, prefix_store=store)
    receipt = {}
    actual = generate_tokens(
        model,
        second,
        config,
        use_cache=True,
        prefix_store=store,
        prefix_observation=receipt,
    )
    assert actual == generate_tokens(model, second, config)
    assert receipt["effective_prompt_tokens"] == 6
    assert receipt["reused_prompt_tokens"] == 6


def test_cancelled_before_prefill_does_not_touch_pool(model):
    store = PrefixKVStore(max_bytes=64 * 1024)
    receipt = {}
    result = generate_tokens(
        model,
        (1, 2),
        SamplingConfig(max_new_tokens=2),
        use_cache=True,
        prefix_store=store,
        prefix_observation=receipt,
        cancelled=lambda: True,
    )
    assert result["tokens"] == []
    assert receipt["outcome"] == "unused"
    assert store.snapshot()["misses"] == store.snapshot()["publications"] == 0


def test_lease_and_mixed_modes_restore_after_callback_failure(model):
    store = PrefixKVStore(max_bytes=64 * 1024)
    config = SamplingConfig(max_new_tokens=3, temperature=0)
    generate_tokens(model, (1, 2), config, use_cache=True, prefix_store=store)
    model.train()
    model.h[0].eval()
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()

    def fail(index, token):
        assert store.snapshot()["active_leases"] == 1
        raise RuntimeError("controlled client callback failure")

    with pytest.raises(RuntimeError, match="controlled"):
        generate_tokens(
            model, (1, 2), config, use_cache=True, prefix_store=store, on_token=fail
        )
    assert store.snapshot()["active_leases"] == store.snapshot()["leased_bytes"] == 0
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)


def test_small_budget_bypasses_cache_without_changing_real_generation(model):
    store = PrefixKVStore(max_bytes=1)
    config = SamplingConfig(max_new_tokens=2, temperature=0)
    assert generate_tokens(
        model, (1, 2), config, use_cache=True, prefix_store=store
    ) == generate_tokens(model, (1, 2), config)
    assert store.snapshot()["entries"] == 0


@pytest.mark.parametrize("failure", [False, True])
def test_owned_generation_releases_borrowed_tensors_before_lease_accounting(
    model, monkeypatch, failure
):
    store = PrefixKVStore(max_bytes=64 * 1024)
    config = SamplingConfig(max_new_tokens=1, temperature=0)
    generate_tokens(model, (1, 2), config, use_cache=True, prefix_store=store)
    acquire = store.acquire
    inspected = []

    @contextmanager
    def checked(*args, **kwargs):
        try:
            with acquire(*args, **kwargs) as match:
                references = [
                    weakref.ref(tensor)
                    for tensor in (
                        *match.cache.keys,
                        *match.cache.values,
                        match.last_logits,
                    )
                ]
                try:
                    yield match
                finally:
                    match = None
        finally:
            inspected.append(True)
            assert all(reference() is None for reference in references)

    monkeypatch.setattr(store, "acquire", checked)
    if failure:

        def fail(index, token):
            raise RuntimeError("controlled delivery failure")

        with pytest.raises(RuntimeError, match="delivery"):
            generate_tokens(
                model, (1, 2), config, use_cache=True, prefix_store=store, on_token=fail
            )
    else:
        generate_tokens(model, (1, 2), config, use_cache=True, prefix_store=store)
    assert inspected == [True]
