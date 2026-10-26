from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import pytest
import torch

from ai_journey.ddp_cache import cached_forward
from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gqa_config import GQAConfig
from ai_journey.prefix_cache import PrefixKVStore, prefix_identity
from ai_journey.training_models import build_training_model


@pytest.fixture(params=["baseline", "rotary", "grouped"])
def prefill(request):
    base = GPT2Config(
        8, 8, 8, 2, 1, embedding_dropout=0, attention_dropout=0, residual_dropout=0
    )
    config = base
    if request.param != "baseline":
        config = DecoderVariantConfig(base, position_kind="rope")
    if request.param == "grouped":
        config = GQAConfig(config, 1)
    with torch.random.fork_rng(devices=[]):
        model = build_training_model(config).double().eval()
    tokens = (1, 2, 3, 4)
    logits, cache = cached_forward(model, torch.tensor([tokens]))
    return model, tokens, logits[:, -1, :], cache


def test_resident_and_leases_are_compact_independent_and_fully_accounted(prefill):
    model, tokens, logits, cache = prefill
    identity = prefix_identity(model)
    size = identity.bytes_for(len(tokens))
    store = PrefixKVStore(max_bytes=size * 3)
    expected = cache.clone()
    assert store.publish(identity, tokens, cache, logits)
    assert store.snapshot()["resident_bytes"] == size
    with torch.inference_mode():
        cache.keys[0].zero_()
        logits.zero_()
    with store.acquire(identity, tokens) as match:
        assert match.prefix_length == 4
        assert match.last_logits is not None
        assert torch.equal(match.cache.keys[0], expected.keys[0])
        assert store.snapshot()["leased_bytes"] == size
        match.cache.keys[0].zero_()
        match.last_logits.zero_()
    with store.acquire(identity, tokens) as match:
        assert torch.equal(match.cache.keys[0], expected.keys[0])
        assert match.last_logits.abs().sum() > 0
    assert store.snapshot()["leased_bytes"] == 0
    assert store.snapshot()["reserved_bytes"] == 0
    assert store.snapshot()["active_leases"] == 0


def test_longest_branch_prefix_and_interior_exact_request_leave_required_logits_work(
    prefill,
):
    model, tokens, logits, cache = prefill
    identity = prefix_identity(model)
    store = PrefixKVStore(max_bytes=4096)
    store.publish(identity, tokens, cache, logits)
    for prompt, expected in [((1, 2, 3, 5), 3), ((1, 2), 1), ((1, 2, 3, 4, 5), 4)]:
        with store.acquire(identity, prompt) as match:
            assert match.prefix_length == match.cache.length == expected
            assert match.last_logits is None
    with store.acquire(identity, (5, 6)) as match:
        assert match is None


def test_model_architecture_dtype_and_trusted_namespace_are_not_interchangeable(
    prefill,
):
    model, tokens, logits, cache = prefill
    identity = prefix_identity(model, namespace="one")
    store = PrefixKVStore(max_bytes=4096)
    store.publish(identity, tokens, cache, logits)
    for other in (
        replace(identity, namespace="two"),
        replace(identity, model_sha256="a" * 64),
        replace(identity, architecture_sha256="a" * 64),
        replace(identity, dtype=torch.float32),
    ):
        with store.acquire(other, tokens) as match:
            assert match is None


def test_baseline_epsilon_is_bound_even_with_identical_weight_fingerprint(prefill):
    model, _, _, _ = prefill
    config = model.config
    if type(config) is GPT2Config:
        changed = replace(config, layer_norm_epsilon=0.1)
    elif isinstance(config, GQAConfig):
        changed = replace(
            config,
            decoder=replace(
                config.decoder, gpt2=replace(config.gpt2, layer_norm_epsilon=0.1)
            ),
        )
    else:
        changed = replace(config, gpt2=replace(config.gpt2, layer_norm_epsilon=0.1))
    other = build_training_model(changed).double().eval()
    other.load_state_dict(model.state_dict())
    first, second = prefix_identity(model), prefix_identity(other)
    if type(config) is GPT2Config:
        assert first.model_sha256 == second.model_sha256
    assert first.architecture_sha256 != second.architecture_sha256


def test_exhausted_memory_bypasses_without_clone_or_corrupting_active_lease(
    prefill, monkeypatch
):
    model, tokens, logits, cache = prefill
    identity = prefix_identity(model)
    size = identity.bytes_for(len(tokens))
    store = PrefixKVStore(max_bytes=2 * size)
    store.publish(identity, tokens, cache, logits)
    with store.acquire(identity, tokens) as match:
        frozen = match.cache.clone()
        store.clear()
        assert store.snapshot()["resident_bytes"] == 0
        assert store.snapshot()["leased_bytes"] == size
        assert store.publish(identity, tokens, cache, logits)

        def unexpected(*args):
            raise AssertionError("capacity bypass must precede tensor clone")

        monkeypatch.setattr("ai_journey.prefix_cache._copy_value", unexpected)
        with store.acquire(identity, tokens) as denied:
            assert denied is None
        assert torch.equal(match.cache.keys[0], frozen.keys[0])
        assert (
            store.snapshot()["resident_bytes"] + store.snapshot()["leased_bytes"]
            <= 2 * size
        )
    assert store.snapshot()["active_leases"] == 0


def test_clone_failure_releases_all_reservations(prefill, monkeypatch):
    model, tokens, logits, cache = prefill
    identity = prefix_identity(model)
    store = PrefixKVStore(max_bytes=4096)

    def fail(*args):
        raise RuntimeError("controlled allocation failure")

    with monkeypatch.context() as patch:
        patch.setattr("ai_journey.prefix_cache._copy_value", fail)
        with pytest.raises(RuntimeError, match="controlled"):
            store.publish(identity, tokens, cache, logits)
    assert store.snapshot()["reserved_bytes"] == 0
    assert store.snapshot()["entries"] == 0
    assert store.publish(identity, tokens, cache, logits)
    with monkeypatch.context() as patch:
        patch.setattr("ai_journey.prefix_cache._copy_value", fail)
        with pytest.raises(RuntimeError, match="controlled"):
            with store.acquire(identity, tokens):
                pass
    assert store.snapshot()["reserved_bytes"] == 0
    assert store.snapshot()["leased_bytes"] == 0


def test_lru_entry_bound_and_lease_exception_cleanup(prefill):
    model, tokens, logits, cache = prefill
    identity = prefix_identity(model)
    store = PrefixKVStore(max_bytes=4096, max_entries=1)
    store.publish(identity, tokens, cache, logits)
    with pytest.raises(RuntimeError):
        with store.acquire(identity, tokens):
            raise RuntimeError("caller stopped")
    other = replace(identity, namespace="other")
    store.publish(other, tokens, cache, logits)
    assert store.snapshot()["entries"] == 1
    assert store.snapshot()["evictions"] == 1
    assert store.snapshot()["active_leases"] == 0
    with store.acquire(identity, tokens) as match:
        assert match is None


def test_concurrent_leases_never_exceed_the_live_byte_budget(prefill):
    model, tokens, logits, cache = prefill
    identity = prefix_identity(model)
    size = identity.bytes_for(len(tokens))
    store = PrefixKVStore(max_bytes=3 * size)
    store.publish(identity, tokens, cache, logits)
    barrier = threading.Barrier(2)

    def borrow():
        with store.acquire(identity, tokens) as match:
            assert match is not None
            barrier.wait(timeout=3)
            state = store.snapshot()
            assert (
                state["resident_bytes"]
                + state["leased_bytes"]
                + state["reserved_bytes"]
                <= 3 * size
            )

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(lambda _: borrow(), range(2)))
    assert store.snapshot()["active_leases"] == 0


@pytest.mark.parametrize("value", [True, 0, -1, 64 * 1024 * 1024 + 1])
def test_invalid_budget_rejects_before_creating_store(value):
    with pytest.raises(ValueError, match="budget"):
        PrefixKVStore(max_bytes=value)
