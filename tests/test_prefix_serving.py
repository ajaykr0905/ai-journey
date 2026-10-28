from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import argparse
import http.client
import json
import threading
import time

import pytest
import torch

from ai_journey.ddp_generation import SamplingConfig
from ai_journey.ddp_http import make_server
from ai_journey.ddp_operations import add_commands
from ai_journey.ddp_service import GenerationRequest, GenerationService, TokenBucket
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.http_measurements import measure_http_generation
from ai_journey.prefix_cache import PrefixKVStore
from ai_journey.prefix_metrics import PrefixMetrics


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        return GPT2LanguageModel(GPT2Config(8, 8, 8, 2, 1)).double().eval()


def request(identifier, prompt=(1, 2, 3, 4), maximum=3):
    return GenerationRequest(identifier, prompt, SamplingConfig(maximum, temperature=0))


@contextmanager
def serving(service):
    server = make_server(service)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server
    finally:
        service.drain(timeout_seconds=2)
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
        assert not thread.is_alive()


def mapping(identifier, prompt=(1, 2, 3, 4)):
    return {
        "request_id": identifier,
        "prompt": list(prompt),
        "sampling": {"max_new_tokens": 3, "temperature": 0},
    }


def test_fresh_http_ids_reuse_prefix_but_generate_and_charge_again(model):
    pool = PrefixKVStore(max_bytes=8192)
    bucket = TokenBucket(21, 1, clock=lambda: 0.0)
    service = GenerationService(model, prefix_store=pool, admission=bucket)
    with serving(service) as (endpoint, server):
        cold = measure_http_generation(endpoint, mapping("cold"), streaming=False)
        warm = measure_http_generation(endpoint, mapping("warm"))
        branch = measure_http_generation(endpoint, mapping("branch", (1, 2, 3, 5)))
        assert cold["outcome"] == warm["outcome"] == branch["outcome"] == "completed"
        assert cold["tokens"] == warm["tokens"]
        assert bucket.tokens == 0
        assert service.completed_requests == 3 and service.generated_tokens == 9
        assert service.metrics.snapshot()["generated_tokens"] == 9
        assert service.metrics.snapshot()["cache_hits"] == 0
        state = service.prefix_metrics.snapshot()
        assert state["attempts"] == {"hit": 2, "miss": 1, "bypass": 0}
        assert state["reused_prompt_tokens"] == 7
        assert state["forwarded_prefill_tokens"] == 5
        assert pool.snapshot()["leased_bytes"] == 0
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        metrics = response.read().decode()
        connection.close()
        assert response.status == 200
        assert 'ai_journey_prefix_attempts_total{outcome="hit"} 2' in metrics
        assert "ai_journey_serving_cache_hits_total 0" in metrics
        for private in ("cold", "warm", "branch", service.model_sha256):
            assert private not in metrics
        replay = service.generate(request("warm"))
        assert replay["tokens"] == warm["tokens"]
        assert service.metrics.snapshot()["cache_hits"] == 1
        assert service.metrics.snapshot()["generated_tokens"] == 9
        assert bucket.tokens == 0
        assert service.prefix_metrics.snapshot()["attempts"] == state["attempts"]
    assert pool.snapshot()["resident_bytes"] == 0


def test_default_off_and_server_owned_namespace_isolation(model):
    default = GenerationService(model)
    default.generate(request("off"))
    assert default.prefix_metrics is None
    assert "ai_journey_prefix" not in default.render_metrics()
    pool = PrefixKVStore(max_bytes=8192)
    one = GenerationService(model, prefix_store=pool, prefix_namespace="operator_one")
    two = GenerationService(model, prefix_store=pool, prefix_namespace="operator_two")
    one.generate(request("one"))
    two.generate(request("two"))
    assert pool.snapshot()["hits"] == 0 and pool.snapshot()["misses"] == 2
    with serving(one) as (_, server):
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        payload = mapping("untrusted") | {"prefix_namespace": "operator_two"}
        connection.request(
            "POST",
            "/generate",
            json.dumps(payload),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.close()
    assert two.prefix_metrics.snapshot()["attempts"]["miss"] == 1


@pytest.mark.parametrize(
    "max_bytes,prompts,publications,evictions,bypasses",
    [
        (8192, ((1, 2, 3, 4), (5, 4, 3, 2)), 2, 1, 0),
        (1, ((1, 2, 3, 4),), 0, 0, 1),
    ],
)
def test_http_scrape_distinguishes_pool_churn_from_lookup_misses(
    model, max_bytes, prompts, publications, evictions, bypasses
):
    pool = PrefixKVStore(max_bytes=max_bytes, max_entries=1)
    service = GenerationService(model, prefix_store=pool)
    uncached = GenerationService(model)

    def scrape(server):
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        try:
            connection.request("GET", "/metrics")
            response = connection.getresponse()
            assert response.status == 200
            text = response.read().decode()
        finally:
            connection.close()
        samples = {
            name: float(value)
            for line in text.splitlines()
            if line and not line.startswith("#")
            for name, value in (line.rsplit(" ", 1),)
        }
        for private in ("private-pool", service.model_sha256):
            assert private not in text
        assert text.count("outcome=") == 8  # Five serving and three prefix labels.
        return samples

    with serving(service) as (endpoint, server):
        for index, prompt in enumerate(prompts):
            identifier = f"private-pool-{index}"
            actual = measure_http_generation(endpoint, mapping(identifier, prompt))
            expected = uncached.generate(request(identifier, prompt))
            assert actual["outcome"] == "completed"
            assert actual["tokens"] == expected["tokens"]
        state = pool.snapshot()
        assert (state["publications"], state["evictions"], state["bypasses"]) == (
            publications,
            evictions,
            bypasses,
        )
        observed = scrape(server)
        assert observed['ai_journey_prefix_attempts_total{outcome="miss"}'] == len(
            prompts
        )
        assert observed['ai_journey_prefix_attempts_total{outcome="bypass"}'] == 0
        for field, count in (
            ("publications", publications),
            ("evictions", evictions),
            ("bypasses", bypasses),
        ):
            assert observed[f"ai_journey_prefix_pool_{field}_total"] == count
        assert observed["ai_journey_serving_generated_tokens_total"] == 3 * len(prompts)
        assert service.drain(timeout_seconds=1)
        drained = scrape(server)
        assert drained["ai_journey_prefix_pool_evictions_total"] == publications
        assert drained["ai_journey_prefix_entries"] == 0
        assert drained["ai_journey_prefix_resident_bytes"] == 0
        assert drained["ai_journey_serving_generated_tokens_total"] == 3 * len(prompts)


@pytest.mark.parametrize("control", ["cancel", "error", "deadline"])
def test_interruption_releases_leases_and_admission_without_replay(model, control):
    pool = PrefixKVStore(max_bytes=8192)
    service = GenerationService(model, prefix_store=pool)
    service.generate(request("warmup"))
    cancelled = threading.Event()

    def observe(index, token):
        assert pool.snapshot()["active_leases"] == 1
        if control == "error":
            raise RuntimeError("controlled delivery failure")
        cancelled.set()

    kwargs = {"on_token": observe, "cancelled": cancelled.is_set}
    if control == "deadline":
        kwargs = {"deadline": time.monotonic() - 1}
    if control == "error":
        with pytest.raises(RuntimeError, match="delivery"):
            service.generate(request("interrupted"), **kwargs)
    else:
        output = service.generate(request("interrupted"), **kwargs)
        assert output["finish_reason"] == (
            "cancelled" if control == "cancel" else "deadline"
        )
        assert len(output["tokens"]) == (1 if control == "cancel" else 0)
    assert pool.snapshot()["active_leases"] == pool.snapshot()["leased_bytes"] == 0
    assert service.health()["active_requests"] == 0
    assert service.generate(request("interrupted"))["finish_reason"] == "length"


def test_drain_clears_residents_without_invalidating_active_charged_lease(model):
    pool = PrefixKVStore(max_bytes=8192)
    service = GenerationService(model, prefix_store=pool)
    service.generate(request("warmup"))
    entered, release = threading.Event(), threading.Event()

    def pause(index, token):
        if index == 0:
            entered.set()
            assert release.wait(timeout=3)

    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(service.generate, request("active"), on_token=pause)
        try:
            assert entered.wait(timeout=3)
            assert not service.drain(timeout_seconds=0)
            state = pool.snapshot()
            assert state["resident_bytes"] == 0
            assert state["active_leases"] == 1 and state["leased_bytes"] > 0
        finally:
            release.set()
        assert len(pending.result(timeout=3)["tokens"]) == 3
    assert service.drain(timeout_seconds=1)
    assert pool.snapshot()["active_leases"] == pool.snapshot()["resident_bytes"] == 0


def test_prefix_metrics_concurrent_fixed_labels_and_atomic_rejection():
    metrics = PrefixMetrics(PrefixKVStore(max_bytes=1))
    observation = {
        "outcome": "hit",
        "reused_prompt_tokens": 3,
        "forwarded_prefill_tokens": 1,
    }
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda _: metrics.record(observation), range(100)))
    before = metrics.snapshot()
    for invalid in (
        observation | {"outcome": "private"},
        observation | {"reused_prompt_tokens": True},
    ):
        with pytest.raises(ValueError):
            metrics.record(invalid)
    assert metrics.snapshot() == before
    assert before["attempts"]["hit"] == 100
    assert before["reused_prompt_tokens"] == 300
    assert metrics.render_prometheus().count("outcome=") == 3


def test_existing_serve_cli_defaults_off_and_admits_operator_flags():
    parser = argparse.ArgumentParser()
    add_commands(parser.add_subparsers(dest="command"))
    required = [
        "serve",
        "--source",
        "source",
        "--spec",
        "spec",
        "--checkpoint",
        "checkpoint",
    ]
    assert parser.parse_args(required).prefix_cache_bytes == 0
    enabled = parser.parse_args(
        required
        + [
            "--prefix-cache-bytes",
            "8192",
            "--prefix-cache-entries",
            "4",
            "--prefix-namespace",
            "operator",
        ]
    )
    assert (
        enabled.prefix_cache_bytes,
        enabled.prefix_cache_entries,
        enabled.prefix_namespace,
    ) == (8192, 4, "operator")
