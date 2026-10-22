import pytest
from ai_journey.ddp_benchmark import percentile, benchmark_service
from ai_journey.ddp_service import GenerationService, GenerationRequest
from ai_journey.ddp_generation import SamplingConfig
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_config import GPT2Config


def test_empirical_percentiles_use_documented_nearest_rank():
    assert percentile([4, 1, 3, 2], 0.5) == 2
    assert percentile([4, 1, 3, 2], 0.95) == 4
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_serving_benchmark_excludes_warmup_and_checks_output_parity():
    service = GenerationService(GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)))
    request = GenerationRequest(
        "bench", (1, 2), SamplingConfig(max_new_tokens=2, temperature=0)
    )
    result = benchmark_service(service, request, repeats=3, warmup=1)
    assert result["tokens"] == 6
    assert result["tokens_per_second"] > 0
    assert result["output_parity"]
    assert (
        result["latency_seconds"]["p50"]
        <= result["latency_seconds"]["p95"]
        <= result["latency_seconds"]["p99"]
    )
    assert service.completed_requests == 4
