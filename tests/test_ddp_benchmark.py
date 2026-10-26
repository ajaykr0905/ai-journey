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


class StubService:
    model_sha256 = "a" * 64

    def __init__(self, **changes):
        self.changes = changes

    def generate(self, request):
        return {
            "model_sha256": self.model_sha256,
            "tokens": [1],
            "finish_reason": "length",
            **self.changes,
        }


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 2, float("nan")],
        [0, 1, 2, float("inf")],
        [0, 1, 2, True],
        [0, 1, 2, "3"],
        [0, 1, 2, 1.5, 2.5, 3],
    ],
)
def test_benchmark_rejects_invalid_absolute_clock_and_between_request_regressions(
    values,
):
    ticks = iter(values)
    request = GenerationRequest("bench", (1,), SamplingConfig(max_new_tokens=1))
    with pytest.raises(ValueError, match="clock"):
        benchmark_service(
            StubService(),
            request,
            repeats=2 if len(values) > 4 else 1,
            warmup=0,
            clock=lambda: next(ticks),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"finish_reason": "cancelled"},
        {"finish_reason": "deadline"},
        {"finish_reason": "error"},
        {"model_sha256": "b" * 64},
        {"tokens": (1,)},
    ],
)
def test_benchmark_never_reports_incomplete_or_mismatched_work_as_parity(changes):
    request = GenerationRequest("bench", (1,), SamplingConfig(max_new_tokens=1))
    with pytest.raises(RuntimeError, match="completed model-bound"):
        benchmark_service(StubService(**changes), request, repeats=1)
