from ai_journey.inference_batching import (
    BatchStep,
    InferenceRequest,
    simulate_continuous_batching,
)


def test_continuous_batching_reuses_a_completed_slot_without_waiting_for_peer():
    requests = (
        InferenceRequest("a", 0, 4, 1),
        InferenceRequest("b", 0, 4, 3),
        InferenceRequest("c", 1, 2, 1),
    )

    steps = simulate_continuous_batching(
        requests, max_batch_size=2, max_resident_tokens=12
    )

    assert steps == (
        BatchStep(0, ("a", "b"), ("a", "b"), ("a",), 12),
        BatchStep(1, ("c",), ("b", "c"), ("c",), 10),
        BatchStep(2, (), ("b",), ("b",), 7),
    )


def test_waiting_request_is_admitted_after_resident_tokens_are_released():
    requests = (
        InferenceRequest("large", 0, 8, 2),
        InferenceRequest("next", 0, 4, 1),
    )

    steps = simulate_continuous_batching(
        requests, max_batch_size=2, max_resident_tokens=10
    )

    assert [step.admitted for step in steps] == [("large",), (), ("next",)]
