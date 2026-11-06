import copy

from ai_journey.ddp_evaluation import Document
from ai_journey.evaluation_suite import (
    EvaluationResult,
    EvaluationSuiteError,
    build_evaluation_suite,
    evaluate_heldout_likelihood,
    result_from_hellaswag,
    result_from_lambada,
    verify_evaluation_suite,
)
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.lambada import _payload_digest
from test_hellaswag_evidence import evidence_fixture as hellaswag_fixture
from test_lambada import evidence_fixture as lambada_fixture
import pytest


def test_heldout_adapter_binds_model_documents_and_lower_is_better_metric():
    model = GPT2LanguageModel(GPT2Config(256, 8, 8, 2, 1)).double()
    result = evaluate_heldout_likelihood(
        model,
        [Document("public-a", b"abcdef"), Document("public-b", b"uvwxyz")],
    )
    assert result.name == "heldout_likelihood"
    assert result.primary_metric == "bits_per_byte"
    assert result.higher_is_better is False
    assert result.sample_count == 10


def test_benchmark_adapters_preserve_verified_scope_and_model_identity():
    hellaswag = result_from_hellaswag(hellaswag_fixture())
    lambada = result_from_lambada(lambada_fixture())
    assert hellaswag.primary_metric == "accuracy"
    assert hellaswag.eligible is True
    assert lambada.primary_metric == "exact_match_accuracy"
    assert lambada.eligible is True
    assert len(hellaswag.dataset_sha256) == len(lambada.dataset_sha256) == 64


def test_lambada_adapter_propagates_contamination_ineligibility():
    evidence = copy.deepcopy(lambada_fixture())
    contamination = evidence["run"]["contamination"]
    contamination["full_text_match_rows"] = [7]
    contamination["contaminated_rows"] = [7]
    contamination["passed"] = False
    evidence["claim_status"] = "contamination_detected"
    evidence["evidence_sha256"] = _payload_digest(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    assert result_from_lambada(evidence).eligible is False


def suite_results(*, eligible=True, model="a" * 64):
    return [
        EvaluationResult(
            "heldout_likelihood",
            model,
            "1" * 64,
            "bits_per_byte",
            7.9,
            False,
            200,
            True,
            "bounded held-out split",
        ),
        EvaluationResult(
            "hellaswag_validation",
            model,
            "2" * 64,
            "accuracy",
            0.25,
            True,
            100,
            True,
            "bounded validation slice",
        ),
        EvaluationResult(
            "lambada_last_word",
            model,
            "3" * 64,
            "exact_match_accuracy",
            0.0,
            True,
            100,
            eligible,
            "bounded test slice",
        ),
    ]


def test_suite_seals_three_axes_for_one_model():
    suite = build_evaluation_suite(reversed(suite_results()))
    assert verify_evaluation_suite(suite) == suite
    assert suite["claim_status"] == "complete_bounded_three_axis"
    assert [item["name"] for item in suite["evaluations"]] == [
        "heldout_likelihood",
        "hellaswag_validation",
        "lambada_last_word",
    ]


def test_suite_blocks_claim_when_an_evaluation_is_ineligible():
    suite = build_evaluation_suite(suite_results(eligible=False))
    assert suite["claim_status"] == "blocked_ineligible_evaluation"


def test_suite_rejects_missing_axes_and_mixed_models():
    with pytest.raises(EvaluationSuiteError, match="three"):
        build_evaluation_suite(suite_results()[:2])
    mixed = suite_results()
    mixed[-1] = EvaluationResult(**{**mixed[-1].to_mapping(), "model_sha256": "b" * 64})
    with pytest.raises(EvaluationSuiteError, match="model"):
        build_evaluation_suite(mixed)
