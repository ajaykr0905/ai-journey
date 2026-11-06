import copy

from ai_journey.ddp_evaluation import Document
from ai_journey.evaluation_suite import (
    evaluate_heldout_likelihood,
    result_from_hellaswag,
    result_from_lambada,
)
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.lambada import _payload_digest
from test_hellaswag_evidence import evidence_fixture as hellaswag_fixture
from test_lambada import evidence_fixture as lambada_fixture


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
