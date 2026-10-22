import pytest
from ai_journey.ddp_evaluation import Document, documents_from_lines
from ai_journey.ddp_evaluation import split_documents
from ai_journey.ddp_evaluation import evaluation_batches
from ai_journey.ddp_evaluation import evaluate_documents
from ai_journey.ddp_evaluation import evaluate_distributed
from ai_journey.ddp_evaluation import score_choices
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
import torch
import math


def test_public_lines_keep_individual_document_boundaries():
    documents = documents_from_lines(b"alice\n\nbob\n")
    assert [document.identifier for document in documents] == ["line-1", "line-3"]
    assert [document.content for document in documents] == [b"alice\n", b"bob\n"]
    assert len({document.digest for document in documents}) == 2


@pytest.mark.parametrize("source", [b"", b"alice\nalice\n"])
def test_empty_or_duplicated_public_documents_are_rejected(source):
    with pytest.raises(ValueError):
        documents_from_lines(source)


def test_one_byte_document_cannot_supply_a_next_token_target():
    with pytest.raises(ValueError):
        Document("short", b"x")


def test_document_split_is_disjoint_and_invariant_to_input_order():
    documents = documents_from_lines(b"alice\nbob\ncarol\ndave\n")
    train, valid = split_documents(documents, validation_count=2)
    assert not {doc.digest for doc in train} & {doc.digest for doc in valid}
    assert (train, valid) == split_documents(reversed(documents), validation_count=2)
    assert len(train) == len(valid) == 2


def test_eval_batches_score_every_target_once_without_cross_document_pairs():
    documents = [Document("a", b"abcdef"), Document("b", b"XYZ")]
    batches = list(evaluation_batches(documents, context_length=3, batch_size=2))
    targets = [target for _, y, mask in batches for target in y[mask].tolist()]
    assert targets == list(b"bcdefYZ")
    assert sum(mask.sum().item() for _, _, mask in batches) == 7
    assert batches[0][2].tolist() == [[True, True, True], [True, True, False]]


def test_document_evaluation_matches_uniform_oracle_and_preserves_state():
    model = GPT2LanguageModel(GPT2Config(256, 4, 8, 2, 1)).double().train()
    model.ln_f.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    before = torch.get_rng_state().clone()
    result = evaluate_documents(model, [Document("public", b"abcdefg")])
    assert result["tokens"] == 6
    assert result["mean_nll"] == pytest.approx(math.log(256))
    assert result["bits_per_byte"] == pytest.approx(8)
    assert result["perplexity"] == pytest.approx(256)
    assert model.training and not model.ln_f.training
    assert torch.equal(before, torch.get_rng_state())


def test_distributed_evaluation_reduces_every_document_target(tmp_path):
    model = GPT2LanguageModel(GPT2Config(256, 4, 8, 2, 1)).double()
    documents = documents_from_lines(b"alice\nbob\ncarol\n")
    reference = evaluate_documents(model, documents)
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        actual = evaluate_distributed(model, documents, Topology(0, 1))
    assert actual["tokens"] == reference["tokens"]
    assert actual["mean_nll"] == reference["mean_nll"]


def test_choice_scoring_excludes_prompt_and_normalizes_candidate_lengths():
    model = GPT2LanguageModel(GPT2Config(256, 4, 8, 2, 1)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = score_choices(model, b"long public prompt", [b"a", b"bbbb"])
    assert [score["tokens"] for score in result["scores"]] == [1, 4]
    assert [score["mean_nll"] for score in result["scores"]] == pytest.approx(
        [math.log(256)] * 2
    )
    assert result["prediction"] == 0
