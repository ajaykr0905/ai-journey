"""One rank's local scoring error must not strand a peer in token reduction."""

import json
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_evaluation import Document, evaluate_distributed, evaluate_documents
from ai_journey.ddp_runtime import process_group
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel

COMMON_ERROR = "at least one rank failed document evaluation"


def evaluation_worker(rank, directory, fault, empty_peer):
    directory = Path(directory)
    torch.set_num_threads(1)
    torch.manual_seed(53)
    model = GPT2LanguageModel(GPT2Config(256, 4, 4, 2, 1)).double().train()
    model.ln_f.eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    original = model.forward
    documents = (Document("one", b"\x01\x01\x01\x01"),)
    if not empty_peer:
        documents += (Document("two", b"\x01\x01\x01"),)
    failed_rank = 0 if empty_peer else 1
    if rank == failed_rank:

        def rejected(inputs):
            if fault == "forward":
                raise RuntimeError("controlled local CPU forward failure")
            if fault == "totals":
                logits = torch.full((*inputs.shape, 256), -1e308, dtype=torch.float64)
                logits[..., 0] = 0
                return logits
            return torch.full((*inputs.shape, 256), float("nan"), dtype=torch.float64)

        model.forward = rejected
    topology = Topology(rank, 2)
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    with process_group(
        topology, rendezvous=(directory / "rendezvous").as_uri(), timeout_seconds=5
    ):
        error = None
        try:
            evaluate_distributed(model, documents, topology)
        except (ValueError, RuntimeError) as failure:
            error = str(failure)
        result = {"error": error, "same_group_retry": False}
        if error == COMMON_ERROR:
            assert modes == [module.training for module in model.modules()]
            assert torch.equal(rng, torch.get_rng_state())
            assert all(
                torch.equal(before, parameter.grad)
                for before, parameter in zip(gradients, model.parameters(), strict=True)
            )
            model.forward = original
            actual = evaluate_distributed(model, documents, topology)
            expected = evaluate_documents(model, documents)
            assert actual["tokens"] == expected["tokens"]
            assert actual["mean_nll"] == pytest.approx(expected["mean_nll"], abs=1e-12)
            result["same_group_retry"] = True
            result["target_tokens"] = actual["tokens"]
        (directory / f"rank-{rank}.json").write_text(json.dumps(result))


@pytest.mark.parametrize(
    "fault,empty_peer",
    [("logits", False), ("forward", False), ("totals", False), ("logits", True)],
)
def test_rank_local_evaluation_failure_is_common_and_group_remains_usable(
    tmp_path, fault, empty_peer
):
    mp.spawn(
        evaluation_worker, args=(str(tmp_path), fault, empty_peer), nprocs=2, join=True
    )
    receipts = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)
    ]
    assert [receipt["error"] for receipt in receipts] == [COMMON_ERROR, COMMON_ERROR]
    assert all(receipt["same_group_retry"] for receipt in receipts)
    assert all(
        receipt["target_tokens"] == (3 if empty_peer else 5) for receipt in receipts
    )
