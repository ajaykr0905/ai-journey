import copy
import json
from pathlib import Path

import pytest
import torch

from ai_journey.ddp_evidence import verify_evidence, digest
from ai_journey.ddp_comparison import verify_comparison
from ai_journey.ddp_launch import launch_training
from ai_journey.ddp_checkpoint import validate_optimizer_state
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.ddp_training import accumulated_update
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]


def reseal(report):
    report["evidence_sha256"] = digest(
        {key: value for key, value in report.items() if key != "evidence_sha256"}
    )
    return report


@pytest.mark.parametrize(
    "field,value", [("schema_version", True), ("model_sha256", "not-a-digest")]
)
def test_resealed_forged_worker_evidence_is_rejected(field, value):
    report = json.loads(
        (ROOT / "artifacts/ddp-lab-2026-10-22/comparison.json").read_text()
    )["arms"][1]["evidence"]
    report[field] = value
    with pytest.raises(ValueError):
        verify_evidence(reseal(report))


def test_resealed_comparison_cannot_relax_the_predeclared_error_threshold():
    report = json.loads(
        (ROOT / "artifacts/ddp-lab-2026-10-22/comparison.json").read_text()
    )
    report["absolute_tolerance"] = 1
    with pytest.raises(ValueError, match="declared tolerance"):
        verify_comparison(reseal(report))


@pytest.mark.parametrize(
    "mode,phase",
    [("nonfinite", "finite gradient validation"), ("schedule", "schedule agreement")],
)
def test_one_bad_rank_stops_all_workers_before_a_checkpoint(tmp_path, mode, phase):
    with pytest.raises(RuntimeError, match="worker failed"):
        launch_training(
            source=ROOT / "data/day-19-demo-names.txt",
            spec=ROOT / "config/ddp-lab.json",
            checkpoint=tmp_path / "state.pt",
            output_dir=tmp_path / mode,
            failure_rank=1,
            failure_mode=mode,
        )
    assert not (tmp_path / "state.pt").exists()
    assert phase in (tmp_path / mode / "rank-1.log").read_text()


def test_nonfinite_optimizer_output_is_rolled_back_on_every_rank(tmp_path):
    model = GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1))
    saved = copy.deepcopy(model.state_dict())

    class PoisonSGD(torch.optim.SGD):
        def step(self, closure=None):
            super().step(closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("inf"))

    optimizer = PoisonSGD(model.parameters(), lr=0.01)
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        with pytest.raises(RuntimeError, match="optimizer update"):
            accumulated_update(
                DistributedDataParallel(model),
                optimizer,
                [(torch.tensor([[0, 1]]), torch.tensor([[1, 2]]))],
            )
    assert all(
        torch.equal(value, saved[name]) for name, value in model.state_dict().items()
    )


def test_optimizer_hyperparameter_poisoning_is_rejected():
    model = GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.param_groups[0]["weight_decay"] = float("nan")
    with pytest.raises(ValueError, match="hyperparameters"):
        validate_optimizer_state(model, optimizer, step=0)
