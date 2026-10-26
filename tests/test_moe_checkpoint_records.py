"""Many small experts require record admission independent of parameter bytes."""

from hashlib import sha256
import zipfile

import pytest
import torch

from ai_journey.checkpoint_archive import MAX_CHECKPOINT_RECORDS
from ai_journey.gpt2_config import GPT2Config
from ai_journey.moe_checkpoint import load_moe_checkpoint, save_moe_checkpoint
from ai_journey.moe_config import MoEConfig, expected_moe_parameter_count
from ai_journey.moe_inference import load_moe_model_checkpoint
from ai_journey.moe_resources import expected_checkpoint_records
from ai_journey.moe_training import build_moe_optimizer, train_moe_step
from ai_journey.training_models import build_training_model, training_model_fingerprint
from test_moe_training import state


def test_initialized_many_expert_checkpoint_loads_and_resumes_inside_declared_budget(
    tmp_path,
):
    config = MoEConfig(
        GPT2Config(256, 4, 4, 1, 8), expert_count=64, moe_layers=tuple(range(8))
    )
    model = build_training_model(config)
    optimizer = build_moe_optimizer(model, learning_rate=0.001)
    tokens = torch.tensor([[1, 2, 3]])
    train_moe_step(model, optimizer, tokens, tokens)
    source, contract = (
        b"public many-expert fixture",
        {"batch_order": "fixed", "balance_weight": 0.01},
    )
    path = tmp_path / "experts.pt"
    save_moe_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        training_contract=contract,
        source_sha256=sha256(source).hexdigest(),
    )
    with zipfile.ZipFile(path) as archive:
        records = len(archive.infolist())
    assert (
        1024 < records <= expected_checkpoint_records(config) <= MAX_CHECKPOINT_RECORDS
    )
    clone = build_training_model(config)
    clone_optimizer = build_moe_optimizer(clone, learning_rate=0.001)
    assert (
        load_moe_checkpoint(
            path,
            model=clone,
            optimizer=clone_optimizer,
            training_contract=contract,
            source_sha256=sha256(source).hexdigest(),
        )
        == 1
    )
    assert training_model_fingerprint(clone) == training_model_fingerprint(model)
    rng = torch.get_rng_state().clone()
    resumed = train_moe_step(clone, clone_optimizer, tokens, tokens)
    torch.set_rng_state(rng)
    assert resumed == train_moe_step(model, optimizer, tokens, tokens)
    frozen = load_moe_model_checkpoint(
        path, config=config, training_contract=contract, source=source
    )
    assert not frozen.training


def test_small_parameter_model_with_excess_records_is_rejected_before_allocation(
    monkeypatch,
):
    config = MoEConfig(
        GPT2Config(256, 4, 2, 1, 64), expert_count=64, moe_layers=tuple(range(64))
    )
    assert expected_moe_parameter_count(config) < 5_000_000
    assert expected_checkpoint_records(config) > MAX_CHECKPOINT_RECORDS
    monkeypatch.setattr(
        "ai_journey.training_models.MoEDecoderModel",
        lambda _: pytest.fail("must not allocate"),
    )
    with pytest.raises(ValueError, match="checkpoint-record"):
        build_training_model(config)


def test_checkpoint_writer_rejects_unsupported_record_work_before_replacing_file(
    tmp_path, monkeypatch
):
    model, optimizer = state()
    path = tmp_path / "preserved.pt"
    path.write_bytes(b"existing durable checkpoint")
    monkeypatch.setattr(
        "ai_journey.moe_checkpoint.expected_checkpoint_records",
        lambda _: MAX_CHECKPOINT_RECORDS + 1,
    )
    with pytest.raises(ValueError, match="record-work"):
        save_moe_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            step=0,
            training_contract={"seed": 53},
            source_sha256=sha256(b"public").hexdigest(),
        )
    assert path.read_bytes() == b"existing durable checkpoint"


@pytest.mark.parametrize(
    "name,value",
    [("amsgrad", True), ("maximize", True), ("decoupled_weight_decay", False)],
)
def test_other_optimizer_algorithms_cannot_invalidate_reference_record_bound(
    name, value
):
    model, optimizer = state()
    optimizer.param_groups[0][name] = value
    with pytest.raises(ValueError, match="decoupled AdamW"):
        train_moe_step(model, optimizer, torch.tensor([[1]]), torch.tensor([[2]]))
