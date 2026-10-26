"""Direct payload restoration must enforce the same contract as archive loading."""

import pytest
import torch

from ai_journey.moe_checkpoint import (
    _digest,
    read_moe_checkpoint,
    restore_moe_payload,
)
from ai_journey.training_models import training_model_fingerprint
from test_moe_checkpoint import CONTRACT, SOURCE, save
from test_moe_training import state


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", False),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("schema_version", 999),
        ("step", False),
        ("step", 0.0),
    ],
)
def test_direct_restore_rejects_resealed_metadata_before_live_mutation(
    tmp_path, monkeypatch, field, value
):
    model, optimizer = state()
    path = tmp_path / "state.pt"
    save(path, model, optimizer, 0)
    payload = read_moe_checkpoint(path)
    payload[field] = value
    payload["checkpoint_sha256"] = _digest(
        {key: item for key, item in payload.items() if key != "checkpoint_sha256"}
    )
    before = training_model_fingerprint(model)
    optimizer_before = _digest(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(
        model,
        "load_state_dict",
        lambda *a, **k: pytest.fail("invalid metadata must fail before model mutation"),
    )
    with pytest.raises(ValueError, match="schema|step"):
        restore_moe_payload(
            payload,
            model=model,
            optimizer=optimizer,
            training_contract=CONTRACT,
            source_sha256=SOURCE,
        )
    assert training_model_fingerprint(model) == before
    assert _digest(optimizer.state_dict()) == optimizer_before
    assert torch.equal(torch.get_rng_state(), rng)


def test_direct_restore_preserves_valid_integer_step_contract(tmp_path):
    model, optimizer = state()
    path = tmp_path / "state.pt"
    save(path, model, optimizer, 0)
    restored, restored_optimizer = state()
    step = restore_moe_payload(
        read_moe_checkpoint(path),
        model=restored,
        optimizer=restored_optimizer,
        training_contract=CONTRACT,
        source_sha256=SOURCE,
    )
    assert type(step) is int and step == 0
    assert training_model_fingerprint(restored) == training_model_fingerprint(model)
