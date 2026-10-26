from pathlib import Path
import zipfile

import pytest
import torch

from ai_journey.checkpoint_archive import load_checkpoint_archive
from ai_journey.gpt2_config import GPT2Config
from ai_journey.moe_config import MoEConfig
from ai_journey.moe_resources import require_moe_job_budget
from ai_journey.training_models import build_training_model


def test_small_admitted_expert_model_has_an_explicit_bounded_archive_path(tmp_path):
    config = MoEConfig(
        GPT2Config(
            vocab_size=256,
            context_length=4,
            embedding_dim=4,
            head_count=2,
            layer_count=8,
        ),
        expert_count=64,
        moe_layers=tuple(range(8)),
    )
    estimate = require_moe_job_budget(
        (config,), seeds=1, steps=1, batch_size=1, evaluation_windows=1
    )
    assert estimate["peak_model_parameters"] == 79_640
    model = build_training_model(config)
    path = tmp_path / "expert-weights.pt"
    torch.save(model.state_dict(), path)
    with zipfile.ZipFile(path) as archive:
        records = len(archive.infolist())
    assert 1024 < records < 16_384
    assert path.stat().st_size < 2 * 1024 * 1024
    with pytest.raises(ValueError, match="expanded storage"):
        load_checkpoint_archive(path, max_bytes=2 * 1024 * 1024)
    loaded = load_checkpoint_archive(
        path, max_bytes=2 * 1024 * 1024, max_records=records
    )
    assert loaded.keys() == model.state_dict().keys()
    assert all(
        torch.equal(value, loaded[name]) for name, value in model.state_dict().items()
    )
    with pytest.raises(ValueError, match="expanded storage"):
        load_checkpoint_archive(
            path, max_bytes=2 * 1024 * 1024, max_records=records - 1
        )


def test_full_dense_adamw_archive_fits_the_predeclared_record_estimate(tmp_path):
    config = MoEConfig(
        GPT2Config(
            vocab_size=256,
            context_length=4,
            embedding_dim=4,
            head_count=2,
            layer_count=8,
        ),
        expert_count=64,
        moe_layers=tuple(range(8)),
    )
    model = build_training_model(config)
    optimizer = torch.optim.AdamW(model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    path = tmp_path / "full-optimizer.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        },
        path,
    )
    estimate = require_moe_job_budget(
        (config,), seeds=1, steps=1, batch_size=1, evaluation_windows=1
    )
    budget = estimate["peak_checkpoint_records"]
    with zipfile.ZipFile(path) as archive:
        assert len(archive.infolist()) <= budget < 16_384
    loaded = load_checkpoint_archive(
        path, max_bytes=8 * 1024 * 1024, max_records=budget
    )
    assert len(loaded["optimizer"]["state"]) == len(tuple(model.parameters()))


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.0, "1024", 16_385, 10**100])
def test_invalid_record_admission_is_rejected_before_opening_a_path(
    invalid, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid archive admission opened a file")

    monkeypatch.setattr("ai_journey.checkpoint_archive.os.open", forbidden)
    with pytest.raises(ValueError, match="max_records"):
        load_checkpoint_archive(Path("not-opened"), max_bytes=1024, max_records=invalid)


@pytest.mark.parametrize("kind", ["compressed", "duplicate", "oversized"])
def test_larger_explicit_record_allowance_keeps_other_archive_safeguards(
    tmp_path, kind
):
    path = tmp_path / "invalid.pt"
    compression = zipfile.ZIP_DEFLATED if kind == "compressed" else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr(
            "archive/data.pkl", b"x" * (2048 if kind == "oversized" else 4)
        )
        if kind == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("archive/data.pkl", b"more")
    with pytest.raises(ValueError, match="budget"):
        load_checkpoint_archive(path, max_bytes=1024, max_records=16_384)
