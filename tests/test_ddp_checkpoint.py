import pytest
import torch
from ai_journey.ddp_checkpoint import save_checkpoint
from ai_journey.ddp_checkpoint import load_checkpoint
from ai_journey.ddp_checkpoint import bounded_checkpoint_load
import zipfile
import os
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_training import GPT2TrainingConfig


def setup_state():
    model = GPT2LanguageModel(GPT2Config(256, 4, 8, 2, 1))
    config = GPT2TrainingConfig(
        optimizer_steps=2,
        micro_batch_size=1,
        accumulation_steps=1,
        sequence_length=4,
        max_learning_rate=0.01,
        min_learning_rate=0.001,
        warmup_steps=0,
        weight_decay=0,
        gradient_clip=1,
        seed=48,
    )
    return model, torch.optim.AdamW(model.parameters(), lr=0.01), config


def test_rank_zero_atomically_publishes_complete_replicated_state(tmp_path):
    model, optimizer, config = setup_state()
    path = tmp_path / "checkpoints" / "state.pt"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        digest = save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=0,
        )
    payload = torch.load(path, weights_only=True)
    assert payload["model_sha256"] == digest
    assert payload["step"] == 0
    assert len(payload["rank_rngs"]) == 1
    assert list(path.parent.iterdir()) == [path]


def test_publication_failure_is_a_collective_error(tmp_path):
    model, optimizer, config = setup_state()
    parent = tmp_path / "not-a-directory"
    parent.write_bytes(b"keep")
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        with pytest.raises(RuntimeError, match="checkpoint publication"):
            save_checkpoint(
                parent / "state.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                source=b"public",
                step=0,
            )
    assert parent.read_bytes() == b"keep"


def test_checkpoint_restores_state_and_rejects_another_corpus(tmp_path):
    model, optimizer, config = setup_state()
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    path = tmp_path / "state.pt"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=1,
        )
        with torch.no_grad():
            model.wte.weight.add_(1)
        assert (
            load_checkpoint(
                path, model=model, optimizer=optimizer, config=config, source=b"public"
            )
            == 1
        )
        with pytest.raises(RuntimeError, match="checkpoint validation"):
            load_checkpoint(
                path, model=model, optimizer=optimizer, config=config, source=b"other"
            )
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_compressed_archive_is_rejected_before_torch_load(tmp_path, monkeypatch):
    path = tmp_path / "compressed.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("storage", b"x" * 100000)

    def unexpected(*args, **kwargs):
        raise AssertionError("deserialization must not run")

    monkeypatch.setattr(torch, "load", unexpected)
    with pytest.raises(ValueError, match="expanded storage"):
        bounded_checkpoint_load(path, 200000)
    link = tmp_path / "alias.pt"
    link.symlink_to(path)
    with pytest.raises(OSError):
        bounded_checkpoint_load(link, 200000)


def test_malformed_adamw_moment_shape_rolls_back_loaded_parameters(tmp_path):
    model, optimizer, config = setup_state()
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    path = tmp_path / "state.pt"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=1,
        )
        payload = torch.load(path, weights_only=True)
        state = next(iter(payload["optimizer"]["state"].values()))
        state["exp_avg"] = torch.zeros(1)
        torch.save(payload, path)
        original = model.wte.weight.detach().clone()
        with pytest.raises(RuntimeError, match="restoration"):
            load_checkpoint(
                path, model=model, optimizer=optimizer, config=config, source=b"public"
            )
        assert torch.equal(model.wte.weight, original)


def test_distributed_checkpoint_fifo_is_rejected_without_waiting_for_a_writer(
    tmp_path, monkeypatch
):
    path = tmp_path / "checkpoint.fifo"
    os.mkfifo(path)
    original_open = os.open

    def checked_open(candidate, flags, *args, **kwargs):
        assert flags & os.O_NONBLOCK, "opening a FIFO must never block the loader"
        return original_open(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", checked_open)
    with pytest.raises(ValueError, match="regular-file"):
        bounded_checkpoint_load(path, 1024)


def test_distributed_checkpoint_duplicate_archive_members_reject_before_loading(
    tmp_path, monkeypatch
):
    path = tmp_path / "duplicate.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("state/data", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("state/data", b"second")

    def unexpected(*args, **kwargs):
        raise AssertionError("ambiguous archives must not reach torch.load")

    monkeypatch.setattr(torch, "load", unexpected)
    with pytest.raises(ValueError, match="expanded storage"):
        bounded_checkpoint_load(path, 1024)
