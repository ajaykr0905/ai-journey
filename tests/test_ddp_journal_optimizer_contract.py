"""Same-run journal recovery must retain the caller's optimizer computation."""

import pytest
import torch

from ai_journey.checkpoint_journal import CheckpointJournal
from ai_journey.ddp_checkpoint import (
    checkpoint_run_sha256,
    load_checkpoint,
    load_checkpoint_journal,
    save_checkpoint,
)
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from test_ddp_checkpoint import setup_state
from test_ddp_checkpoint_journal import advance


@pytest.mark.parametrize(
    "field,value",
    [("weight_decay", 0.5), ("betas", (0.1, 0.5)), ("eps", 0.1), ("maximize", True)],
)
def test_finite_edited_optimizer_controls_are_not_same_run_candidates(
    tmp_path, field, value
):
    model, optimizer, config = setup_state()
    path, directory = tmp_path / "latest.pt", tmp_path / "journal"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        advance(model, optimizer)
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=1,
            journal_directory=directory,
        )
        run_identity = checkpoint_run_sha256(
            model, config, b"public", optimizer=optimizer
        )
        advance(model, optimizer)
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=2,
        )
        payload = torch.load(path, weights_only=True)
        payload["optimizer"]["param_groups"][0][field] = value
        torch.save(payload, path)
        store = CheckpointJournal(directory, run_sha256=run_identity)
        store.publish(path, step=2)
        assert store.candidates()[0].step == 2  # Storage integrity is valid.
        expected = optimizer.param_groups[0][field]
        restored = load_checkpoint_journal(
            directory, model=model, optimizer=optimizer, config=config, source=b"public"
        )
        assert restored.step == 1
        assert optimizer.param_groups[0][field] == expected


def test_actual_optimizer_controls_and_order_bind_identity_but_lr_state_do_not(
    tmp_path,
):
    model, optimizer, config = setup_state()
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):

        def identity():
            return checkpoint_run_sha256(model, config, b"public", optimizer=optimizer)

        original = identity()
        optimizer.param_groups[0]["lr"] = 0.003
        advance(model, optimizer)
        assert identity() == original
        for name, value in (
            ("weight_decay", 0.5),
            ("betas", (0.2, 0.7)),
            ("eps", 0.01),
            ("amsgrad", True),
            ("maximize", True),
            ("foreach", False),
        ):
            previous = optimizer.param_groups[0][name]
            optimizer.param_groups[0][name] = value
            assert identity() != original
            optimizer.param_groups[0][name] = previous
        optimizer.param_groups[0]["params"].reverse()
        assert identity() != original


def test_custom_group_controls_restore_while_scheduled_lr_can_differ(tmp_path):
    model, _, config = setup_state()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": parameters[::2], "weight_decay": 0.02},
            {"params": parameters[1::2], "weight_decay": 0.03},
        ],
        lr=0.006,
        betas=(0.7, 0.8),
        eps=1e-6,
    )
    path, directory = tmp_path / "latest.pt", tmp_path / "journal"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        advance(model, optimizer)
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=1,
            journal_directory=directory,
        )
        original_identity = checkpoint_run_sha256(
            model, config, b"public", optimizer=optimizer
        )
        for group in optimizer.param_groups:
            group["lr"] = 0.009
        assert (
            checkpoint_run_sha256(model, config, b"public", optimizer=optimizer)
            == original_identity
        )
        recovered = load_checkpoint_journal(
            directory, model=model, optimizer=optimizer, config=config, source=b"public"
        )
        assert recovered.step == 1
        assert [group["weight_decay"] for group in optimizer.param_groups] == [
            0.02,
            0.03,
        ]
        assert all(
            group["betas"] == (0.7, 0.8)
            and group["eps"] == 1e-6
            and group["lr"] == 0.006
            for group in optimizer.param_groups
        )


def test_legacy_single_file_keeps_its_existing_control_restore_contract(tmp_path):
    model, optimizer, config = setup_state()
    path = tmp_path / "legacy.pt"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        advance(model, optimizer)
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=1,
        )
        payload = torch.load(path, weights_only=True)
        assert payload["schema_version"] == 1
        payload["optimizer"]["param_groups"][0]["weight_decay"] = 0.5
        torch.save(payload, path)
        assert (
            load_checkpoint(
                path, model=model, optimizer=optimizer, config=config, source=b"public"
            )
            == 1
        )
        assert optimizer.param_groups[0]["weight_decay"] == 0.5
