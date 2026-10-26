"""Storage candidates become usable only after collective typed restoration."""

import copy

import pytest
import torch

from ai_journey.checkpoint_journal import CheckpointJournal
from ai_journey.ddp_acceptance import exact_state
from ai_journey.ddp_checkpoint import (
    checkpoint_run_sha256,
    load_checkpoint_journal,
    save_checkpoint,
)
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from test_ddp_checkpoint import setup_state


def advance(model, optimizer):
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()


def snapshot(model, optimizer):
    return (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        torch.get_rng_state().clone(),
    )


def assert_unchanged(before, model, optimizer):
    assert exact_state(before[0], model.state_dict())
    assert exact_state(before[1], optimizer.state_dict())
    assert torch.equal(before[2], torch.get_rng_state())


@pytest.mark.parametrize(
    "corruption",
    ["bytes", "source", "world", "model", "moment", "counter", "rng", "step"],
)
def test_latest_invalid_candidate_falls_back_to_fully_valid_state(tmp_path, corruption):
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
        expected = snapshot(model, optimizer)
        store = CheckpointJournal(
            directory, run_sha256=checkpoint_run_sha256(model, config, b"public")
        )
        assert store.recorded_step == 1
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
        if corruption == "source":
            payload["source_sha256"] = "0" * 64
        elif corruption == "world":
            payload["world_size"] = 2
        elif corruption == "model":
            payload["model_sha256"] = "0" * 64
        elif corruption == "moment":
            next(iter(payload["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(
                1
            )
        elif corruption == "counter":
            next(iter(payload["optimizer"]["state"].values()))["step"] = torch.tensor(
                True
            )
        elif corruption == "rng":
            payload["rank_rngs"][0] = torch.zeros(3, dtype=torch.uint8)
        elif corruption == "step":
            payload["step"] = 1
        torch.save(payload, path)
        newest = store.publish(path, step=2)
        if corruption == "bytes":
            newest.path.write_bytes(b"damaged committed bytes")
        # A storage-intact semantic candidate is deliberately not called valid.
        if corruption != "bytes":
            assert store.candidates()[0] == newest
        with torch.no_grad():
            model.wte.weight.add_(5)
        torch.rand(3)
        recovery = load_checkpoint_journal(
            directory, model=model, optimizer=optimizer, config=config, source=b"public"
        )
        assert recovery.step == 1 and recovery.recorded_step == 2
        assert recovery.checkpoint_sha256 == store.candidates()[-1].sha256
        assert_unchanged(expected, model, optimizer)


@pytest.mark.parametrize("source", [b"public", b"different"])
def test_no_fully_valid_candidate_rejects_without_live_state_mutation(tmp_path, source):
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
        )
        payload = torch.load(path, weights_only=True)
        payload["optimizer"]["param_groups"][0]["capturable"] = True
        torch.save(payload, path)
        store = CheckpointJournal(
            directory, run_sha256=checkpoint_run_sha256(model, config, b"public")
        )
        store.publish(path, step=1)
        with torch.no_grad():
            model.wte.weight.add_(3)
        before = snapshot(model, optimizer)
        with pytest.raises(RuntimeError, match="fully valid|journal candidates"):
            load_checkpoint_journal(
                directory,
                model=model,
                optimizer=optimizer,
                config=config,
                source=source,
            )
        assert_unchanged(before, model, optimizer)


def test_path_replacement_after_enumeration_rejects_snapshot_and_falls_back(
    tmp_path, monkeypatch
):
    model, optimizer, config = setup_state()
    path, directory = tmp_path / "latest.pt", tmp_path / "journal"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        for step in (1, 2):
            advance(model, optimizer)
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                config=config,
                source=b"public",
                step=step,
                journal_directory=directory,
            )
        enumerate_candidates = CheckpointJournal.candidates

        def replace_after_enumeration(store):
            candidates = enumerate_candidates(store)
            candidates[0].path.write_bytes(candidates[-1].path.read_bytes())
            return candidates

        monkeypatch.setattr(CheckpointJournal, "candidates", replace_after_enumeration)
        assert (
            load_checkpoint_journal(
                directory,
                model=model,
                optimizer=optimizer,
                config=config,
                source=b"public",
            ).step
            == 1
        )


def test_manifest_failure_preserves_last_publication_and_ignores_orphan(
    tmp_path, monkeypatch
):
    import os

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
        before_path, before_manifest = (
            path.read_bytes(),
            (directory / "manifest.json").read_bytes(),
        )
        advance(model, optimizer)
        replace = os.replace

        def fail_manifest(source, destination):
            if destination == directory / "manifest.json":
                raise OSError("injected manifest fault")
            return replace(source, destination)

        with monkeypatch.context() as patched:
            patched.setattr(os, "replace", fail_manifest)
            with pytest.raises(RuntimeError, match="publication"):
                save_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    source=b"public",
                    step=2,
                    journal_directory=directory,
                )
        assert path.read_bytes() == before_path
        assert (directory / "manifest.json").read_bytes() == before_manifest
        assert (
            load_checkpoint_journal(
                directory,
                model=model,
                optimizer=optimizer,
                config=config,
                source=b"public",
            ).step
            == 1
        )
        store = CheckpointJournal(
            directory, run_sha256=checkpoint_run_sha256(model, config, b"public")
        )
        assert store.recorded_step == 1 and len(store.candidates()) == 1
        # Unknown orphan bytes are preserved for explicit inspection, not
        # treated as restorable or automatically deleted on recovery.
        assert len(list(directory.glob("*.pt"))) == 2


def test_nonadvancing_programmatic_publication_preserves_latest_and_manifest(tmp_path):
    model, optimizer, config = setup_state()
    path, directory = tmp_path / "latest.pt", tmp_path / "journal"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=0,
            journal_directory=directory,
        )
        before = path.read_bytes(), (directory / "manifest.json").read_bytes()
        with pytest.raises(RuntimeError, match="publication"):
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                config=config,
                source=b"public",
                step=0,
                journal_directory=directory,
            )
        assert before == (path.read_bytes(), (directory / "manifest.json").read_bytes())


def test_transport_failure_is_not_misclassified_as_a_rejected_candidate(
    tmp_path, monkeypatch
):
    from ai_journey import ddp_checkpoint

    model, optimizer, config = setup_state()
    path, directory = tmp_path / "latest.pt", tmp_path / "journal"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=0,
            journal_directory=directory,
        )
        attempts = []

        def broken_transport(*args, **kwargs):
            attempts.append(args)
            raise RuntimeError("connection lost")

        monkeypatch.setattr(ddp_checkpoint, "load_checkpoint", broken_transport)
        before = snapshot(model, optimizer)
        with pytest.raises(RuntimeError, match="connection lost"):
            load_checkpoint_journal(
                directory,
                model=model,
                optimizer=optimizer,
                config=config,
                source=b"public",
            )
        assert len(attempts) == 1
        assert_unchanged(before, model, optimizer)
