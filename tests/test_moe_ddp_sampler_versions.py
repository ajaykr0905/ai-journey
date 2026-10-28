"""Explicit sampler versions preserve legacy replay and shuffle full v2 epochs."""

import copy
from hashlib import sha256
import json
from pathlib import Path
import random

import pytest
import torch

from ai_journey.ddp_acceptance import exact_state
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.moe_ddp_checkpoint import (
    load_moe_ddp_checkpoint,
    save_moe_ddp_checkpoint,
)
from ai_journey.moe_ddp_cli import main
from ai_journey.moe_ddp_data import public_moe_ddp_schedule, public_moe_ddp_window
from ai_journey.moe_ddp_evidence import read_moe_ddp_evidence
from ai_journey.moe_ddp_plan import (
    PublicMoEDDPPlan,
    canonical,
    public_moe_ddp_plan_from_mapping,
    read_public_moe_ddp_plan,
    source_admission,
)
from ai_journey.moe_ddp_training import MoEDDPTrainingSession

LEGACY_SHA = "2be586d9d1f36a78a0e52a25c87d072259fea200ba4d1999e477502a69647c6e"


def _source(sample_count=10):
    length = sample_count * 4 + 1
    source = Path("data/day-19-demo-names.txt").read_bytes()[:length]
    assert len(source) == length
    return source


def _mapping(version=2, sample_count=10):
    mapping = json.loads(Path("config/moe-ddp-cpu.json").read_text())
    if version == 2:
        mapping["schema"] = "ai-journey-public-moe-ddp-plan-v2"
        mapping["sampler"] = "rank-strided-full-epoch-shuffle-v2"
    mapping["source"]["name"] += f" (first {sample_count * 4 + 1} bytes)"
    mapping["source"]["revision"] += f";prefix_bytes={sample_count * 4 + 1}"
    mapping["source"]["sha256"] = sha256(_source(sample_count)).hexdigest()
    return mapping


def test_v2_shuffles_all_samples_before_dropping_the_incomplete_window():
    plan = public_moe_ddp_plan_from_mapping(_mapping())

    # Python Random("53:0") over all ten IDs starts [9, 0, 5, 6].
    assert public_moe_ddp_schedule(plan, 41, rank=0, update_index=0)["sample_ids"] == [
        [9],
        [5],
    ]
    assert public_moe_ddp_schedule(plan, 41, rank=1, update_index=0)["sample_ids"] == [
        [0],
        [6],
    ]


def test_default_v1_keeps_exact_canonical_bytes_hash_and_data_contract():
    parsed = public_moe_ddp_plan_from_mapping(
        json.loads(Path("config/moe-ddp-cpu.json").read_text())
    )
    default = PublicMoEDDPPlan(parsed.run, parsed.source)

    assert len(canonical(default.to_mapping()).encode()) == 1181
    assert default.fingerprint() == LEGACY_SHA
    assert default.to_mapping() == parsed.to_mapping()
    assert set(default.to_mapping()) == {"schema", "run", "source"}
    assert default.data_contract() == {
        "schema": "ai-journey-public-byte-window-v1",
        "plan_sha256": LEGACY_SHA,
        "sampler": "rank-strided-epoch-shuffle-v1",
        "tail_policy": "drop_global_incomplete_window",
        "capacity_scope": "rank_microbatch",
        "rank_rng": "cpu_generator_seed_plus_1000_plus_rank",
    }


@pytest.mark.parametrize(
    "change", ["missing", "legacy", "unknown", "bool", "extra", "v1_selector"]
)
def test_versioned_mapping_rejects_implicit_or_conflicting_sampler_selection(change):
    mapping = _mapping()
    if change == "missing":
        del mapping["sampler"]
    elif change == "legacy":
        mapping["sampler"] = "rank-strided-epoch-shuffle-v1"
    elif change == "unknown":
        mapping["sampler"] = "another-shuffle"
    elif change == "bool":
        mapping["sampler"] = True
    elif change == "extra":
        mapping["auto_upgrade"] = True
    else:
        mapping["schema"] = "ai-journey-public-moe-ddp-plan-v1"

    with pytest.raises(ValueError):
        public_moe_ddp_plan_from_mapping(mapping)


def test_v2_reader_roundtrip_has_distinct_plan_and_checkpoint_data_identity(tmp_path):
    plan = public_moe_ddp_plan_from_mapping(_mapping())
    legacy = PublicMoEDDPPlan(plan.run, plan.source)
    config, source = tmp_path / "plan.json", tmp_path / "source.txt"
    config.write_text(canonical(plan.to_mapping()))
    source.write_bytes(_source())

    actual, actual_source = read_public_moe_ddp_plan(config, source)

    assert actual.to_mapping() == plan.to_mapping()
    assert actual_source == _source()
    assert actual.fingerprint() != legacy.fingerprint()
    assert actual.data_contract()["schema"] == "ai-journey-public-byte-window-v2"
    assert actual.data_contract()["sampler"] == "rank-strided-full-epoch-shuffle-v2"
    assert actual.data_contract() != legacy.data_contract()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "world,microbatches,accumulation,samples",
    [(2, 1, 2, 10), (2, 2, 2, 19), (1, 2, 3, 14), (2, 2, 1, 11)],
)
def test_versioned_epochs_match_global_rng_oracle_and_exact_next_byte_windows(
    version, world, microbatches, accumulation, samples
):
    mapping = _mapping(version, samples)
    mapping["run"].update(
        world_size=world,
        micro_batch_size=microbatches,
        accumulation_steps=accumulation,
    )
    plan = public_moe_ddp_plan_from_mapping(mapping)
    source = _source(samples)
    global_batch = world * microbatches * accumulation
    usable = samples - samples % global_batch
    updates = samples // global_batch
    cpu_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    for update in range(plan.run.optimizer_steps):
        epoch, offset = divmod(update, updates)
        order = list(range(samples if version == 2 else usable))
        random.Random(f"53:{epoch}").shuffle(order)
        selected = []
        for rank in range(world):
            local = order[:usable][rank::world]
            begin = offset * microbatches * accumulation
            expected = [
                local[begin + batch * microbatches : begin + (batch + 1) * microbatches]
                for batch in range(accumulation)
            ]
            window, schedule = public_moe_ddp_window(
                plan, source, rank=rank, update_index=update
            )
            assert schedule == {
                "epoch": epoch,
                "update_offset": offset,
                "sample_ids": expected,
            }
            selected.extend(item for batch in schedule["sample_ids"] for item in batch)
            for (inputs, targets, mask), ids in zip(window, expected, strict=True):
                assert inputs.tolist() == [
                    list(source[sample * 4 : sample * 4 + 4]) for sample in ids
                ]
                assert targets.tolist() == [
                    list(source[sample * 4 + 1 : sample * 4 + 5]) for sample in ids
                ]
                assert mask.tolist() == [[True] * 4 for _ in ids]
        assert len(selected) == len(set(selected)) == global_batch
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert python_rng == random.getstate()
    assert source_admission(plan, source)["discarded_samples"] == samples % global_batch


@pytest.mark.parametrize("saved_version", [1, 2])
def test_cross_version_checkpoint_rejects_without_mutation_and_same_version_loads(
    tmp_path, saved_version
):
    saved_mapping = _mapping(saved_version)
    saved_mapping["run"]["world_size"] = 1
    saved_plan = public_moe_ddp_plan_from_mapping(saved_mapping)
    changed_mapping = _mapping(3 - saved_version)
    changed_mapping["run"]["world_size"] = 1
    changed_plan = public_moe_ddp_plan_from_mapping(changed_mapping)
    source = _source()
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        saved = MoEDDPTrainingSession(saved_plan.run)
        window, _ = public_moe_ddp_window(saved_plan, source, rank=0, update_index=0)
        saved.step(window)
        checkpoint = tmp_path / "state.pt"
        save_moe_ddp_checkpoint(
            checkpoint,
            session=saved,
            source=source,
            data_contract=saved_plan.data_contract(),
        )
        expected_rng = torch.get_rng_state().clone()
        target = MoEDDPTrainingSession(changed_plan.run)
        window, _ = public_moe_ddp_window(changed_plan, source, rank=0, update_index=0)
        target.step(window)
        target.wrapped.eval()
        before = (
            copy.deepcopy(target.model.state_dict()),
            copy.deepcopy(target.optimizer.state_dict()),
            torch.get_rng_state().clone(),
            [
                None if p.grad is None else p.grad.clone()
                for p in target.model.parameters()
            ],
            [module.training for module in target.wrapped.modules()],
        )

        with pytest.raises(RuntimeError, match="checkpoint validation"):
            load_moe_ddp_checkpoint(
                checkpoint,
                session=target,
                source=source,
                data_contract=changed_plan.data_contract(),
            )

        assert exact_state(target.model.state_dict(), before[0])
        assert exact_state(target.optimizer.state_dict(), before[1])
        assert torch.equal(torch.get_rng_state(), before[2])
        assert exact_state([p.grad for p in target.model.parameters()], before[3])
        assert [module.training for module in target.wrapped.modules()] == before[4]
        assert not target.poisoned
        assert (
            load_moe_ddp_checkpoint(
                checkpoint,
                session=target,
                source=source,
                data_contract=saved_plan.data_contract(),
            )
            == 1
        )
        assert exact_state(target.model.state_dict(), saved.model.state_dict())
        assert exact_state(target.optimizer.state_dict(), saved.optimizer.state_dict())
        assert torch.equal(torch.get_rng_state(), expected_rng)


def test_actual_v2_two_rank_dropout_restart_across_epoch_is_exact(tmp_path, capsys):
    mapping = _mapping()
    plan = public_moe_ddp_plan_from_mapping(mapping)
    config, source = tmp_path / "plan.json", tmp_path / "source.txt"
    config.write_text(canonical(plan.to_mapping()))
    source.write_bytes(_source())
    for phase, steps, resume in (
        ("continuous", 4, False),
        ("initial", 2, False),
        ("resumed", 2, True),
    ):
        checkpoint = tmp_path / (
            "continuous.pt" if phase == "continuous" else "recovered.pt"
        )
        arguments = [
            "train",
            "--config",
            str(config),
            "--source",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--workdir",
            str(tmp_path / phase),
            "--output",
            str(tmp_path / f"{phase}.json"),
            "--steps",
            str(steps),
        ]
        assert main(arguments + (["--resume"] if resume else [])) == 0
        capsys.readouterr()
        assert (
            main(
                [
                    "verify",
                    "--report",
                    str(tmp_path / f"{phase}.json"),
                    "--checkpoint",
                    str(checkpoint),
                ]
            )
            == 0
        )
        capsys.readouterr()
    continuous, initial, resumed = (
        read_moe_ddp_evidence(tmp_path / f"{phase}.json")
        for phase in ("continuous", "initial", "resumed")
    )
    assert initial["completed_step"] == resumed["started_step"] == 2
    assert initial["metrics"] + resumed["metrics"] == continuous["metrics"]
    for rank in range(2):
        receipts = [
            report["workers"][rank] for report in (continuous, initial, resumed)
        ]
        assert len({receipt["pid"] for receipt in receipts}) == 3
        assert receipts[1]["metrics"] + receipts[2]["metrics"] == receipts[0]["metrics"]
        assert receipts[2]["metrics"][0]["data_schedule"]["epoch"] == 1
    expected = torch.load(tmp_path / "continuous.pt", weights_only=True)
    actual = torch.load(tmp_path / "recovered.pt", weights_only=True)
    assert exact_state(actual, expected)
    assert actual["data_contract"] == plan.data_contract()
    assert actual["step"] == 4
    assert all(
        dropout > 0
        for dropout in (
            plan.run.model.gpt2.embedding_dropout,
            plan.run.model.gpt2.attention_dropout,
            plan.run.model.gpt2.residual_dropout,
        )
    )
