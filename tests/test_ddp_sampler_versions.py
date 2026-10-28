"""Explicit dense sampler versions preserve replay and admit the full population."""

from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_data import SAMPLER_V1, byte_batches, data_contract

import copy
from hashlib import sha256
import json
from pathlib import Path
import random

import pytest
import torch

from ai_journey.ddp_acceptance import exact_state
from ai_journey.ddp_acceptance import certify_crash_recovery, verify_crash_recovery
from ai_journey.ddp_checkpoint import load_checkpoint, save_checkpoint
from ai_journey.ddp_runtime import process_group
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_training import GPT2TrainingConfig, build_gpt2_optimizer
from ai_journey.training_models import build_training_model
from ai_journey.ddp_launch import launch_training
from ai_journey.ddp_evidence import build_evidence, verify_evidence
from ai_journey.gpt2_training import load_training_spec
from ai_journey.ddp_pipeline import load_serving_model
from ai_journey.ddp_cli import main
from ai_journey.ddp_comparison import verify_comparison
from ai_journey.ddp_evidence import digest


V2 = "rank-strided-full-epoch-shuffle-v2"


def test_opt_in_sampler_shuffles_all_samples_before_dropping_an_update_tail():
    # Worked Python-RNG oracle: Random("48:0") shuffles ten sample IDs to
    # [0, 6, 8, 7, 2, 3, 1, 9, 5, 4]. Only the final incomplete group is dropped.
    ranks = []
    for rank in range(2):
        batches, plan = byte_batches(
            bytes(range(41)),
            sequence_length=4,
            topology=Topology(rank, 2),
            micro_batch_size=1,
            accumulation_steps=2,
            seed=48,
            sampler="rank-strided-full-epoch-shuffle-v2",
        )
        ranks.append(tuple(int(inputs[0, 0]) // 4 for inputs, _ in batches))
        assert plan.updates_per_epoch == 2
        assert plan.discarded_samples == 2
    assert ranks == [(0, 8, 2, 1), (6, 7, 3, 9)]


@pytest.mark.parametrize("sampler", [SAMPLER_V1, V2])
def test_checkpoint_version_binds_sampler_before_mutating_caller_state(
    tmp_path, sampler
):
    config = GPT2TrainingConfig(
        optimizer_steps=4,
        micro_batch_size=1,
        accumulation_steps=2,
        sequence_length=4,
        warmup_steps=1,
        max_learning_rate=0.003,
        min_learning_rate=0.0003,
        weight_decay=0.1,
        gradient_clip=1.0,
        seed=48,
    )
    model = build_training_model(GPT2Config(256, 4, 4, 1, 1)).double()
    optimizer = build_gpt2_optimizer(model, config)
    source = bytes(range(41))
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        path = tmp_path / "v2.pt"
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=source,
            step=0,
            sampler=sampler,
        )
        payload = torch.load(path, weights_only=True)
        assert payload["schema_version"] == (2 if sampler == V2 else 1)
        if sampler == V2:
            assert payload["data_contract"]["sampler"] == V2
            assert payload["data_contract"]["source_bytes"] == 41
        else:
            assert "data_contract" not in payload
        before = (
            copy.deepcopy(model.state_dict()),
            copy.deepcopy(optimizer.state_dict()),
            torch.get_rng_state().clone(),
        )
        with pytest.raises(RuntimeError, match="checkpoint validation"):
            load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                config=config,
                source=source,
                sampler=SAMPLER_V1 if sampler == V2 else V2,
            )
        assert exact_state(
            before, (model.state_dict(), optimizer.state_dict(), torch.get_rng_state())
        )
        assert (
            load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                config=config,
                source=source,
                sampler=sampler,
            )
            == 0
        )


def test_default_schedule_retains_the_legacy_worked_order_and_global_rng():
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    ranks = []
    for rank in range(2):
        options = dict(
            sequence_length=4,
            topology=Topology(rank, 2),
            micro_batch_size=1,
            accumulation_steps=2,
            seed=48,
        )
        default, _ = byte_batches(bytes(range(41)), **options)
        explicit, _ = byte_batches(bytes(range(41)), **options, sampler=SAMPLER_V1)
        assert exact_state(default, explicit)
        ranks.append(tuple(int(inputs[0, 0]) // 4 for inputs, _ in default))
    assert ranks == [(0, 3, 6, 5), (7, 1, 2, 4)]
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize(
    "world,micro,accumulation", [(1, 2, 2), (2, 1, 2), (2, 2, 1), (3, 1, 1)]
)
def test_full_epoch_rank_batches_match_an_independent_rng_partition(
    world, micro, accumulation
):
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    eligible = set()
    for epoch in range(5):
        order = list(range(10))
        random.Random(f"48:{epoch}").shuffle(order)
        usable = 10 // (world * micro * accumulation) * (world * micro * accumulation)
        observed = []
        for rank in range(world):
            batches, plan = byte_batches(
                bytes(range(41)),
                sequence_length=4,
                topology=Topology(rank, world),
                micro_batch_size=micro,
                accumulation_steps=accumulation,
                seed=48,
                epoch=epoch,
                sampler=V2,
            )
            ids = [int(row[0]) // 4 for inputs, _ in batches for row in inputs]
            assert ids == order[:usable][rank::world]
            assert len(batches) == plan.updates_per_epoch * accumulation
            assert all(
                inputs.shape == targets.shape == (micro, 4)
                for inputs, targets in batches
            )
            assert all(
                torch.equal(inputs[:, 1:], targets[:, :-1])
                for inputs, targets in batches
            )
            observed.extend(ids)
        assert len(set(observed)) == usable
        eligible.update(observed)
    assert eligible == set(range(10))
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize(
    "sampler", [None, True, 2, "v2", "rank-strided-full-epoch-shuffle-v3"]
)
def test_unknown_sampler_rejects_at_the_public_data_interface(sampler):
    with pytest.raises(ValueError, match="sampler"):
        byte_batches(
            bytes(range(41)),
            sequence_length=4,
            topology=Topology(0, 2),
            micro_batch_size=1,
            accumulation_steps=2,
            seed=48,
            sampler=sampler,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_size", True),
        ("world_size", 1.0),
        ("source_bytes", 41.0),
        ("sequence_length", 4.0),
        ("micro_batch_size", True),
        ("accumulation_steps", 2.0),
        ("seed", 48.0),
        ("extra", 1),
    ],
)
def test_checkpoint_rejects_noncanonical_data_contract_without_mutation(
    tmp_path, field, value
):
    source, spec_path = public_files(tmp_path)
    spec = load_training_spec(spec_path)
    model = build_training_model(spec.model).double()
    optimizer = build_gpt2_optimizer(model, spec.training)
    path = tmp_path / "checkpoint.pt"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=spec.training,
            source=source.read_bytes(),
            step=0,
            sampler=V2,
        )
        payload = torch.load(path, weights_only=True)
        assert payload["data_contract"] == data_contract(
            source.read_bytes(), spec.training, 1
        )
        payload["data_contract"][field] = value
        torch.save(payload, path)
        before = (
            copy.deepcopy(model.state_dict()),
            copy.deepcopy(optimizer.state_dict()),
            torch.get_rng_state().clone(),
        )
        with pytest.raises(RuntimeError, match="checkpoint validation"):
            load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                config=spec.training,
                source=source.read_bytes(),
                sampler=V2,
            )
        assert exact_state(
            before, (model.state_dict(), optimizer.state_dict(), torch.get_rng_state())
        )


def public_files(directory, *, dropout=0.0):
    source = directory / "public.bin"
    source.write_bytes(bytes(range(41)))
    payload = json.loads(
        (Path(__file__).resolve().parents[1] / "config/gpt2-training.json").read_bytes()
    )
    payload["model"].update(
        context_length=4,
        embedding_dim=4,
        head_count=1,
        embedding_dropout=dropout,
        attention_dropout=dropout,
        residual_dropout=dropout,
    )
    payload["training"].update(
        optimizer_steps=4,
        micro_batch_size=1,
        accumulation_steps=2,
        sequence_length=4,
        seed=48,
        warmup_steps=1,
    )
    payload["source"].update(
        name="Authored public byte IDs",
        url="https://github.com/ajaykr0905/ai-journey",
        sha256=sha256(source.read_bytes()).hexdigest(),
        revision="authored-byte-fixture-v1",
    )
    spec = directory / "spec.json"
    spec.write_text(json.dumps(payload))
    return source, spec


def test_actual_v2_two_rank_operator_restart_across_an_epoch_is_exact(tmp_path):
    source, spec = public_files(tmp_path)
    common = dict(source=source, spec=spec, world_size=2, sampler=V2)
    continuous = launch_training(
        **common,
        checkpoint=tmp_path / "continuous.pt",
        output_dir=tmp_path / "continuous",
        steps=4,
    )
    first = launch_training(
        **common,
        checkpoint=tmp_path / "resumed.pt",
        output_dir=tmp_path / "first",
        steps=2,
    )
    resumed = launch_training(
        **common,
        checkpoint=tmp_path / "resumed.pt",
        output_dir=tmp_path / "resumed",
        steps=2,
        resume=True,
    )
    assert not {rank["pid"] for rank in first} & {rank["pid"] for rank in resumed}
    assert all(
        first[rank]["metrics"] + resumed[rank]["metrics"] == continuous[rank]["metrics"]
        for rank in range(2)
    )
    assert exact_state(
        torch.load(tmp_path / "continuous.pt", weights_only=True),
        torch.load(tmp_path / "resumed.pt", weights_only=True),
    )
    report = build_evidence(
        source.read_bytes(), load_training_spec(spec), resumed, sampler=V2
    )
    assert report["schema_version"] == 2
    assert report["data_contract"] == resumed[0]["data_contract"]
    assert verify_evidence(report) == report
    with pytest.raises(ValueError, match="schema"):
        load_serving_model(
            tmp_path / "resumed.pt", load_training_spec(spec), source.read_bytes()
        )
    serving = load_serving_model(
        tmp_path / "resumed.pt",
        load_training_spec(spec),
        source.read_bytes(),
        sampler=V2,
    )
    assert not any(parameter.requires_grad for parameter in serving.parameters())
    assert exact_state(
        serving.state_dict(),
        torch.load(tmp_path / "continuous.pt", weights_only=True)["model"],
    )


def test_actual_v2_crash_recovery_binds_every_retained_archive(tmp_path):
    source, spec = public_files(tmp_path)
    directory = tmp_path / "crash"
    report = certify_crash_recovery(
        source=source,
        spec=spec,
        directory=directory,
        sampler=V2,
    )
    assert report["schema"] == "ai-journey-ddp-crash-recovery-v2"
    assert report["injection"]["data_contract"] == report["data_contract"]
    assert report["metrics"]["lost_optimizer_updates"] == 1
    assert verify_crash_recovery(report, directory=directory) == report
    for filename in ("continuous.pt", "recovered.pt", "last-durable.pt"):
        payload = torch.load(directory / filename, weights_only=True)
        assert payload["schema_version"] == 2
        assert payload["data_contract"] == report["data_contract"]
    # A byte-valid final archive cannot silently become another sampler run.
    checkpoint = directory / "recovered.pt"
    payload = torch.load(checkpoint, weights_only=True)
    payload["data_contract"]["seed"] += 1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="artifacts"):
        verify_crash_recovery(report, directory=directory)


def test_cli_carries_v2_through_train_resume_verify_benchmark_and_serve(
    tmp_path, capsys
):
    source, spec = public_files(tmp_path)
    checkpoint = tmp_path / "state.pt"
    base = [
        "--source",
        str(source),
        "--spec",
        str(spec),
        "--checkpoint",
        str(checkpoint),
        "--sampler",
        V2,
    ]
    for phase, extra in (("first", []), ("resumed", ["--resume"])):
        assert (
            main(
                [
                    "train",
                    *base,
                    "--output",
                    str(tmp_path / f"{phase}.json"),
                    "--workdir",
                    str(tmp_path / phase),
                    "--steps",
                    "2",
                    *extra,
                ]
            )
            == 0
        )
        report = json.loads(capsys.readouterr().out)
        assert report["schema_version"] == 2
        assert report["data_contract"]["sampler"] == V2
    assert main(["verify", "--report", str(tmp_path / "resumed.json")]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == 2
    assert (
        main(
            [
                "benchmark",
                *base,
                "--output",
                str(tmp_path / "benchmark.json"),
                "--repeats",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["output_parity"]
    assert main(["serve", *base, "--port", "0", "--max-seconds", "1"]) == 0
    observations = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert observations[0]["listening"].startswith("http://127.0.0.1:")
    assert observations[0]["model_sha256"] == report["model_sha256"]


@pytest.mark.parametrize("operation", ["matrix", "pipeline"])
def test_v2_model_data_operations_report_the_selected_contract(
    tmp_path, capsys, operation
):
    root = Path(__file__).resolve().parents[1]
    report = tmp_path / "report.json"
    work = tmp_path / "work"
    assert (
        main(
            [
                operation,
                "--source",
                str(root / "data/day-19-demo-names.txt"),
                "--spec",
                str(root / "config/ddp-lab.json"),
                "--output",
                str(report),
                "--workdir",
                str(work),
                "--steps",
                "2",
                "--sampler",
                V2,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == 2
    assert json.loads(report.read_bytes()) == result
    if operation == "matrix":
        assert result["sampler"] == V2
        assert all(arm["evidence"]["schema_version"] == 2 for arm in result["arms"])
        assert main(["verify-comparison", "--report", str(report)]) == 0
        capsys.readouterr()
        mixed = copy.deepcopy(result)
        evidence = mixed["arms"][0]["evidence"]
        evidence.pop("data_contract")
        evidence["schema_version"] = 1
        evidence["evidence_sha256"] = digest(
            {key: value for key, value in evidence.items() if key != "evidence_sha256"}
        )
        mixed["evidence_sha256"] = digest(
            {key: value for key, value in mixed.items() if key != "evidence_sha256"}
        )
        with pytest.raises(ValueError, match="mix sampler"):
            verify_comparison(mixed)
    else:
        assert result["data_contract"]["source_bytes"] == len(
            (work / "train.txt").read_bytes()
        )
        assert result["data_contract"] == result["distributed"]["data_contract"]
        assert result["serving_benchmark"]["output_parity"]
        assert not set(result["training_documents"]) & set(
            result["validation_documents"]
        )


def test_v1_crash_certificate_cannot_reinterpret_a_v2_final_archive(tmp_path):
    source, spec = public_files(tmp_path)
    directory = tmp_path / "legacy-crash"
    report = certify_crash_recovery(source=source, spec=spec, directory=directory)
    assert report["schema"] == "ai-journey-ddp-crash-recovery-v1"
    for filename in ("continuous.pt", "recovered.pt"):
        path = directory / filename
        original = path.read_bytes()
        payload = torch.load(path, weights_only=True)
        payload["schema_version"] = 2
        payload["data_contract"] = data_contract(
            source.read_bytes(), load_training_spec(spec).training, 2
        )
        torch.save(payload, path)
        try:
            with pytest.raises(ValueError, match="artifacts"):
                verify_crash_recovery(report, directory=directory)
        finally:
            path.write_bytes(original)
    assert verify_crash_recovery(report, directory=directory) == report
