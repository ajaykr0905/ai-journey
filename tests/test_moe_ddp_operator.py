"""Actual public-byte subprocess restart, strict receipts and bounded operators."""

import copy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess

import pytest
import torch

from ai_journey.ddp_acceptance import exact_state
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_training import GPT2TrainingSource
from ai_journey.moe_checkpoint import _digest
from ai_journey.moe_config import MoEConfig
from ai_journey.moe_ddp_cli import main
from ai_journey.moe_ddp_config import MoEDDPConfig
from ai_journey.moe_ddp_data import public_moe_ddp_schedule, public_moe_ddp_window
from ai_journey.moe_ddp_evidence import (
    build_moe_ddp_evidence,
    read_moe_ddp_evidence,
    validate_moe_ddp_evidence,
    verify_moe_ddp_archive,
)
from ai_journey.moe_ddp_launch import launch_moe_ddp
from ai_journey.moe_ddp_plan import (
    PublicMoEDDPPlan,
    canonical,
    public_moe_ddp_plan_from_mapping,
    read_public_moe_ddp_plan,
    source_admission,
)


def public_plan(world_size=2):
    source = GPT2TrainingSource(
        **json.loads(Path("config/ddp-lab.json").read_text())["source"]
    )
    return PublicMoEDDPPlan(
        MoEDDPConfig(
            MoEConfig(
                GPT2Config(
                    256,
                    4,
                    4,
                    2,
                    1,
                    embedding_dropout=0.1,
                    attention_dropout=0.1,
                    residual_dropout=0.1,
                )
            ),
            world_size=world_size,
        ),
        source,
    )


def inputs(directory, world_size=2):
    directory.mkdir(exist_ok=True)
    config, source = directory / "plan.json", directory / "source.txt"
    plan = public_plan(world_size)
    config.write_text(canonical(plan.to_mapping()))
    source.write_bytes(Path("data/day-19-demo-names.txt").read_bytes())
    return plan, config, source


def seal(report):
    report["evidence_sha256"] = sha256(
        canonical(
            {key: value for key, value in report.items() if key != "evidence_sha256"}
        ).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    directory = tmp_path_factory.mktemp("moe-public-observed")
    plan, config, source = inputs(directory, world_size=1)
    checkpoint = directory / "state.pt"
    workers = launch_moe_ddp(
        config=config,
        source=source,
        checkpoint=checkpoint,
        workdir=directory / "workers",
        steps=1,
    )
    report = build_moe_ddp_evidence(plan, source.read_bytes(), workers)
    return directory, plan, config, source, checkpoint, report


def test_actual_two_rank_public_cli_fresh_restart_is_exact(tmp_path, capsys):
    plan, config, source = inputs(tmp_path)
    for phase, steps, resume in (
        ("continuous", 4, False),
        ("first", 2, False),
        ("resume", 2, True),
    ):
        checkpoint = tmp_path / (
            "restart.pt" if phase != "continuous" else "continuous.pt"
        )
        report = tmp_path / f"{phase}.json"
        args = [
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
            str(report),
            "--steps",
            str(steps),
        ]
        assert main(args + (["--resume"] if resume else [])) == 0
        assert json.loads(capsys.readouterr().out)["completed_step"] == (
            2 if phase == "first" else 4
        )
        assert (
            main(["verify", "--report", str(report), "--checkpoint", str(checkpoint)])
            == 0
        )
        assert (
            json.loads(capsys.readouterr().out)["checkpoint"]["scope"]
            == "single_cpu_typed_restore_not_training_replay"
        )
    reports = [
        read_moe_ddp_evidence(tmp_path / f"{phase}.json")
        for phase in ("continuous", "first", "resume")
    ]
    continuous, first, resumed = reports
    assert first["started_step"] == 0 and first["completed_step"] == 2
    assert resumed["started_step"] == 2 and resumed["completed_step"] == 4
    for rank in range(2):
        assert (
            first["workers"][rank]["metrics"] + resumed["workers"][rank]["metrics"]
            == continuous["workers"][rank]["metrics"]
        )
        assert len({report["workers"][rank]["pid"] for report in reports}) == 3
        assert (tmp_path / "resume" / f"rank-{rank}.log").is_file()
    assert first["metrics"] + resumed["metrics"] == continuous["metrics"]
    baseline = torch.load(tmp_path / "continuous.pt", weights_only=True)
    actual = torch.load(tmp_path / "restart.pt", weights_only=True)
    for key in ("model_state", "optimizer_state"):
        assert exact_state(baseline["inner"][key], actual["inner"][key])
    for key in ("rank_rngs", "data_cursor", "step"):
        assert exact_state(baseline[key], actual[key])
    assert actual["step"] == plan.run.optimizer_steps


def test_read_only_plan_reports_actual_source_without_execution(
    tmp_path, capsys, monkeypatch
):
    plan, config, source = inputs(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("plan must not construct a model or spawn")

    monkeypatch.setattr("ai_journey.moe_ddp_cli.launch_moe_ddp", forbidden)
    assert main(["plan", "--config", str(config), "--source", str(source)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["execution"] == "not_executed"
    assert result["plan_sha256"] == plan.fingerprint()
    assert result["source"] == source_admission(plan, source.read_bytes())
    source.write_bytes(b"unrelated bytes")
    with pytest.raises(SystemExit):
        main(["plan", "--config", str(config), "--source", str(source)])


def test_sampler_has_exact_rank_strided_epoch_oracle_and_only_selected_tensors(
    monkeypatch,
):
    base = public_plan()
    source = bytes(range(41))
    plan = replace(
        base,
        run=replace(base.run, optimizer_steps=4),
        source=replace(base.source, sha256=sha256(source).hexdigest()),
    )
    cpu_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    allocated = []
    tensor = torch.tensor

    def bounded(value, *args, **kwargs):
        allocated.append(sum(len(row) for row in value))
        return tensor(value, *args, **kwargs)

    monkeypatch.setattr("ai_journey.moe_ddp_data.torch.tensor", bounded)
    for update in range(4):
        epoch, offset = divmod(update, 2)
        order = list(range(8))
        random.Random(f"{plan.run.seed}:{epoch}").shuffle(order)
        selected = []
        for rank in range(2):
            window, schedule = public_moe_ddp_window(
                plan, source, rank=rank, update_index=update
            )
            assert schedule == {
                "epoch": epoch,
                "update_offset": offset,
                "sample_ids": [
                    [order[rank + (offset * 2 + batch) * 2]] for batch in range(2)
                ],
            }
            selected.extend(item for batch in schedule["sample_ids"] for item in batch)
            for (x, y, mask), ids in zip(window, schedule["sample_ids"], strict=True):
                assert x.tolist() == [list(source[ids[0] * 4 : ids[0] * 4 + 4])]
                assert y.tolist() == [list(source[ids[0] * 4 + 1 : ids[0] * 4 + 5])]
                assert bool(mask.all()) and x.shape == (1, 4)
        assert len(set(selected)) == 4
    assert allocated and max(allocated) == 4
    assert (
        torch.equal(cpu_rng, torch.get_rng_state()) and python_rng == random.getstate()
    )
    assert source_admission(plan, source)["discarded_samples"] == 2


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "source_unknown",
        "url",
        "url_secret",
        "sha",
        "bool_steps",
        "huge_steps",
        "huge_beta",
        "unknown",
    ],
)
def test_plan_rejects_untrusted_mappings(change):
    value = public_plan().to_mapping()
    if change == "schema":
        value["schema"] = "other"
    elif change == "source_unknown":
        value["source"]["private"] = "no"
    elif change == "url":
        value["source"]["url"] = "file:///private"
    elif change == "url_secret":
        value["source"]["url"] = "https://example.test/data?token=not-real"
    elif change == "sha":
        value["source"]["sha256"] = False
    elif change == "bool_steps":
        value["run"]["optimizer_steps"] = True
    elif change == "huge_steps":
        value["run"]["optimizer_steps"] = 10**1000
    elif change == "huge_beta":
        value["run"]["optimizer"]["beta1"] = 10**1000
    else:
        value["unknown"] = 1
    with pytest.raises(ValueError):
        public_moe_ddp_plan_from_mapping(value)


def test_duplicate_json_and_source_size_guards_precede_materialization(tmp_path):
    _, config, source = inputs(tmp_path)
    config.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        read_public_moe_ddp_plan(config, source)
    plan = public_plan()
    for byte_count in (True, 2, 1024 * 1024 + 1, 10**1000):
        with pytest.raises(ValueError):
            public_moe_ddp_schedule(plan, byte_count, rank=0, update_index=0)
    for rank in (-1, True, 2):
        with pytest.raises(ValueError):
            public_moe_ddp_schedule(plan, 100, rank=rank, update_index=0)


@pytest.mark.parametrize(
    "change",
    [
        "objective",
        "routes",
        "capacity",
        "schedule",
        "runtime",
        "pid",
        "step",
        "source",
        "balance",
        "clip",
    ],
)
def test_resealed_reports_reject_semantic_changes(observed, change):
    report = copy.deepcopy(observed[-1])
    metric = report["workers"][0]["metrics"][0]
    if change == "objective":
        metric["objective"] += 0.1
    elif change == "routes":
        metric["routing"][0][0]["accepted_routes"][0][0] = not metric["routing"][0][0][
            "accepted_routes"
        ][0][0]
    elif change == "capacity":
        metric["routing"][0][0]["capacity_applied"] = False
    elif change == "schedule":
        metric["data_schedule"]["sample_ids"][0][0] += 1
    elif change == "runtime":
        report["runtime"]["threads"] = True
    elif change == "pid":
        report["workers"][0]["pid"] = False
    elif change == "step":
        metric["step"] = True
    elif change == "source":
        report["source_sha256"] = "0" * 64
    elif change == "balance":
        metric["balance_loss"] += 0.1
    else:
        metric["gradient_norm_after_clip"] = 100
    seal(report)
    with pytest.raises(ValueError):
        validate_moe_ddp_evidence(report)


def test_report_archive_binding_and_local_restore_preserve_caller_rng(
    observed, tmp_path
):
    report = observed[-1]
    torch.manual_seed(4321)
    before = torch.get_rng_state().clone()
    result = verify_moe_ddp_archive(report, observed[-2])
    assert result["step"] == 1 and torch.equal(before, torch.get_rng_state())
    payload = torch.load(observed[-2], weights_only=True)
    payload["inner"]["optimizer_state"]["param_groups"][0]["weight_decay"] = 0.5
    payload["inner"]["checkpoint_sha256"] = _digest(
        {k: v for k, v in payload["inner"].items() if k != "checkpoint_sha256"}
    )
    payload["checkpoint_sha256"] = _digest(
        {k: v for k, v in payload.items() if k != "checkpoint_sha256"}
    )
    path = tmp_path / "changed.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        verify_moe_ddp_archive(report, path)
    changed = copy.deepcopy(report)
    identity = {
        "payload_sha256": payload["checkpoint_sha256"],
        "archive_sha256": sha256(path.read_bytes()).hexdigest(),
    }
    changed["checkpoint_identity"] = identity
    changed["workers"][0]["checkpoint_identity"] = identity
    seal(changed)
    with pytest.raises(ValueError):
        verify_moe_ddp_archive(changed, path)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "change",
    ["source_alias", "symlink", "existing", "inside", "output_exists", "output_worker"],
)
def test_operating_paths_reject_before_spawning(tmp_path, monkeypatch, change):
    _, config, source = inputs(tmp_path)
    checkpoint, workdir, output = (
        tmp_path / "state.pt",
        tmp_path / "workers",
        tmp_path / "report.json",
    )
    if change == "source_alias":
        checkpoint = source
    elif change == "symlink":
        checkpoint.symlink_to(source)
    elif change == "existing":
        checkpoint.write_text("preserved")
    elif change == "inside":
        workdir = tmp_path
    elif change == "output_exists":
        output.write_text("preserved")
    else:
        output = workdir / "rank-0.json"

    def forbidden(*args, **kwargs):
        pytest.fail("invalid paths must not spawn")

    monkeypatch.setattr("ai_journey.moe_ddp_launch.subprocess.Popen", forbidden)
    with pytest.raises(SystemExit):
        main(
            [
                "train",
                "--config",
                str(config),
                "--source",
                str(source),
                "--checkpoint",
                str(checkpoint),
                "--workdir",
                str(workdir),
                "--output",
                str(output),
                "--steps",
                "1",
            ]
        )
    assert source.read_bytes() == Path("data/day-19-demo-names.txt").read_bytes()


def test_later_spawn_failure_reaps_only_owned_rank_and_closes_logs(
    tmp_path, monkeypatch
):
    _, config, source = inputs(tmp_path)
    calls, logs = [], []

    class Process:
        status = None
        terminated = False
        waited = False

        def poll(self):
            return self.status

        def terminate(self):
            self.terminated = True
            self.status = -15

        def wait(self, timeout):
            self.waited = True
            return self.status

        def kill(self):
            pytest.fail("terminated child should have exited")

    process = Process()

    def spawn(command, **kwargs):
        calls.append(command)
        logs.append(kwargs["stdout"])
        assert "start_new_session" not in kwargs
        if len(calls) == 2:
            raise OSError("authored later-spawn failure")
        return process

    monkeypatch.setattr("ai_journey.moe_ddp_launch.subprocess.Popen", spawn)
    with pytest.raises(OSError, match="later-spawn"):
        launch_moe_ddp(
            config=config,
            source=source,
            checkpoint=tmp_path / "state.pt",
            workdir=tmp_path / "workers",
            steps=1,
        )
    assert process.terminated and process.waited and all(log.closed for log in logs)


def test_deadline_kills_and_reaps_stubborn_owned_rank(tmp_path, monkeypatch):
    _, config, source = inputs(tmp_path, world_size=1)

    class Process:
        status = None
        terminated = False
        killed = False

        def poll(self):
            return self.status

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired("authored worker", timeout)
            return self.status

        def kill(self):
            self.killed = True
            self.status = -9

    process = Process()
    monkeypatch.setattr(
        "ai_journey.moe_ddp_launch.subprocess.Popen", lambda *args, **kwargs: process
    )
    clocks = iter((0.0, 2.0))
    monkeypatch.setattr(
        "ai_journey.moe_ddp_launch.time.monotonic", lambda: next(clocks)
    )
    with pytest.raises(TimeoutError):
        launch_moe_ddp(
            config=config,
            source=source,
            checkpoint=tmp_path / "state.pt",
            workdir=tmp_path / "workers",
            steps=1,
            timeout_seconds=1,
        )
    assert process.terminated and process.killed
