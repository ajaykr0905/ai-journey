"""Supervised workers consume the exact public inputs admitted by their caller."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import subprocess

import pytest

from ai_journey.moe_ddp_evidence import (
    build_moe_ddp_evidence,
    verify_moe_ddp_archive,
)
from ai_journey.moe_ddp_launch import launch_moe_ddp
from ai_journey.moe_ddp_plan import canonical
from test_moe_ddp_operator import inputs


def test_workers_keep_admitted_inputs_when_originals_change_at_first_spawn(
    tmp_path, monkeypatch
):
    plan, config, source = inputs(tmp_path)
    admitted_source = source.read_bytes()
    replacement_source = b"replacement public training bytes\n" * 4
    replacement = replace(
        plan,
        run=replace(plan.run, seed=plan.run.seed + 1),
        source=replace(plan.source, sha256=sha256(replacement_source).hexdigest()),
    )
    spawn = subprocess.Popen
    supplied_inputs = []

    def replace_originals(command, **kwargs):
        if not supplied_inputs:
            config.write_text(canonical(replacement.to_mapping()))
            source.write_bytes(replacement_source)
        supplied_inputs.append(
            tuple(
                Path(command[command.index(flag) + 1])
                for flag in ("--config", "--source")
            )
        )
        return spawn(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", replace_originals)
    checkpoint = tmp_path / "state.pt"
    workers = launch_moe_ddp(
        config=config,
        source=source,
        checkpoint=checkpoint,
        workdir=tmp_path / "workers",
        steps=1,
    )

    report = build_moe_ddp_evidence(plan, admitted_source, workers)
    assert verify_moe_ddp_archive(report, checkpoint)["step"] == 1
    assert len(workers) == 2
    assert config.read_text() == canonical(replacement.to_mapping())
    assert source.read_bytes() == replacement_source
    assert all(not path.exists() for pair in supplied_inputs for path in pair)


def test_changed_plan_rejects_before_spawning_or_creating_outputs(
    tmp_path, monkeypatch
):
    admitted, config, source = inputs(tmp_path)
    config.write_text(
        canonical(replace(admitted, run=replace(admitted.run, seed=99)).to_mapping())
    )

    def forbidden(*args, **kwargs):
        pytest.fail("a plan that differs from caller admission must not spawn")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    checkpoint, workdir = tmp_path / "state.pt", tmp_path / "workers"
    with pytest.raises(ValueError, match="identity"):
        launch_moe_ddp(
            config=config,
            source=source,
            checkpoint=checkpoint,
            workdir=workdir,
            steps=1,
            expected_plan_sha256=admitted.fingerprint(),
        )
    assert not checkpoint.exists() and not workdir.exists()


def test_failed_spawn_keeps_input_snapshots_until_child_reaping(tmp_path, monkeypatch):
    _, config, source = inputs(tmp_path)
    original = (config.read_bytes(), source.read_bytes())
    supplied_inputs = []

    class OwnedChild:
        status = None

        def poll(self):
            return self.status

        def terminate(self):
            self.status = -15

        def wait(self, timeout):
            assert all(path.is_file() for pair in supplied_inputs for path in pair)
            return self.status

    def fail_second_spawn(command, **kwargs):
        supplied_inputs.append(
            tuple(
                Path(command[command.index(flag) + 1])
                for flag in ("--config", "--source")
            )
        )
        if len(supplied_inputs) == 2:
            raise OSError("authored second-spawn failure")
        return OwnedChild()

    monkeypatch.setattr(subprocess, "Popen", fail_second_spawn)
    with pytest.raises(OSError, match="second-spawn"):
        launch_moe_ddp(
            config=config,
            source=source,
            checkpoint=tmp_path / "state.pt",
            workdir=tmp_path / "workers",
            steps=1,
        )
    assert (config.read_bytes(), source.read_bytes()) == original
    assert all(not path.exists() for pair in supplied_inputs for path in pair)
