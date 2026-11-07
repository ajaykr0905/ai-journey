import json
from hashlib import sha256

import pytest

from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_training import (
    GPT2TrainingConfig,
    GPT2TrainingSource,
    GPT2TrainingSpec,
)
from ai_journey.project1_contract import Project1RunContract
from ai_journey.project1_journal_io import load_event_chain
from ai_journey.project1_operator import (
    execute_fresh_run,
    finalize_run_bundle,
    initialize_run,
    verify_run_bundle,
)
from ai_journey.project1_state import reduce_run_state

SOURCE = b"public project one training text\n" * 4


def prepare_inputs(tmp_path):
    source_path = tmp_path / "source.txt"
    source_path.write_bytes(SOURCE)
    spec = GPT2TrainingSpec(
        model=GPT2Config(
            vocab_size=256,
            context_length=8,
            embedding_dim=16,
            head_count=2,
            layer_count=1,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
        ),
        training=GPT2TrainingConfig(
            optimizer_steps=2,
            micro_batch_size=2,
            accumulation_steps=2,
            sequence_length=8,
            max_learning_rate=3e-3,
            min_learning_rate=3e-4,
            warmup_steps=0,
            weight_decay=0.1,
            gradient_clip=1.0,
            seed=64,
        ),
        source=GPT2TrainingSource(
            name="Public test fixture",
            url="https://github.com/ajaykr0905/ai-journey",
            revision="fixture-v1",
            license="CC0-1.0",
            sha256=sha256(SOURCE).hexdigest(),
        ),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_mapping()))
    contract = Project1RunContract(
        run_name="day-64-project1",
        spec_sha256=spec.fingerprint(),
        source_sha256=spec.source.sha256,
        max_source_bytes=1024,
        max_parameters=100_000,
        telemetry_interval_steps=1,
        checkpoint_interval_steps=1,
    )
    contract_path = tmp_path / "contract-input.json"
    contract_path.write_text(json.dumps(contract.to_mapping()))
    return source_path, spec_path, contract_path


def test_initialize_run_atomically_persists_admission_and_started_event(tmp_path):
    source, spec, contract = prepare_inputs(tmp_path)
    run_dir = tmp_path / "run"

    result = initialize_run(
        source_path=source,
        spec_path=spec,
        contract_path=contract,
        run_dir=run_dir,
        recorded_at_utc="2026-11-07T07:30:00+00:00",
    )

    assert result["run_id"].startswith("day-64-project1-")
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "contract.json",
        "journal.json",
        "preflight.json",
    ]
    started = load_event_chain(run_dir / "journal.json")
    assert started[0].attempt_id == result["attempt_id"]
    assert started[0].payload["run_id"] == result["run_id"]


def test_initialize_run_rejects_existing_destination_without_changes(tmp_path):
    source, spec, contract = prepare_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / "owned.txt"
    marker.write_text("preserve")

    with pytest.raises(ValueError, match="must not already exist"):
        initialize_run(
            source_path=source,
            spec_path=spec,
            contract_path=contract,
            run_dir=run_dir,
            recorded_at_utc="2026-11-07T07:30:00+00:00",
        )

    assert marker.read_text() == "preserve"


def test_execute_fresh_run_journals_each_step_checkpoint_and_completion(tmp_path):
    source, spec, contract = prepare_inputs(tmp_path)
    run_dir = tmp_path / "run"
    initialize_run(
        source_path=source,
        spec_path=spec,
        contract_path=contract,
        run_dir=run_dir,
        recorded_at_utc="2026-11-07T07:30:00+00:00",
    )

    report = execute_fresh_run(
        source_path=source,
        spec_path=spec,
        run_dir=run_dir,
        recorded_at_utc="2026-11-07T07:31:00+00:00",
    )

    events = load_event_chain(run_dir / "journal.json")
    state = reduce_run_state(events)
    assert state.status == "completed"
    assert state.completed_step == 2
    assert [event.kind for event in events] == [
        "started",
        "step",
        "step",
        "checkpoint",
        "completed",
    ]
    assert report["progress"] == {"started_step": 0, "completed_step": 2}
    assert (run_dir / "checkpoint.pt").is_file()
    assert (run_dir / "training-report.json").is_file()


def test_execute_fresh_run_refuses_to_overwrite_completed_attempt(tmp_path):
    source, spec, contract = prepare_inputs(tmp_path)
    run_dir = tmp_path / "run"
    initialize_run(
        source_path=source,
        spec_path=spec,
        contract_path=contract,
        run_dir=run_dir,
        recorded_at_utc="2026-11-07T07:30:00+00:00",
    )
    execute_fresh_run(
        source_path=source,
        spec_path=spec,
        run_dir=run_dir,
        recorded_at_utc="2026-11-07T07:31:00+00:00",
    )

    with pytest.raises(ValueError, match="untouched journal"):
        execute_fresh_run(
            source_path=source,
            spec_path=spec,
            run_dir=run_dir,
            recorded_at_utc="2026-11-07T07:32:00+00:00",
        )


def test_finalize_and_verify_run_bundle_from_stored_bytes(tmp_path):
    source, spec, contract = prepare_inputs(tmp_path)
    run_dir = tmp_path / "run"
    initialize_run(
        source_path=source,
        spec_path=spec,
        contract_path=contract,
        run_dir=run_dir,
        recorded_at_utc="2026-11-07T07:30:00+00:00",
    )
    execute_fresh_run(
        source_path=source,
        spec_path=spec,
        run_dir=run_dir,
        recorded_at_utc="2026-11-07T07:31:00+00:00",
    )

    bundle = finalize_run_bundle(run_dir)

    assert verify_run_bundle(run_dir) == bundle
    assert bundle["artifacts"]["checkpoint.pt"]["bytes"] > 0


def test_bundle_verifier_detects_changed_training_report(tmp_path):
    source, spec, contract = prepare_inputs(tmp_path)
    run_dir = tmp_path / "run"
    initialize_run(
        source_path=source,
        spec_path=spec,
        contract_path=contract,
        run_dir=run_dir,
        recorded_at_utc="2026-11-07T07:30:00+00:00",
    )
    execute_fresh_run(
        source_path=source,
        spec_path=spec,
        run_dir=run_dir,
        recorded_at_utc="2026-11-07T07:31:00+00:00",
    )
    finalize_run_bundle(run_dir)
    (run_dir / "training-report.json").write_text("{}")

    with pytest.raises(ValueError, match="manifest does not match"):
        verify_run_bundle(run_dir)
