from dataclasses import asdict
from hashlib import sha256
import json

import pytest

from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_training import (
    GPT2TrainingConfig,
    GPT2TrainingSource,
    GPT2TrainingSpec,
)
from ai_journey.gpt2_training_cli import (
    load_source_bytes,
    main,
    replay_report,
    run_and_publish,
    verify_report,
)


def prepare_files(tmp_path):
    source = b"public training text\n" * 8
    source_path = tmp_path / "source.txt"
    source_path.write_bytes(source)
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
            seed=47,
        ),
        source=GPT2TrainingSource(
            name="Synthetic public names",
            url="https://github.com/ajaykr0905/ai-journey",
            revision="fixture-v1",
            license="CC0-1.0",
            sha256=sha256(source).hexdigest(),
        ),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": asdict(spec.model),
                "training": asdict(spec.training),
                "source": asdict(spec.source),
            }
        ),
        encoding="utf-8",
    )
    return source_path, spec_path


def test_cli_library_runs_verifies_and_replays_training(tmp_path):
    source, spec = prepare_files(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    report = tmp_path / "report.json"

    evidence = run_and_publish(
        source_path=source,
        spec_path=spec,
        checkpoint_path=checkpoint,
        report_path=report,
    )

    assert verify_report(report) == evidence
    assert replay_report(source_path=source, report_path=report) == evidence


def test_cli_main_prints_canonical_verified_report(tmp_path, capsys):
    source, spec = prepare_files(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    report = tmp_path / "report.json"

    assert (
        main(
            [
                "train",
                "--source",
                str(source),
                "--spec",
                str(spec),
                "--checkpoint",
                str(checkpoint),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed == verify_report(report)


def test_source_loader_rejects_symlink_and_oversize(tmp_path):
    target = tmp_path / "source.txt"
    target.write_bytes(b"abc")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="non-symlink"):
        load_source_bytes(link)
    with pytest.raises(ValueError, match="byte limits"):
        load_source_bytes(target, max_bytes=2)
