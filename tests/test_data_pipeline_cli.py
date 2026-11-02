from dataclasses import asdict, replace
import json

import pytest

from ai_journey.data_documents import DataDocument
from ai_journey.data_pipeline_cli import main
from ai_journey.data_pipeline_source import SourceEvidence
from ai_journey.tokenizer_sources import ENGLISH


def _source():
    documents = tuple(
        DataDocument(str(index), f"Useful public sentence number {index}.", ENGLISH.url)
        for index in range(8)
    )
    source = replace(ENGLISH, size_bytes=1)
    return SourceEvidence(asdict(source), source.url, source.license_url, documents)


def _config(path):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "quality": {
                    "min_characters": 5,
                    "max_characters": 4096,
                    "min_alphanumeric_fraction": 0.5,
                    "max_repeated_run": 8,
                },
                "validation_count": 2,
                "train_shards": 2,
                "validation_shards": 1,
                "seed": 59,
                "vocabulary_size": 280,
                "min_frequency": 2,
            }
        )
    )


def test_cli_builds_then_independently_verifies_token_observations(
    tmp_path, monkeypatch, capsys
):
    config = tmp_path / "config.json"
    snapshot = tmp_path / "source.conllu"
    output = tmp_path / "dataset"
    _config(config)
    snapshot.write_text("validated by source adapter")
    monkeypatch.setattr(
        "ai_journey.data_pipeline_cli.load_pipeline_source", lambda path: _source()
    )

    assert (
        main(
            [
                "build",
                "--snapshot",
                str(snapshot),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    built = json.loads(capsys.readouterr().out)
    assert main(["verify", "--dataset", str(output)]) == 0
    verified = json.loads(capsys.readouterr().out)

    assert built == verified
    assert verified["total_tokens"] > 0
    assert verified["bytes_per_token"] > 1


def test_cli_rejects_output_alias_before_reading_the_source(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    _config(config)
    called = False

    def load(_path):
        nonlocal called
        called = True

    monkeypatch.setattr("ai_journey.data_pipeline_cli.load_pipeline_source", load)
    with pytest.raises(SystemExit):
        main(
            [
                "build",
                "--snapshot",
                str(config),
                "--config",
                str(config),
                "--output",
                str(config),
            ]
        )

    assert called is False
