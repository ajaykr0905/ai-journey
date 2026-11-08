import json

import pytest

from ai_journey.data_cleaning import clean_documents
from ai_journey.data_documents import DataDocument
from ai_journey.minhash_cli import build_parser, main
from ai_journey.minhash_corpus import Project1Corpus
from ai_journey.minhash_dedup import MinHashDedupPolicy


def _corpus():
    documents = clean_documents(
        (
            DataDocument("a", "same public text", "public://a"),
            DataDocument("b", "same public text", "public://b"),
        )
    )
    return Project1Corpus(documents, "a" * 64, ("b" * 64,))


def test_build_command_publishes_and_prints_removal_summary(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "ai_journey.minhash_cli.load_project1_corpus", lambda _: _corpus()
    )
    monkeypatch.setattr(
        "ai_journey.minhash_cli.load_minhash_policy",
        lambda _: MinHashDedupPolicy(2, 64, 66, 32, 0.8),
    )
    output = tmp_path / "output"

    assert (
        main(
            [
                "build",
                "--dataset",
                str(tmp_path / "dataset"),
                "--config",
                str(tmp_path / "config.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["input_documents"] == 2
    assert summary["kept_documents"] == 1
    assert summary["removed_documents"] == 1
    assert summary["removed_fraction"] == 0.5
    assert len(summary["report_sha256"]) == 64
    assert (output / "manifest.json").is_file()


def test_build_command_rejects_output_aliases(tmp_path):
    path = tmp_path / "same"
    with pytest.raises(SystemExit):
        main(
            [
                "build",
                "--dataset",
                str(path),
                "--config",
                str(tmp_path / "config.json"),
                "--output",
                str(path),
            ]
        )


def test_parser_requires_a_command_and_build_arguments():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["build"])
