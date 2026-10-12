"""A public-data dropout comparison must preserve paired controls and inputs."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest import mock

from ai_journey.day38_ablation import run_dropout_ablation, main
from ai_journey.gpt_bigram import CorpusSource
from ai_journey.ablation_protocol import verify_ablation_report


class Day38AblationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.text = "First Citizen:\nWe know the model.\n" * 20
        self.source = CorpusSource(
            "test",
            "https://example.test/text",
            hashlib.sha256(self.text.encode()).hexdigest(),
            len(self.text.encode()),
            "test",
        )
        protocol = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "config/day-38-dropout-ablation.json"
            ).read_text()
        )
        protocol["model_config"].update(
            block_size=4, embedding_dim=8, head_count=2, layer_count=1
        )
        protocol["training_config"].update(steps=2, batch_size=2)
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text(json.dumps(protocol))

    def test_three_arms_and_three_seeds_keep_paired_controls(self):
        report = run_dropout_ablation(
            self.text,
            self.source,
            self.protocol,
            train_tokens=128,
            validation_tokens=64,
        )
        verify_ablation_report(report["ablation"])
        trials = report["ablation"]["result"]["trials"]
        self.assertEqual(len(trials), 9)
        for seed in (38, 39, 40):
            paired = [trial for trial in trials if trial["seed"] == seed]
            self.assertEqual(
                len({trial["initial_model_fingerprint"] for trial in paired}), 1
            )
            self.assertEqual(
                len({trial["first_batch_fingerprint"] for trial in paired}), 1
            )
        self.assertEqual(report["source"]["sha256"], self.source.sha256)

    def test_cli_publishes_evidence_and_protects_protocol(self):
        corpus = self.root / "corpus.txt"
        corpus.write_text(self.text)
        output = self.root / "report.json"
        args = [
            "--corpus",
            str(corpus),
            "--protocol",
            str(self.protocol),
            "--output",
            str(output),
            "--train-tokens",
            "128",
            "--validation-tokens",
            "64",
        ]
        with mock.patch(
            "ai_journey.day38_ablation.TINY_SHAKESPEARE", self.source
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertEqual(
            json.loads(output.read_text())["schema"], "day38-dropout-ablation-v1"
        )
        args[5] = str(self.protocol)
        original = self.protocol.read_bytes()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(args), 2)
        self.assertEqual(original, self.protocol.read_bytes())
