from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_journey.lora_cli import (
    CONFIG_BYTES,
    REPORT_BYTES,
    load_config,
    main,
    read_json,
    verify_main,
)
from ai_journey.lora_data import LoRAExperimentError
from ai_journey.lora_experiment import ComparisonConfig
from ai_journey.lora_report import replay_report


def tiny_controls():
    return ComparisonConfig(
        block_size=3,
        embedding_dim=4,
        head_count=1,
        batch_size=2,
        pretrain_steps=1,
        finetune_steps=1,
    )


class LoRACLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(asdict(tiny_controls())), encoding="utf-8")
        self.output = self.root / "report.json"
        self.argv = ["--config", str(self.config), "--output", str(self.output)]

    def test_compare_and_verify_entry_functions_with_repeatable_bytes(self):
        with redirect_stdout(StringIO()) as stdout:
            self.assertEqual(main(self.argv), 0)
        self.assertIn("2 paired seeds / 8 CPU trials", stdout.getvalue())
        before = self.output.read_bytes()
        with redirect_stdout(StringIO()):
            self.assertEqual(main(self.argv), 0)
            self.assertEqual(verify_main(["--report", str(self.output)]), 0)
        self.assertEqual(before, self.output.read_bytes())

    def test_config_output_collision_symlink_and_hardlink_preserve_inputs(self):
        before = self.config.read_bytes()
        aliases = [self.config, self.root / "symbolic.json", self.root / "hard.json"]
        aliases[1].symlink_to(self.config)
        os.link(self.config, aliases[2])
        for alias in aliases:
            with self.subTest(alias=alias), redirect_stderr(StringIO()):
                self.assertEqual(
                    main(["--config", str(self.config), "--output", str(alias)]), 2
                )
            self.assertEqual(before, self.config.read_bytes())

    def test_malformed_duplicate_nonfinite_and_oversized_json_are_rejected(self):
        for value in (
            '{"seeds":[],"seeds":[]}',
            '{"nested":{"a":1,"a":2}}',
            '{"x":NaN}',
            '{"x":Infinity}',
            "[]",
            "{}",
            " " * (CONFIG_BYTES + 1),
        ):
            self.config.write_text(value, encoding="utf-8")
            with self.subTest(value=value[:50]), self.assertRaises(LoRAExperimentError):
                load_config(self.config)
        self.config.write_bytes(b"\xff\xfe")
        with self.assertRaises(LoRAExperimentError):
            load_config(self.config)

    def test_invalid_work_controls_fail_before_training_and_leave_report_intact(self):
        self.output.write_bytes(b"previous report")
        for change in (
            {"seeds": [1]},
            {"finetune_steps": True},
            {
                "block_size": 32,
                "embedding_dim": 32,
                "batch_size": 8,
                "pretrain_steps": 64,
                "finetune_steps": 64,
            },
            {"unexpected": 1},
        ):
            self.config.write_text(json.dumps({**asdict(tiny_controls()), **change}))
            with patch("ai_journey.lora_cli.run_comparison") as runner, redirect_stderr(
                StringIO()
            ):
                self.assertEqual(main(self.argv), 2)
                runner.assert_not_called()
            self.assertEqual(self.output.read_bytes(), b"previous report")

    def test_training_and_publication_failures_do_not_replace_prior_output(self):
        self.output.write_bytes(b"previous report")
        for target in ("run_comparison", "write_report"):
            with patch(
                f"ai_journey.lora_cli.{target}",
                side_effect=RuntimeError("injected failure"),
            ), redirect_stderr(StringIO()):
                self.assertEqual(main(self.argv), 2)
            self.assertEqual(self.output.read_bytes(), b"previous report")

    def test_verify_rejects_oversized_and_corrupt_reports_without_writing(self):
        for data in (
            b" " * (REPORT_BYTES + 1),
            b'{"schema":"invalid"}',
            b'{"a":1,"a":2}',
        ):
            self.output.write_bytes(data)
            with patch(
                "ai_journey.lora_cli.replay_report", wraps=replay_report
            ) as replay, redirect_stderr(StringIO()):
                self.assertEqual(verify_main(["--report", str(self.output)]), 2)
                if len(data) > REPORT_BYTES or b'"a"' in data:
                    replay.assert_not_called()
            self.assertEqual(self.output.read_bytes(), data)
        with self.assertRaises(LoRAExperimentError):
            read_json(self.output, limit=1)

    def test_missing_paths_return_errors(self):
        self.config.unlink()
        with redirect_stderr(StringIO()):
            self.assertEqual(main(self.argv), 2)
            self.assertEqual(verify_main(["--report", str(self.output)]), 2)
        self.assertFalse(self.output.exists())
