import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ai_journey.sft_cli import (
    CONFIG_BYTES,
    REPORT_BYTES,
    load_config,
    main,
    read_json,
    verify_main,
)
from ai_journey.sft_data import SFTDataError
from ai_journey.sft_experiment import SFTComparisonConfig


def tiny_controls():
    return SFTComparisonConfig(
        block_size=16,
        embedding_dim=8,
        head_count=2,
        batch_size=2,
        base_steps=1,
        sft_steps=1,
        evaluation_batch_size=64,
    )


class SFTCLITests(unittest.TestCase):
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
        self.assertIn("2 paired bases / 8 CPU chat SFT trials", stdout.getvalue())
        before = self.output.read_bytes()
        with redirect_stdout(StringIO()):
            self.assertEqual(main(self.argv), 0)
            self.assertEqual(verify_main(["--report", str(self.output)]), 0)
        self.assertEqual(before, self.output.read_bytes())

    def test_config_output_collision_symlink_and_hardlink_preserve_input(self):
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
            with self.subTest(value=value[:50]), self.assertRaises(SFTDataError):
                load_config(self.config)
        self.config.write_bytes(b"\xff\xfe")
        with self.assertRaises(SFTDataError):
            load_config(self.config)

    def test_invalid_controls_fail_before_training_and_preserve_report(self):
        self.output.write_bytes(b"previous report")
        for change in (
            {"seeds": [1]},
            {"sft_steps": True},
            {
                "block_size": 256,
                "embedding_dim": 64,
                "batch_size": 16,
                "base_steps": 128,
                "sft_steps": 128,
            },
            {"unexpected": 1},
        ):
            self.config.write_text(json.dumps({**asdict(tiny_controls()), **change}))
            with (
                patch("ai_journey.sft_cli.run_sft_comparison") as runner,
                redirect_stderr(StringIO()),
            ):
                self.assertEqual(main(self.argv), 2)
                runner.assert_not_called()
            self.assertEqual(self.output.read_bytes(), b"previous report")

    def test_failures_and_corrupt_reports_do_not_replace_inputs(self):
        self.output.write_bytes(b"previous report")
        for target in ("run_sft_comparison", "write_sft_report"):
            with (
                patch(
                    f"ai_journey.sft_cli.{target}", side_effect=RuntimeError("injected")
                ),
                redirect_stderr(StringIO()),
            ):
                self.assertEqual(main(self.argv), 2)
            self.assertEqual(self.output.read_bytes(), b"previous report")
        for data in (
            b" " * (REPORT_BYTES + 1),
            b'{"schema":"invalid"}',
            b'{"a":1,"a":2}',
        ):
            self.output.write_bytes(data)
            with redirect_stderr(StringIO()):
                self.assertEqual(verify_main(["--report", str(self.output)]), 2)
            self.assertEqual(self.output.read_bytes(), data)

    def test_missing_paths_return_errors(self):
        self.config.unlink()
        with redirect_stderr(StringIO()):
            self.assertEqual(main(self.argv), 2)
            self.assertEqual(verify_main(["--report", str(self.output)]), 2)
        self.assertFalse(self.output.exists())
        with self.assertRaises(OSError):
            read_json(self.root / "missing", limit=1)


if __name__ == "__main__":
    unittest.main()
