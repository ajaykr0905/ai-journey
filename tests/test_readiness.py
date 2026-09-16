from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import (
    ConfigurationError,
    check_readiness,
    load_config,
    main,
)


class ReadinessTests(unittest.TestCase):
    def test_repository_config_is_ready(self) -> None:
        config = load_config(ROOT / "config" / "deployment_readiness.json")
        report = check_readiness(ROOT, config)
        self.assertTrue(report["ready"], report["checks"])
        serialized = json.dumps(report)
        self.assertNotIn("GITHUB_TOKEN=", serialized)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY=", serialized)

    def test_missing_required_item_fails(self) -> None:
        config = {
            "required_files": ["not-present.txt"],
            "required_directories": [],
            "required_commands": [],
            "required_env": [],
            "optional_env": [],
        }
        report = check_readiness(ROOT, config)
        self.assertFalse(report["ready"])
        self.assertEqual(report["checks"][0]["target"], "not-present.txt")
        self.assertEqual(report["checks"][0]["status"], "fail")

    def test_invalid_config_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "bad.json"
            config_path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_config(config_path)

    def test_cli_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            missing_config = directory_path / "missing.json"
            missing_config.write_text(
                json.dumps(
                    {
                        "required_files": ["absent.txt"],
                        "required_directories": [],
                        "required_commands": [],
                        "required_env": [],
                        "optional_env": [],
                    }
                ),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["--root", str(ROOT), "--config", str(missing_config)]),
                    1,
                )

            invalid_config = directory_path / "invalid.json"
            invalid_config.write_text("[]", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--config", str(invalid_config)]), 2)


if __name__ == "__main__":
    unittest.main()
