from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
import tomllib
import unittest


class InstalledBPECommandTests(unittest.TestCase):
    def test_package_declares_callable_and_help_safe_operating_commands(self):
        configuration = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        )
        scripts = configuration["project"]["scripts"]
        expected = {
            "ai-journey-tokenizer-resume": "ai_journey.tokenizer_restart_cli:main",
            "ai-journey-tokenizer-compare": "ai_journey.day40_cli:main",
            "ai-journey-corpus-export": "ai_journey.parallel_export:main",
            "ai-journey-bpe-train": "ai_journey.bpe_train_cli:main",
            "ai-journey-bpe-infer": "ai_journey.bpe_infer_cli:main",
            "ai-journey-bpe-encode-benchmark": "ai_journey.bpe_benchmark:main",
        }
        for name, target in expected.items():
            with self.subTest(command=name):
                self.assertEqual(scripts[name], target)
                module, function = target.split(":")
                entrypoint = getattr(importlib.import_module(module), function)
                self.assertTrue(callable(entrypoint))
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                    entrypoint(["--help"])
                self.assertEqual(stopped.exception.code, 0)
                self.assertIn("usage:", output.getvalue())
