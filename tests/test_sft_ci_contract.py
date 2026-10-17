import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SFTContinuousVerificationTests(unittest.TestCase):
    def test_ci_runs_static_checks_comparison_and_exact_replay(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        for command in (
            "ruff check src/ai_journey/sft*.py tests/test_sft*.py",
            "ai-journey-chat-sft-compare --config config/chat-sft-comparison.json",
            "ai-journey-chat-sft-verify --report",
        ):
            self.assertIn(command, workflow)

    def test_readiness_requires_the_complete_operating_surface(self):
        required = json.loads((ROOT / "config/deployment_readiness.json").read_text())[
            "required_files"
        ]
        for path in (
            "src/ai_journey/sft_data.py",
            "src/ai_journey/sft_experiment.py",
            "src/ai_journey/sft_report.py",
            "src/ai_journey/sft_cli.py",
            "config/chat-sft-comparison.json",
            "docs/chat-sft-experiment.md",
            "tests/test_sft_data.py",
            "tests/test_sft_experiment.py",
            "tests/test_sft_report.py",
            "tests/test_sft_cli.py",
            "tests/test_sft_package.py",
            "tests/test_sft_ci_contract.py",
        ):
            with self.subTest(path=path):
                self.assertIn(path, required)
                self.assertTrue((ROOT / path).is_file())


if __name__ == "__main__":
    unittest.main()
