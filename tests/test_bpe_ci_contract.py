import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BPEOperatingGateTests(unittest.TestCase):
    def test_pipeline_has_ci_lint_and_readiness_gates(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        certification = (ROOT / "scripts/verify_bpe_lab.py").read_text()
        self.assertIn("python scripts/verify_bpe_lab.py", workflow)
        self.assertIn("ruff check src/ai_journey/bpe*.py", workflow)
        self.assertIn('"audit-training"', certification)
        self.assertIn('"verify-training-audit"', certification)
        self.assertIn('"tokenizer_audit_sha256"', certification)
        required = json.loads((ROOT / "config/deployment_readiness.json").read_text())[
            "required_files"
        ]
        for path in (
            "scripts/verify_bpe_lab.py",
            "src/ai_journey/export_verification.py",
            "docs/bpe-training-serving.md",
        ):
            self.assertIn(path, required)
            self.assertTrue((ROOT / path).is_file())

    def test_missing_source_fails_pipeline_without_producing_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/verify_bpe_lab.py"),
                    "--hindi",
                    str(Path(directory) / "missing-hi"),
                    "--english",
                    str(Path(directory) / "missing-en"),
                    "--artifact-dir",
                    str(output),
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertFalse((output / "certification.json").exists())

    def test_operations_guide_includes_source_backed_training_audit(self):
        guide = (ROOT / "docs/bpe-training-serving.md").read_text()
        self.assertIn("ai-journey-tokenizer audit-training", guide)
        self.assertIn("ai-journey-tokenizer verify-training-audit", guide)
        self.assertIn("document boundaries", guide)
        self.assertIn("Unicode normalization collisions", guide)
