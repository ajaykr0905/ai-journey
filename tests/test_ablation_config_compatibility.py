import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_journey.ablation_protocol import load_ablation_protocol
from ai_journey.transformer_lab import TransformerConfig

ROOT = Path(__file__).resolve().parents[1]


class AblationConfigCompatibilityTests(unittest.TestCase):
    def test_legacy_protocol_retains_original_equations_with_current_defaults(self):
        path = ROOT / "config/day-31-learning-rate-ablation.json"
        raw = json.loads(path.read_text())
        protocol = load_ablation_protocol(path, vocab_size=12)
        expected = TransformerConfig(vocab_size=12, **raw["model_config"])
        self.assertEqual(asdict(protocol.baseline_arm.model_config), asdict(expected))

    def test_missing_legacy_or_unknown_control_is_rejected(self):
        payload = json.loads(
            (ROOT / "config/day-31-learning-rate-ablation.json").read_text()
        )
        with TemporaryDirectory() as directory:
            path = Path(directory, "protocol.json")
            for problem in ("missing", "unknown"):
                raw = json.loads(json.dumps(payload))
                if problem == "missing":
                    del raw["model_config"]["dropout"]
                else:
                    raw["model_config"]["invented_backend"] = "unsafe"
                path.write_text(json.dumps(raw))
                with self.subTest(problem=problem), self.assertRaisesRegex(
                    ValueError, "fields"
                ):
                    load_ablation_protocol(path, vocab_size=12)
