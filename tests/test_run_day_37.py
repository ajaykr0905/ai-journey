from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ai_journey.day37_cli import main
from ai_journey.transformer_evidence import load_certification_report

ROOT = Path(__file__).resolve().parents[1]


class Day37CLITests(unittest.TestCase):
    def test_cli_publishes_certification_and_matched_trials(self):
        with TemporaryDirectory() as directory:
            output = Path(directory, "certification.json")
            ablation = Path(directory, "ablation.json")
            with redirect_stdout(StringIO()):
                code = main(
                    [
                        "--output",
                        str(output),
                        "--protocol",
                        str(ROOT / "config/day-37-head-count-ablation.json"),
                        "--corpus",
                        str(ROOT / "data/day-19-demo-names.txt"),
                        "--ablation-output",
                        str(ablation),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(ablation.exists())
            load_certification_report(output)

    def test_invalid_controls_preserve_existing_output(self):
        with TemporaryDirectory() as directory:
            output = Path(directory, "certification.json")
            output.write_text("previous")
            for extra in (
                ["--heads", "3"],
                ["--seed", "-1"],
                ["--tolerance", "nan"],
                ["--corpus", "missing"],
            ):
                with self.subTest(extra=extra), redirect_stderr(StringIO()):
                    self.assertEqual(main(["--output", str(output), *extra]), 2)
                self.assertEqual(output.read_text(), "previous")

    def test_report_cannot_overwrite_corpus_or_protocol(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "input.txt")
            path.write_text("abcde\n" * 20)
            with redirect_stderr(StringIO()):
                code = main(
                    [
                        "--output",
                        str(path),
                        "--corpus",
                        str(path),
                        "--protocol",
                        str(ROOT / "config/day-37-head-count-ablation.json"),
                        "--ablation-output",
                        str(Path(directory, "ablation.json")),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertEqual(path.read_text(), "abcde\n" * 20)

    def test_post_norm_certification(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory, "post.json")
            self.assertEqual(
                main(["--output", str(output), "--norm-placement", "post"]), 0
            )
            self.assertEqual(
                load_certification_report(output)["result"]["config"]["model"][
                    "normalization_placement"
                ],
                "post",
            )
