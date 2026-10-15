from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_journey.bpe import TokenizerError
from ai_journey.bpe_model_ablation import VocabularyProtocol
from ai_journey.bpe_model_cli import evidence_fingerprint, run_public_comparison
from ai_journey.bpe_model_verify import verify_comparison
from ai_journey.tokenizer_sources import PublicCorpus


class ModelComparisonReplayTests(unittest.TestCase):
    def test_replay_binds_source_training_and_metrics_not_just_self_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, sources = [], []
            for language in ("hi", "en"):
                data = "".join(
                    f"# sent_id = s{index}\n# text = {language} banana banana abracadabra sentence {index}\n\n"
                    for index in range(6)
                ).encode()
                path = root / f"{language}.conllu"
                path.write_bytes(data)
                paths.append(path)
                sources.append(
                    PublicCorpus(
                        language,
                        "public-test/fixture",
                        "a" * 40,
                        path.name,
                        sha256(data).hexdigest(),
                        len(data),
                    )
                )
            with patch("ai_journey.bpe_model_cli.HINDI", sources[0]), patch(
                "ai_journey.bpe_model_cli.ENGLISH", sources[1]
            ), patch("ai_journey.bpe_model_verify.HINDI", sources[0]), patch(
                "ai_journey.bpe_model_verify.ENGLISH", sources[1]
            ):
                report = run_public_comparison(
                    *paths,
                    VocabularyProtocol(
                        vocab_sizes=(256, 258, 260), seeds=(1, 2), steps=1, block_size=3
                    ),
                    train_count=2,
                    validation_count=2,
                )
                self.assertEqual(
                    verify_comparison(report, *paths)["verified_trials"], 12
                )
                corrupt = deepcopy(report)
                corrupt["languages"][0]["experiment"]["trials"][0][
                    "final_model_fingerprint"
                ] = "0" * 64
                corrupt["sha256"] = evidence_fingerprint(corrupt)
                with self.assertRaisesRegex(TokenizerError, "does not reproduce"):
                    verify_comparison(corrupt, *paths)
                corrupt = deepcopy(report)
                corrupt["languages"][0]["source"]["revision"] = "b" * 40
                corrupt["sha256"] = evidence_fingerprint(corrupt)
                with patch(
                    "ai_journey.bpe_model_verify.run_public_comparison"
                ) as replay:
                    with self.assertRaises(TokenizerError):
                        verify_comparison(corrupt, *paths)
                    replay.assert_not_called()
                paths[0].write_bytes(paths[0].read_bytes() + b"\n")
                with self.assertRaises(TokenizerError):
                    verify_comparison(report, *paths)
