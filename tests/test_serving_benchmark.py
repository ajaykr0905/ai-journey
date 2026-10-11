from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from ai_journey.serving_benchmark import (
    benchmark_generation,
    build_parser,
    main,
    _publish,
)
from ai_journey.transformer_lab import DecoderLanguageModel, TransformerConfig


class ServingBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(37)
            self.model = DecoderLanguageModel(
                TransformerConfig(
                    vocab_size=7,
                    block_size=4,
                    embedding_dim=4,
                    head_count=1,
                    layer_count=1,
                )
            )
        self.prompt = torch.tensor([[1, 2]])

    def test_actual_generation_rollover_preserves_caller_state(self) -> None:
        self.model.train()
        self.model.blocks[0].eval()
        modes = [module.training for module in self.model.modules()]
        parameters = copy.deepcopy(self.model.state_dict())
        prompt = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        original = prompt.clone()
        rng = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        report = benchmark_generation(
            self.model, prompt, new_tokens=3, repeats=1, warmup=0
        )
        self.assertEqual(report["output_shape"], [2, 8])
        self.assertTrue(report["outputs_match"])
        self.assertEqual([module.training for module in self.model.modules()], modes)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(prompt, original))
        self.assertEqual(threads, torch.get_num_threads())
        for name, value in parameters.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]))
        json.dumps(report, allow_nan=False)
        self.assertEqual(
            set(report["runtime"]),
            {"system", "machine", "python", "torch", "cpu_threads"},
        )
        for digest in report["identity"].values():
            self.assertEqual(len(digest), 64)
        for result in report["methods"].values():
            self.assertGreater(result["median_ns"], 0)
            self.assertGreater(result["generated_tokens_per_second"], 0)

    def test_control_and_prompt_bounds(self) -> None:
        for name, values in {
            "new_tokens": (True, 0, 33),
            "repeats": (True, 0, 21),
            "warmup": (True, -1, 6),
            "seed": (True, -1, 2**63),
        }.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(
                    (TypeError, ValueError)
                ):
                    benchmark_generation(self.model, self.prompt, **{name: value})
        for prompt in (
            torch.zeros(5, 1, dtype=torch.long),
            torch.zeros(1, 129, dtype=torch.long),
            torch.zeros(1, 2),
            torch.tensor([[7]]),
            torch.ones(1, 2, dtype=torch.long, device="meta"),
        ):
            with self.subTest(shape=prompt.shape), self.assertRaises(
                (TypeError, ValueError)
            ):
                benchmark_generation(self.model, prompt)
        with self.assertRaises(TypeError):
            benchmark_generation(object(), self.prompt)
        with mock.patch.object(self.model, "_validate_token_ids") as validate:
            for prompt in (
                torch.zeros(5, 1, dtype=torch.long),
                torch.zeros(1, 129, dtype=torch.long),
                torch.empty(1, 0, dtype=torch.long),
                torch.zeros(2, dtype=torch.long),
            ):
                with self.subTest(shape=prompt.shape), self.assertRaises(ValueError):
                    benchmark_generation(self.model, prompt)
            validate.assert_not_called()
        for field, value in (
            ("block_size", 129),
            ("embedding_dim", 132),
            ("layer_count", 5),
            ("normalization_mode", "scratch_batch_norm"),
            ("vocab_size", 300_000),
        ):
            config = {**self.model.config.__dict__, field: value}
            model = DecoderLanguageModel(TransformerConfig(**config))
            with self.subTest(field=field), self.assertRaises(ValueError):
                benchmark_generation(model, self.prompt)
        with self.assertRaisesRegex(ValueError, "CPU model"):
            benchmark_generation(self.model.to("meta"), self.prompt)

    def test_summary_excludes_warmup_and_uses_nearest_rank(self) -> None:
        output = torch.tensor([[1, 2, 3]])
        durations = (999, 888, 10, 20, 30, 40, 50, 60)
        clock = []
        for index, duration in enumerate(durations):
            clock.extend([index * 1000, index * 1000 + duration])
        with mock.patch.object(
            self.model, "generate", return_value=output
        ) as full, mock.patch(
            "ai_journey.serving_benchmark.generate_cached", return_value=output
        ) as cached, mock.patch(
            "ai_journey.serving_benchmark.perf_counter_ns", side_effect=clock
        ):
            report = benchmark_generation(
                self.model, self.prompt, new_tokens=1, repeats=3, warmup=1
            )
        self.assertEqual(full.call_count, 4)
        self.assertEqual(cached.call_count, 4)
        result = report["methods"]["full"]
        self.assertEqual(result["sample_ns"], [10, 30, 50])
        self.assertEqual(result["median_ns"], 30)
        self.assertEqual(result["p95_ns"], 50)
        self.assertAlmostEqual(result["generated_tokens_per_second"], 1e9 / 30)
        self.assertEqual(report["methods"]["cached"]["sample_ns"], [20, 40, 60])

    def test_rejects_mismatch_zero_duration_and_generation_errors(self) -> None:
        output = torch.tensor([[1, 2, 3]])
        with mock.patch.object(self.model, "generate", return_value=output), mock.patch(
            "ai_journey.serving_benchmark.generate_cached", return_value=output + 1
        ):
            with self.assertRaisesRegex(ValueError, "match exactly"):
                benchmark_generation(
                    self.model, self.prompt, new_tokens=1, repeats=1, warmup=0
                )
        with mock.patch.object(
            self.model, "generate", side_effect=[output, output + 1]
        ), mock.patch(
            "ai_journey.serving_benchmark.generate_cached", return_value=output
        ):
            with self.assertRaisesRegex(ValueError, "match exactly"):
                benchmark_generation(
                    self.model, self.prompt, new_tokens=1, repeats=2, warmup=0
                )
        for elapsed in (0, -1, float("inf"), float("nan")):
            with mock.patch.object(
                self.model, "generate", return_value=output
            ), mock.patch(
                "ai_journey.serving_benchmark.perf_counter_ns", side_effect=[0, elapsed]
            ):
                with self.assertRaisesRegex(ValueError, "duration"):
                    benchmark_generation(
                        self.model, self.prompt, new_tokens=1, repeats=1, warmup=0
                    )
        with mock.patch.object(
            self.model, "generate", side_effect=RuntimeError("broken")
        ):
            with self.assertRaisesRegex(RuntimeError, "broken"):
                benchmark_generation(self.model, self.prompt, repeats=1, warmup=0)

    def test_cli_publication_and_rng_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("old", encoding="utf-8")
            rng = torch.get_rng_state().clone()
            output = torch.tensor([[1, 2, 3, 4, 1]])
            with mock.patch.object(
                DecoderLanguageModel, "generate", return_value=output
            ), mock.patch(
                "ai_journey.serving_benchmark.generate_cached", return_value=output
            ):
                self.assertEqual(
                    main(
                        [
                            "--output",
                            str(path),
                            "--new-tokens",
                            "1",
                            "--repeats",
                            "1",
                            "--warmup",
                            "0",
                        ]
                    ),
                    0,
                )
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("untrained", report["model_source"])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            previous = path.read_bytes()
            with mock.patch(
                "ai_journey.serving_benchmark.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaises(OSError):
                    _publish(path, {"valid": True})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with mock.patch(
                "ai_journey.serving_benchmark.os.fsync",
                side_effect=OSError("sync failed"),
            ):
                with self.assertRaises(OSError):
                    _publish(path, {"valid": True})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with self.assertRaises(ValueError):
                _publish(path, {"invalid": float("nan")})
            self.assertEqual(path.read_bytes(), previous)
            with self.assertRaises(SystemExit):
                main(["--output", str(path), "--new-tokens", "0"])
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])


if __name__ == "__main__":
    unittest.main()
