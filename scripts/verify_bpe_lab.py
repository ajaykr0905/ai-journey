"""Certify the installed, pinned-public-data BPE operating workflow on CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ai_journey.bpe import TokenizerError
from ai_journey.bpe_benchmark import main as benchmark_encoder
from ai_journey.bpe_infer_cli import main as serve
from ai_journey.bpe_train_cli import main as train
from ai_journey.parallel_export import main as export
from ai_journey.serving_benchmark import _publish
from ai_journey.tokenizer_restart_cli import main as train_tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hindi", type=Path, required=True)
    parser.add_argument("--english", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.artifact_dir
    train_text, validation_text, manifest = (
        root / name for name in ("train.txt", "validation.txt", "data.json")
    )
    export(
        [
            "--hindi",
            str(args.hindi),
            "--english",
            str(args.english),
            "--train-output",
            str(train_text),
            "--validation-output",
            str(validation_text),
            "--manifest",
            str(manifest),
        ]
    )
    report, checkpoint, codec = (
        root / name for name in ("training.json", "model.pt", "codec.json")
    )
    training_args = [
        "--train",
        str(train_text),
        "--validation",
        str(validation_text),
        "--manifest",
        str(manifest),
        "--source-snapshot",
        str(args.hindi),
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(codec),
        "--output",
        str(report),
        "--steps",
        "5",
    ]
    train(training_args)
    fingerprint = json.loads(report.read_text())["experiment"]["model_fingerprint"]
    train(training_args + ["--resume"])
    if json.loads(report.read_text())["experiment"]["model_fingerprint"] != fingerprint:
        raise TokenizerError("completed model restart changed the trained weights")
    serving_args = [
        "--report",
        str(report),
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(codec),
        "--train",
        str(train_text),
        "--validation",
        str(validation_text),
        "--prompt",
        "नमस्ते",
        "--new-tokens",
        "4",
        "--top-k",
        "1",
    ]
    samples = []
    for backend in ("full", "cached"):
        output = root / f"{backend}-sample.json"
        serve(
            serving_args
            + ["--output", str(output)]
            + (["--cached"] if backend == "cached" else [])
        )
        sample = json.loads(output.read_text())["sample"]
        sample.pop("backend")
        samples.append(sample)
    if samples[0] != samples[1]:
        raise TokenizerError("pipeline cache/full-forward token parity failed")
    serve(
        serving_args
        + [
            "--operation",
            "benchmark",
            "--repeats",
            "2",
            "--output",
            str(root / "serving-benchmark.json"),
        ]
    )
    base_codec, state = root / "base-codec.json", root / "tokenizer-state.json"
    tokenizer_args = [
        "--corpus",
        str(train_text),
        "--tokenizer",
        str(base_codec),
        "--state",
        str(state),
    ]
    train_tokenizer(tokenizer_args + ["--vocab-size", "270"])
    train_tokenizer(tokenizer_args + ["--vocab-size", "288", "--resume"])
    benchmark_encoder(
        [
            "--tokenizer",
            str(base_codec),
            "--corpus",
            str(train_text),
            "--output",
            str(root / "encoder-benchmark.json"),
        ]
    )
    _publish(
        root / "certification.json",
        {
            "schema_version": 1,
            "model_sha256": fingerprint,
            "sample_sha256": sha256(
                json.dumps(samples[0], sort_keys=True).encode()
            ).hexdigest(),
            "dataset": json.loads(manifest.read_text()),
            "scope": "Pinned public-data CPU smoke; exact model restart, durable tokenizer resume, paired UTF-8 serving and encoder parity passed.",
        },
    )
    print("BPE operating workflow certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
