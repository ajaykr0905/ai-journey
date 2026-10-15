"""Replay every bounded decoder trial instead of trusting reported model metrics."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch

from .bpe import TokenizerError
from .bpe_ablation_summary import summarize_ablation
from .bpe_model_ablation import VocabularyProtocol
from .bpe_model_cli import evidence_fingerprint, read_json, run_public_comparison
from .tokenizer_sources import HINDI, ENGLISH


def _semantic(report):
    """Only wall time and host metadata are excluded from exact replay."""
    result = {
        key: value for key, value in report.items() if key not in {"runtime", "sha256"}
    }
    result["languages"] = []
    for language in report["languages"]:
        item = dict(language)
        item["experiment"] = dict(language["experiment"])
        item["experiment"]["trials"] = [
            {key: value for key, value in trial.items() if key != "training_seconds"}
            for trial in language["experiment"]["trials"]
        ]
        result["languages"].append(item)
    return result


def verify_comparison(report, hindi, english):
    """A digest detects corruption; source-backed full replay verifies the claims."""
    try:
        if report["schema"] != "public-bpe-model-comparison-v1" or report[
            "sha256"
        ] != evidence_fingerprint(report):
            raise TokenizerError("comparison schema or digest mismatch")
        if (
            report["runtime"]["torch"] != torch.__version__
            or report["runtime"]["device"] != "cpu"
        ):
            raise TokenizerError(
                "exact replay requires the recorded PyTorch version and CPU backend"
            )
        if len(report["languages"]) != 2 or report["split_seed"] != 40:
            raise TokenizerError(
                "comparison requires both pinned languages and split seed 40"
            )
        raw = report["languages"][0]["experiment"]["protocol"]
        protocol = VocabularyProtocol(
            **{
                **raw,
                "vocab_sizes": tuple(raw["vocab_sizes"]),
                "seeds": tuple(raw["seeds"]),
            }
        )
        for language, source in zip(report["languages"], (HINDI, ENGLISH)):
            if (
                language["source"] != asdict(source)
                or language["source_url"] != source.url
                or language["license_url"] != source.license_url
            ):
                raise TokenizerError("comparison source or licensing identity mismatch")
            if language["experiment"]["protocol"] != raw or language[
                "summary"
            ] != summarize_ablation(language["experiment"]):
                raise TokenizerError("comparison protocols or paired summaries differ")
        # Public-corpus byte/hash checks happen before any replay model is built.
        replay = run_public_comparison(
            hindi,
            english,
            protocol,
            train_count=report["train_count"],
            validation_count=report["validation_count"],
        )
        # Normalize tuples through the same JSON representation used by persisted evidence.
        import json

        if json.loads(json.dumps(_semantic(report))) != json.loads(
            json.dumps(_semantic(replay))
        ):
            raise TokenizerError(
                "comparison does not reproduce source identities, weights, losses, or trace"
            )
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise TokenizerError(f"invalid replay evidence: {exc}") from exc
    return {
        "verified_trials": sum(
            len(item["experiment"]["trials"]) for item in report["languages"]
        ),
        "sha256": report["sha256"],
        "policy": "exact CPU model/metric replay; elapsed times are observations only",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "hindi", "english"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_comparison(read_json(args.report), args.hindi, args.english)
    except (OSError, TypeError, ValueError, RuntimeError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Exactly replayed {result['verified_trials']} CPU decoder trials; sha256={result['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
