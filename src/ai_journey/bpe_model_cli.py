"""Run licensed, source-pinned Hindi/English vocabulary-to-decoder comparisons."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import torch

from .bpe import TokenizerError
from .bpe_ablation_summary import summarize_ablation
from .bpe_model_ablation import VocabularyProtocol, run_vocabulary_ablation
from .parallel_corpus import align_and_split
from .serving_benchmark import _publish
from .tokenizer_sources import HINDI, ENGLISH, load_public_corpus


def read_json(path: Path, *, limit=2 * 1024 * 1024):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise TokenizerError("duplicate JSON field")
            value[key] = item
        return value

    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise TokenizerError("JSON exceeds its byte budget")
    return json.loads(data, object_pairs_hook=unique)


def load_protocol(path):
    raw = read_json(path, limit=16384)
    return VocabularyProtocol(
        **{
            **raw,
            "vocab_sizes": tuple(raw["vocab_sizes"]),
            "seeds": tuple(raw["seeds"]),
        }
    )


def evidence_fingerprint(report):
    return sha256(
        json.dumps(
            {key: value for key, value in report.items() if key != "sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def run_public_comparison(
    hindi, english, protocol, *, train_count=10, validation_count=10
):
    for count in (train_count, validation_count):
        if type(count) is not int or not 1 <= count <= 20:
            raise TokenizerError("public comparison sample counts must be in [1, 20]")
    first = load_public_corpus(hindi, HINDI)
    second = load_public_corpus(english, ENGLISH)
    split = align_and_split(first, second, train_count=train_count, seed=40)
    if len(split.first_validation) < validation_count:
        raise TokenizerError("not enough held-out sentences for the requested sample")
    languages = []
    for source, train, heldout in (
        (HINDI, split.first_train, split.first_validation[:validation_count]),
        (ENGLISH, split.second_train, split.second_validation[:validation_count]),
    ):
        experiment = run_vocabulary_ablation(
            tuple(row.text for row in train),
            tuple(row.text for row in heldout),
            protocol,
        )
        languages.append(
            {
                "source": asdict(source),
                "source_url": source.url,
                "license_url": source.license_url,
                "attribution": "Parallel Universal Dependencies, CoNLL 2017; DFKI, Google, and the UD community; see source README for contributors.",
                "train_sentence_ids": [row.sentence_id for row in train],
                "validation_sentence_ids": [row.sentence_id for row in heldout],
                "unused_sentences": len(first) - len(train) - len(heldout),
                "experiment": experiment,
                "summary": summarize_ablation(experiment),
            }
        )
    report = {
        "schema": "public-bpe-model-comparison-v1",
        "split_seed": 40,
        "train_count": train_count,
        "validation_count": validation_count,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
        },
        "languages": languages,
    }
    report["sha256"] = evidence_fingerprint(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hindi", "english", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--validation-count", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {
            args.hindi.resolve(),
            args.english.resolve(),
            args.protocol.resolve(),
        }:
            raise TokenizerError("comparison output must preserve input files")
        report = run_public_comparison(
            args.hindi,
            args.english,
            load_protocol(args.protocol),
            train_count=args.train_count,
            validation_count=args.validation_count,
        )
        _publish(args.output, report)
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
        RecursionError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Verified {sum(len(row['experiment']['trials']) for row in report['languages'])} CPU decoder trials; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
