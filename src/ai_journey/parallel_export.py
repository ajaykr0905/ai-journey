"""Export small, disjoint public text partitions with licensing and provenance."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
import tempfile

from .bpe import TokenizerError
from .parallel_corpus import align_and_split
from .serving_benchmark import _publish
from .tokenizer_sources import HINDI, ENGLISH, load_public_corpus


def _write_text(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hindi", "english", "train-output", "validation-output", "manifest"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--language", choices=("hi", "en"), default="hi")
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--validation-count", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        paths = [
            getattr(args, name).resolve()
            for name in (
                "hindi",
                "english",
                "train_output",
                "validation_output",
                "manifest",
            )
        ]
        if (
            len(set(paths)) != 5
            or not 1 <= args.train_count <= 100
            or not 1 <= args.validation_count <= 100
        ):
            raise TokenizerError(
                "export paths must be distinct and sample counts in [1, 100]"
            )
        first = load_public_corpus(args.hindi, HINDI)
        second = load_public_corpus(args.english, ENGLISH)
        split = align_and_split(first, second, train_count=args.train_count, seed=40)
        if len(split.first_validation) < args.validation_count:
            raise TokenizerError(
                "requested validation sample exceeds available held-out data"
            )
        train, heldout = (
            (split.first_train, split.first_validation)
            if args.language == "hi"
            else (split.second_train, split.second_validation)
        )
        heldout = heldout[: args.validation_count]
        source = HINDI if args.language == "hi" else ENGLISH
        manifest = {
            "schema_version": 1,
            "source": asdict(source),
            "source_url": source.url,
            "license_url": source.license_url,
            "attribution": "Parallel Universal Dependencies, CoNLL 2017; DFKI, Google, and the UD community; see source README for contributor credits.",
            "split_seed": 40,
            "unused_sentences": len(first) - len(train) - len(heldout),
            "partitions": {},
        }
        for name, records, path in (
            ("train", train, args.train_output),
            ("validation", heldout, args.validation_output),
        ):
            data = ("\n".join(row.text for row in records) + "\n").encode("utf-8")
            manifest["partitions"][name] = {
                "sentence_ids": [row.sentence_id for row in records],
                "sha256": sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
            _write_text(path, data)
        # Published last: readers should verify both partition hashes before use.
        _publish(args.manifest, manifest)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Exported disjoint {args.language} sample; wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
