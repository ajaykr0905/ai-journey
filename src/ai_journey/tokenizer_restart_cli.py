"""Persist and resume exact corpus-bound tokenizer training between processes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .bpe import DEFAULT_PATTERN, TokenizerError
from .serving_benchmark import _publish
from .tokenizer_cli import read_documents
from .tokenizer_io import save_tokenizer
from .tokenizer_training import BPETrainingSession


def _digest(state):
    return sha256(
        json.dumps(
            state, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def save_training_state(path: Path, session: BPETrainingSession):
    if not isinstance(session, BPETrainingSession):
        raise TypeError("checkpoint requires a BPETrainingSession")
    state = session.snapshot()
    _publish(path, {"schema_version": 1, "state": state, "sha256": _digest(state)})


def load_training_state(path: Path, documents):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TokenizerError("duplicate training checkpoint field")
            result[key] = value
        return result

    with path.open("rb") as stream:
        payload = stream.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise TokenizerError("training checkpoint exceeds 2 MiB")
    checkpoint = json.loads(payload, object_pairs_hook=unique)
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"schema_version", "state", "sha256"}
        or type(checkpoint["schema_version"]) is not int
        or checkpoint["schema_version"] != 1
    ):
        raise TokenizerError("training checkpoint fields are invalid")
    if _digest(checkpoint["state"]) != checkpoint["sha256"]:
        raise TokenizerError("training checkpoint digest mismatch")
    return BPETrainingSession.restore(documents, checkpoint["state"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "tokenizer", "state"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=320)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--regex", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if (
            len({args.corpus.resolve(), args.tokenizer.resolve(), args.state.resolve()})
            != 3
            or not 256 <= args.vocab_size <= 4096
        ):
            raise TokenizerError("training paths or target vocabulary are invalid")
        documents = read_documents(args.corpus)
        pattern = DEFAULT_PATTERN if args.regex else None
        session = (
            load_training_state(args.state, documents)
            if args.resume
            else BPETrainingSession(
                documents, pattern=pattern, min_frequency=args.min_frequency
            )
        )
        snapshot = session.snapshot()
        if (
            snapshot["pattern"] != pattern
            or snapshot["min_frequency"] != args.min_frequency
        ):
            raise TokenizerError("resume controls must match the saved training policy")
        if args.vocab_size < session.tokenizer.vocab_size:
            raise TokenizerError("resume cannot shrink the existing merge prefix")
        tokenizer = session.advance(args.vocab_size - session.tokenizer.vocab_size)
        save_tokenizer(args.tokenizer, tokenizer)
        save_training_state(args.state, session)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Verified canonical training prefix; vocabulary={tokenizer.vocab_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
