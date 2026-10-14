"""Bounded portable tokenizer checkpoints with corruption detection."""

import json
from hashlib import sha256
from pathlib import Path

from .bpe import ByteTokenizer, RegexTokenizer, SpecialTokenizer, TokenizerError


def tokenizer_state(tokenizer) -> dict:
    if isinstance(tokenizer, SpecialTokenizer):
        return {
            "kind": "special",
            "base": tokenizer_state(tokenizer.tokenizer),
            "names": list(tokenizer.names),
        }
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("unsupported tokenizer")
    state = {
        "kind": "regex" if isinstance(tokenizer, RegexTokenizer) else "byte",
        "merges": [list(pair) for pair in tokenizer.merges],
    }
    if isinstance(tokenizer, RegexTokenizer):
        state["pattern"] = tokenizer.pattern
    return state


def tokenizer_fingerprint(tokenizer) -> str:
    return sha256(
        json.dumps(
            tokenizer_state(tokenizer),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _restore(state):
    if not isinstance(state, dict):
        raise TokenizerError("tokenizer state must be an object")
    kind = state.get("kind")
    if kind == "special":
        if set(state) != {"kind", "base", "names"} or not isinstance(
            state["names"], list
        ):
            raise TokenizerError("special tokenizer fields are invalid")
        base = state["base"]
        if not isinstance(base, dict) or base.get("kind") == "special":
            raise TokenizerError("special tokenizer base must be byte or regex")
        return SpecialTokenizer(_restore(base), tuple(state["names"]))
    expected = {"kind", "merges"} | ({"pattern"} if kind == "regex" else set())
    if kind not in {"byte", "regex"} or set(state) != expected:
        raise TokenizerError("tokenizer state fields are invalid")
    if not isinstance(state["merges"], list) or any(
        not isinstance(pair, list) for pair in state["merges"]
    ):
        raise TokenizerError("merges must be an array of pairs")
    merges = tuple(tuple(pair) for pair in state["merges"])
    return (
        RegexTokenizer(merges, pattern=state["pattern"])
        if kind == "regex"
        else ByteTokenizer(merges)
    )


def save_tokenizer(path: Path, tokenizer) -> None:
    # Publication uses the shared durable writer; identity does not load models.
    from .serving_benchmark import _publish

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    _publish(
        path,
        {
            "schema_version": 1,
            "state": tokenizer_state(tokenizer),
            "sha256": tokenizer_fingerprint(tokenizer),
        },
    )


def load_tokenizer(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TokenizerError("duplicate tokenizer checkpoint field")
            result[key] = value
        return result

    try:
        with path.open("rb") as stream:
            data = stream.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise TokenizerError("tokenizer checkpoint exceeds 2 MiB")
        payload = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TokenizerError("unable to read tokenizer checkpoint") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "state",
        "sha256",
    }:
        raise TokenizerError("tokenizer checkpoint fields are invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise TokenizerError("unsupported tokenizer checkpoint version")
    tokenizer = _restore(payload["state"])
    if tokenizer_fingerprint(tokenizer) != payload["sha256"]:
        raise TokenizerError("tokenizer checkpoint fingerprint mismatch")
    return tokenizer
