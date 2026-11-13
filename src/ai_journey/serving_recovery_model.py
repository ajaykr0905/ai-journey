"""Tiny deterministic character model served by the recovery lab server.

Integer bigram counts come from a public one-word-per-line fixture such as
``data/day-19-demo-names.txt``, using the Day 19 corpus contract of lowercase
ASCII letters. Each sampled character is selected with exact integer arithmetic
from a SHA-256 draw over the model digest, prompt, seed and step. A completion
is therefore a pure function of (prompt, seed, output budget, model digest) and
can be byte-compared across server lifetimes. Only the standard library is
imported so the server child cold-starts without numpy or torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from .serving_recovery_config import (
    ServingRecoveryError,
    canonical_json_bytes,
    canonical_sha256,
    require_int,
)

BOUNDARY = "."
MAX_SOURCE_BYTES = 64 * 1024
MAX_WORDS = 4096
MAX_PROMPT_CHARACTERS = 32
MAX_NEW_TOKENS = 32
MODEL_KIND = "serving-recovery-character-bigram"
_REQUEST_FIELDS = frozenset({"prompt", "seed", "max_new_tokens"})


def parse_corpus(source: bytes) -> tuple[str, ...]:
    """Return lowercase ASCII words from bounded UTF-8 one-word-per-line bytes."""

    if type(source) is not bytes:
        raise TypeError("source must be bytes")
    if not source or len(source) > MAX_SOURCE_BYTES:
        raise ServingRecoveryError("source must be non-empty and at most 64 KiB")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ServingRecoveryError("source must be valid UTF-8") from exc
    words = []
    for number, line in enumerate(text.splitlines(), start=1):
        word = line.strip()
        if not word:
            raise ServingRecoveryError(f"source line {number} is empty")
        if not (word.isascii() and word.isalpha() and word == word.lower()):
            raise ServingRecoveryError(
                f"source line {number} must contain lowercase ASCII letters only"
            )
        words.append(word)
    if len(words) > MAX_WORDS:
        raise ServingRecoveryError("source exceeds the word bound")
    return tuple(words)


@dataclass(frozen=True)
class ServingModel:
    """Boundary-first vocabulary and exact integer bigram transition counts."""

    vocabulary: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]
    model_sha256: str = field(init=False)
    row_totals: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vocabulary = self.vocabulary
        if (
            type(vocabulary) is not tuple
            or len(vocabulary) < 2
            or vocabulary[0] != BOUNDARY
        ):
            raise ServingRecoveryError("vocabulary must start with the boundary")
        letters = vocabulary[1:]
        if any(
            type(token) is not str or len(token) != 1 or not ("a" <= token <= "z")
            for token in letters
        ) or list(letters) != sorted(set(letters)):
            raise ServingRecoveryError(
                "vocabulary letters must be unique sorted lowercase ASCII"
            )
        size = len(vocabulary)
        if (
            type(self.counts) is not tuple
            or len(self.counts) != size
            or any(type(row) is not tuple or len(row) != size for row in self.counts)
        ):
            raise ServingRecoveryError("counts must be a square vocabulary table")
        for row in self.counts:
            for count in row:
                require_int(count, "transition count", 0, 2**31 - 1)
        if self.counts[0][0] != 0:
            raise ServingRecoveryError("empty words cannot be counted")
        totals = tuple(sum(row) for row in self.counts)
        if any(total == 0 for total in totals):
            raise ServingRecoveryError("every vocabulary row needs a transition")
        object.__setattr__(self, "row_totals", totals)
        object.__setattr__(self, "model_sha256", canonical_sha256(self.to_mapping()))

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical parameters that define the model digest."""

        return {
            "schema_version": 1,
            "kind": MODEL_KIND,
            "vocabulary": list(self.vocabulary),
            "counts": [list(row) for row in self.counts],
        }


def build_serving_model(source: bytes) -> ServingModel:
    """Count boundary-delimited character bigrams in a validated corpus."""

    words = parse_corpus(source)
    vocabulary = (BOUNDARY, *sorted({letter for word in words for letter in word}))
    index = {token: position for position, token in enumerate(vocabulary)}
    counts = [[0] * len(vocabulary) for _ in vocabulary]
    for word in words:
        tokens = [0, *(index[letter] for letter in word), 0]
        for previous, following in zip(tokens, tokens[1:]):
            counts[previous][following] += 1
    return ServingModel(vocabulary, tuple(tuple(row) for row in counts))


def model_from_mapping(payload: object) -> ServingModel:
    """Rebuild a model from its exact canonical mapping."""

    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "kind",
        "vocabulary",
        "counts",
    }:
        raise ServingRecoveryError("model fields do not match schema")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["kind"] != MODEL_KIND
    ):
        raise ServingRecoveryError("model schema is unsupported")
    vocabulary, counts = payload["vocabulary"], payload["counts"]
    if type(vocabulary) is not list or type(counts) is not list:
        raise ServingRecoveryError("model vocabulary and counts must be lists")
    if any(type(row) is not list for row in counts):
        raise ServingRecoveryError("model count rows must be lists")
    return ServingModel(tuple(vocabulary), tuple(tuple(row) for row in counts))


@dataclass(frozen=True)
class GenerationRequest:
    """One generation input; the idempotency key travels separately."""

    prompt: str
    seed: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        if (
            type(self.prompt) is not str
            or not 1 <= len(self.prompt) <= MAX_PROMPT_CHARACTERS
            or not (self.prompt.isascii() and self.prompt.isalpha())
            or self.prompt != self.prompt.lower()
        ):
            raise ServingRecoveryError("prompt must be 1 to 32 lowercase ASCII letters")
        require_int(self.seed, "seed", 0, 2**32 - 1)
        require_int(self.max_new_tokens, "max_new_tokens", 1, MAX_NEW_TOKENS)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
        }

    def body_bytes(self) -> bytes:
        """Return the canonical HTTP request body."""

        return canonical_json_bytes(self.to_mapping())

    def request_sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


def request_from_mapping(payload: object) -> GenerationRequest:
    """Parse a generation request without ignoring unknown fields."""

    if type(payload) is not dict or set(payload) != _REQUEST_FIELDS:
        raise ServingRecoveryError("generation request fields do not match schema")
    return GenerationRequest(**payload)


def _draw(model_sha256: str, request: GenerationRequest, step: int) -> int:
    material = canonical_json_bytes(
        {
            "model_sha256": model_sha256,
            "prompt": request.prompt,
            "seed": request.seed,
            "step": step,
        }
    )
    return int.from_bytes(sha256(material).digest()[:8], "big")


def generate_completion(
    model: ServingModel, request: GenerationRequest
) -> tuple[str, str]:
    """Return ``(output, finish_reason)`` using exact integer sampling."""

    if not isinstance(model, ServingModel):
        raise TypeError("model must be ServingModel")
    if not isinstance(request, GenerationRequest):
        raise TypeError("request must be GenerationRequest")
    index = {token: position for position, token in enumerate(model.vocabulary)}
    if any(letter not in index for letter in request.prompt):
        raise ServingRecoveryError("prompt contains letters outside the vocabulary")
    current = index[request.prompt[-1]]
    output = []
    for step in range(request.max_new_tokens):
        target = _draw(model.model_sha256, request, step) % model.row_totals[current]
        cumulative = 0
        for token, count in enumerate(model.counts[current]):
            cumulative += count
            if target < cumulative:
                break
        if token == 0:
            return "".join(output), "eos"
        output.append(model.vocabulary[token])
        current = token
    return "".join(output), "length"


def completion_body(model: ServingModel, request: GenerationRequest) -> bytes:
    """Return the exact canonical response body for one generation request."""

    output, finish_reason = generate_completion(model, request)
    return canonical_json_bytes(
        {
            **request.to_mapping(),
            "finish_reason": finish_reason,
            "model_sha256": model.model_sha256,
            "output": output,
        }
    )
