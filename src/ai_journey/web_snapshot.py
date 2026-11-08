"""Strict JSONL loader for immutable, replayable public HTML snapshots."""

import base64
import binascii
from hashlib import sha256
import json
from pathlib import Path

from .bpe import TokenizerError
from .web_source import WebPage


MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
MAX_SNAPSHOT_PAGES = 1_000


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TokenizerError("duplicate web snapshot field")
        result[key] = value
    return result


def _decode_row(payload: bytes, index: int) -> WebPage:
    try:
        row = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TokenizerError(f"invalid web snapshot JSON on row {index}") from exc
    if not isinstance(row, dict) or set(row) != {
        "url",
        "content_type",
        "payload_base64",
        "payload_sha256",
    }:
        raise TokenizerError("web snapshot row fields are invalid")
    if not isinstance(row["payload_base64"], str):
        raise TokenizerError("web snapshot payload encoding is invalid")
    digest = row["payload_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise TokenizerError("web snapshot payload digest is invalid")
    try:
        page_bytes = base64.b64decode(row["payload_base64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise TokenizerError("web snapshot payload is not canonical base64") from exc
    if base64.b64encode(page_bytes).decode("ascii") != row["payload_base64"]:
        raise TokenizerError("web snapshot payload is not canonical base64")
    if sha256(page_bytes).hexdigest() != digest:
        raise TokenizerError("web snapshot payload digest mismatch")
    return WebPage(row["url"], row["content_type"], page_bytes)


def load_web_snapshot(path: Path) -> tuple[WebPage, ...]:
    """Load a bounded regular file and verify every page before returning any."""
    if not isinstance(path, Path):
        raise TypeError("web snapshot path must be pathlib.Path")
    if not path.is_file() or path.is_symlink():
        raise TokenizerError("web snapshot must be a regular nonsymlink file")
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_SNAPSHOT_BYTES + 1)
    except OSError as exc:
        raise TokenizerError("unable to read web snapshot") from exc
    if len(payload) > MAX_SNAPSHOT_BYTES:
        raise TokenizerError("web snapshot exceeds its byte budget")
    if not payload or not payload.endswith(b"\n"):
        raise TokenizerError("web snapshot must be nonempty newline-terminated JSONL")
    lines = payload.splitlines()
    if not 1 <= len(lines) <= MAX_SNAPSHOT_PAGES:
        raise TokenizerError("web snapshot page count is outside its budget")
    pages = tuple(_decode_row(line, index) for index, line in enumerate(lines, 1))
    urls = [page.url for page in pages]
    if len(set(urls)) != len(urls):
        raise TokenizerError("web snapshot contains duplicate page URLs")
    return pages
