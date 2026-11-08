"""Independent structural and cryptographic HTML extraction verification."""

from hashlib import sha256
import json
import math
from pathlib import Path

from .bpe import TokenizerError
from .web_corpus import ExtractionPolicy
from .web_extraction_report import canonical_json_bytes
from .web_source import canonical_public_url


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TokenizerError("duplicate HTML extraction artifact field")
        result[key] = value
    return result


def _is_digest(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(payload: bytes, label: str):
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TokenizerError(f"invalid {label} JSON") from exc


def _read_bounded(path: Path, limit: int) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise TokenizerError("HTML extraction artifact must be a regular file")
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as exc:
        raise TokenizerError("unable to read HTML extraction artifact") from exc
    if len(payload) > limit:
        raise TokenizerError("HTML extraction artifact exceeds its byte budget")
    return payload


def _verify_pages(report: dict) -> tuple[set[str], int]:
    pages = report["pages"]
    fields = {
        "url",
        "payload_sha256",
        "payload_bytes",
        "encoding",
        "replacement_characters",
        "extracted_characters",
        "extracted_lines",
        "extraction_ratio",
        "accepted",
        "rejection_reason",
    }
    if not isinstance(pages, list) or not pages:
        raise TokenizerError("HTML extraction pages are invalid")
    accepted_urls = set()
    accepted_characters = 0
    seen_urls = set()
    for page in pages:
        if not isinstance(page, dict) or set(page) != fields:
            raise TokenizerError("HTML extraction page fields are invalid")
        url = page["url"]
        if canonical_public_url(url) != url or url in seen_urls:
            raise TokenizerError("HTML extraction page URL is invalid")
        seen_urls.add(url)
        integer_fields = (
            "payload_bytes",
            "replacement_characters",
            "extracted_characters",
            "extracted_lines",
        )
        if (
            not _is_digest(page["payload_sha256"])
            or any(
                type(page[name]) is not int or page[name] < 0 for name in integer_fields
            )
            or page["payload_bytes"] < 1
            or not isinstance(page["extraction_ratio"], float)
            or not math.isfinite(page["extraction_ratio"])
            or page["extraction_ratio"] < 0
            or type(page["accepted"]) is not bool
        ):
            raise TokenizerError("HTML extraction page values are invalid")
        if page["accepted"]:
            if (
                page["rejection_reason"] is not None
                or not isinstance(page["encoding"], str)
                or not page["encoding"]
                or page["extracted_characters"] < 1
                or page["extracted_lines"] < 1
            ):
                raise TokenizerError("accepted HTML page evidence is inconsistent")
            accepted_urls.add(url)
            accepted_characters += page["extracted_characters"]
        elif (
            not isinstance(page["rejection_reason"], str)
            or not page["rejection_reason"]
        ):
            raise TokenizerError("rejected HTML page lacks a reason")
    return accepted_urls, accepted_characters


def verify_extraction_artifacts(directory: Path) -> dict:
    """Verify exact inventory, report seal, page evidence, and corpus rows."""
    if not isinstance(directory, Path):
        raise TypeError("HTML extraction artifact root must be pathlib.Path")
    if not directory.is_dir() or directory.is_symlink():
        raise TokenizerError("HTML extraction artifact root must be a directory")
    entries = tuple(directory.iterdir())
    if {entry.name for entry in entries} != {"manifest.json", "corpus.jsonl"}:
        raise TokenizerError("HTML extraction artifact inventory is invalid")
    manifest = _load_json(
        _read_bounded(directory / "manifest.json", 8 * 1024 * 1024), "manifest"
    )
    fields = {
        "schema_version",
        "snapshot_sha256",
        "policy",
        "counts",
        "pages",
        "corpus",
        "evidence_boundary",
        "report_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != fields
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not _is_digest(manifest["snapshot_sha256"])
        or not _is_digest(manifest["report_sha256"])
        or not isinstance(manifest["evidence_boundary"], str)
        or not manifest["evidence_boundary"]
    ):
        raise TokenizerError("HTML extraction manifest fields are invalid")
    unsealed = {key: value for key, value in manifest.items() if key != "report_sha256"}
    if sha256(canonical_json_bytes(unsealed)).hexdigest() != manifest["report_sha256"]:
        raise TokenizerError("HTML extraction report seal mismatch")
    try:
        policy = ExtractionPolicy(**manifest["policy"])
    except (TypeError, TokenizerError) as exc:
        raise TokenizerError("HTML extraction manifest policy is invalid") from exc
    if set(manifest["policy"]) != {
        "minimum_pages",
        "minimum_text_characters",
        "maximum_replacement_fraction",
    }:
        raise TokenizerError("HTML extraction manifest policy fields are invalid")
    accepted_urls, accepted_characters = _verify_pages(manifest)
    counts = manifest["counts"]
    if (
        not isinstance(counts, dict)
        or set(counts)
        != {
            "input_pages",
            "accepted_pages",
            "rejected_pages",
            "extracted_characters",
        }
        or any(type(value) is not int or value < 0 for value in counts.values())
        or counts["input_pages"] != len(manifest["pages"])
        or counts["input_pages"] < policy.minimum_pages
        or counts["accepted_pages"] != len(accepted_urls)
        or counts["rejected_pages"] != len(manifest["pages"]) - len(accepted_urls)
        or counts["extracted_characters"] != accepted_characters
    ):
        raise TokenizerError("HTML extraction manifest counts are invalid")
    corpus = manifest["corpus"]
    if (
        not isinstance(corpus, dict)
        or set(corpus) != {"path", "documents", "artifact_bytes", "sha256"}
        or corpus["path"] != "corpus.jsonl"
        or type(corpus["documents"]) is not int
        or corpus["documents"] != len(accepted_urls)
        or type(corpus["artifact_bytes"]) is not int
        or not 1 <= corpus["artifact_bytes"] <= 64 * 1024 * 1024
        or not _is_digest(corpus["sha256"])
    ):
        raise TokenizerError("HTML extraction corpus descriptor is invalid")
    payload = _read_bounded(directory / "corpus.jsonl", corpus["artifact_bytes"])
    if (
        len(payload) != corpus["artifact_bytes"]
        or sha256(payload).hexdigest() != corpus["sha256"]
        or not payload.endswith(b"\n")
    ):
        raise TokenizerError("HTML extraction corpus identity mismatch")
    rows = [_load_json(line, "corpus row") for line in payload.splitlines()]
    row_fields = {"document_id", "source", "text", "text_sha256"}
    sources = set()
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != row_fields:
            raise TokenizerError("HTML extraction corpus row fields are invalid")
        source = row["source"]
        expected_id = f"web-{sha256(source.encode('utf-8')).hexdigest()[:24]}"
        if (
            source not in accepted_urls
            or source in sources
            or row["document_id"] != expected_id
            or row["document_id"] in identities
            or not isinstance(row["text"], str)
            or not row["text"]
            or not _is_digest(row["text_sha256"])
            or sha256(row["text"].encode("utf-8")).hexdigest() != row["text_sha256"]
        ):
            raise TokenizerError("HTML extraction corpus row values are invalid")
        sources.add(source)
        identities.add(row["document_id"])
    if len(rows) != corpus["documents"] or sources != accepted_urls:
        raise TokenizerError("HTML extraction corpus coverage mismatch")
    return manifest
