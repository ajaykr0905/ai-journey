"""Canonical manifest and corpus payloads for HTML extraction evidence."""

from dataclasses import asdict
from hashlib import sha256
import json
import math

from .bpe import TokenizerError
from .web_corpus import ExtractionPolicy, WebCorpusResult


def canonical_json_bytes(value) -> bytes:
    """The one encoding of corpus rows, the manifest and its seal.

    Values JSON cannot represent exactly, such as NaN, raise TokenizerError.
    """
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise TokenizerError("HTML extraction evidence is not canonical JSON") from exc


def corpus_payload(result: WebCorpusResult) -> bytes:
    """Accepted documents as canonical JSONL with text digests: the hashed bytes."""
    if not isinstance(result, WebCorpusResult):
        raise TypeError("corpus payload requires a WebCorpusResult")
    return b"".join(
        canonical_json_bytes(
            {
                "document_id": document.document_id,
                "source": document.source,
                "text": document.text,
                "text_sha256": sha256(document.text.encode("utf-8")).hexdigest(),
            }
        )
        for document in result.documents
    )


def build_extraction_report(
    result: WebCorpusResult,
    policy: ExtractionPolicy,
    *,
    snapshot_sha256: str,
) -> dict:
    """Seal inputs, thresholds, page outcomes, and extracted corpus identity."""
    if not isinstance(result, WebCorpusResult) or not isinstance(
        policy, ExtractionPolicy
    ):
        raise TypeError("extraction report requires a result and policy")
    if (
        not isinstance(snapshot_sha256, str)
        or len(snapshot_sha256) != 64
        or any(character not in "0123456789abcdef" for character in snapshot_sha256)
    ):
        raise TokenizerError("web snapshot digest is invalid")
    if any(not math.isfinite(row.extraction_ratio) for row in result.observations):
        raise TokenizerError("page extraction ratio must be finite")
    payload = corpus_payload(result)
    report = {
        "schema_version": 1,
        "snapshot_sha256": snapshot_sha256,
        "policy": asdict(policy),
        "counts": {
            "input_pages": len(result.observations),
            "accepted_pages": len(result.documents),
            "rejected_pages": result.rejected_pages,
            "extracted_characters": sum(
                row.extracted_characters for row in result.observations if row.accepted
            ),
        },
        "pages": [asdict(row) for row in result.observations],
        "corpus": {
            "path": "corpus.jsonl",
            "documents": len(result.documents),
            "artifact_bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        },
        "evidence_boundary": (
            "offline extraction from caller-supplied public HTML snapshot; "
            "no claim of live crawling, Common Crawl scale, or learner completion"
        ),
    }
    report["report_sha256"] = sha256(canonical_json_bytes(report)).hexdigest()
    return report
