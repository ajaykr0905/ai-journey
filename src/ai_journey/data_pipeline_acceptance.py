"""Pinned public-data acceptance contract for reproducible Day 59 evidence."""

from hashlib import sha256
import json
from pathlib import Path

from .bpe import TokenizerError
from .data_pipeline_verify import replay_pipeline_artifacts, verify_pipeline_artifacts


FIELDS = {
    "schema_version",
    "source_sha256",
    "config_sha256",
    "report_sha256",
    "tokenizer_sha256",
    "input_documents",
    "quality_rejected",
    "exact_duplicates",
    "train_documents",
    "validation_documents",
    "total_tokens",
    "total_utf8_bytes",
}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TokenizerError("duplicate acceptance contract field")
        result[key] = value
    return result


def load_acceptance_contract(path: Path) -> dict:
    try:
        payload = path.read_bytes()
        if len(payload) > 64 * 1024:
            raise TokenizerError("acceptance contract exceeds 64 KiB")
        contract = json.loads(payload, object_pairs_hook=_unique)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise TokenizerError("unable to read acceptance contract") from exc
    if (
        not isinstance(contract, dict)
        or set(contract) != FIELDS
        or type(contract["schema_version"]) is not int
        or contract["schema_version"] != 1
    ):
        raise TokenizerError("acceptance contract fields are invalid")
    digest_fields = {name for name in FIELDS if name.endswith("sha256")}
    if any(
        not isinstance(contract[name], str) or len(contract[name]) != 64
        for name in digest_fields
    ) or any(
        type(contract[name]) is not int or contract[name] < 0
        for name in FIELDS - digest_fields - {"schema_version"}
    ):
        raise TokenizerError("acceptance contract values are invalid")
    return contract


def _sha256_bounded(path: Path, limit: int) -> str:
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as exc:
        raise TokenizerError("unable to read acceptance input") from exc
    if len(payload) > limit:
        raise TokenizerError("acceptance input exceeds its byte budget")
    return sha256(payload).hexdigest()


def verify_pipeline_acceptance(
    dataset: Path, contract_path: Path, config_path: Path, source_path: Path
) -> dict:
    """Bind replayed observations to exact source, config, and expected evidence."""
    contract = load_acceptance_contract(contract_path)
    report = verify_pipeline_artifacts(dataset)
    replay = replay_pipeline_artifacts(dataset)
    observed = {
        "schema_version": 1,
        "source_sha256": _sha256_bounded(source_path, 8 * 1024 * 1024),
        "config_sha256": _sha256_bounded(config_path, 64 * 1024),
        "report_sha256": report["report_sha256"],
        "tokenizer_sha256": replay["tokenizer_sha256"],
        "input_documents": report["counts"]["input_documents"],
        "quality_rejected": report["counts"]["quality_rejected"],
        "exact_duplicates": report["counts"]["exact_duplicates"],
        "train_documents": report["partitions"]["train"]["documents"],
        "validation_documents": report["partitions"]["validation"]["documents"],
        "total_tokens": replay["total_tokens"],
        "total_utf8_bytes": replay["total_utf8_bytes"],
    }
    if observed != contract:
        raise TokenizerError("dataset evidence does not match acceptance contract")
    return observed
