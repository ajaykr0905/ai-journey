"""Normalize complementary model evaluations into one evidence vocabulary."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

from .ddp_evaluation import Document, evaluate_documents
from .gpt2_training import gpt2_model_fingerprint
from .hellaswag import verify_evidence as verify_hellaswag_evidence
from .lambada import verify_evidence as verify_lambada_evidence


class EvaluationSuiteError(ValueError):
    """Raised when evaluation suite inputs are incomplete or inconsistent."""


SUITE_LIMITATIONS = [
    "The suite reports three bounded evaluations and is not a general capability score.",
    "Scores are comparable only when model identity and each dataset contract match.",
    "CPU execution does not establish GPU, distributed, or production performance.",
]


@dataclass(frozen=True)
class EvaluationResult:
    """One normalized primary metric with its model and data identities."""

    name: str
    model_sha256: str
    dataset_sha256: str
    primary_metric: str
    value: float
    higher_is_better: bool
    sample_count: int
    eligible: bool
    limitation: str

    def __post_init__(self) -> None:
        if self.name not in {
            "heldout_likelihood",
            "hellaswag_validation",
            "lambada_last_word",
        }:
            raise EvaluationSuiteError("evaluation name is not supported")
        for name in ("model_sha256", "dataset_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise EvaluationSuiteError(f"{name} must be a SHA-256 digest")
        if (
            not isinstance(self.primary_metric, str)
            or not self.primary_metric
            or isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
            or type(self.higher_is_better) is not bool
            or isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
            or type(self.eligible) is not bool
            or not isinstance(self.limitation, str)
            or not self.limitation
        ):
            raise EvaluationSuiteError("evaluation result fields are invalid")

    def to_mapping(self) -> dict:
        """Return the validated result as the plain mapping the suite seals."""
        return asdict(self)


def _document_digest(documents) -> str:
    payload = [
        {"identifier": document.identifier, "sha256": document.digest}
        for document in documents
    ]
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluate_heldout_likelihood(model, documents) -> EvaluationResult:
    """Normalize exact document likelihood as a lower-is-better evaluation."""
    documents = tuple(documents)
    if not documents or any(not isinstance(item, Document) for item in documents):
        raise EvaluationSuiteError("held-out evaluation requires documents")
    metrics = evaluate_documents(model, documents)
    return EvaluationResult(
        name="heldout_likelihood",
        model_sha256=gpt2_model_fingerprint(model),
        dataset_sha256=_document_digest(documents),
        primary_metric="bits_per_byte",
        value=metrics["bits_per_byte"],
        higher_is_better=False,
        sample_count=metrics["tokens"],
        eligible=True,
        limitation="Byte-level held-out likelihood does not measure task accuracy.",
    )


def result_from_hellaswag(evidence: dict) -> EvaluationResult:
    """Normalize verified HellaSwag accuracy without broadening its slice claim."""
    evidence = verify_hellaswag_evidence(evidence)
    run = evidence["run"]
    return EvaluationResult(
        name="hellaswag_validation",
        model_sha256=run["model_sha256"],
        dataset_sha256=run["records_sha256"],
        primary_metric="accuracy",
        value=run["evaluation"]["accuracy"],
        higher_is_better=True,
        sample_count=run["evaluation"]["examples"],
        eligible=True,
        limitation=evidence["limitations"][0],
    )


def result_from_lambada(evidence: dict) -> EvaluationResult:
    """Normalize verified exact-match accuracy and preserve contamination status."""
    evidence = verify_lambada_evidence(evidence)
    run = evidence["run"]
    return EvaluationResult(
        name="lambada_last_word",
        model_sha256=run["model_sha256"],
        dataset_sha256=run["examples_sha256"],
        primary_metric="exact_match_accuracy",
        value=run["evaluation"]["exact_match_accuracy"],
        higher_is_better=True,
        sample_count=run["evaluation"]["examples"],
        eligible=run["contamination"]["passed"],
        limitation=evidence["limitations"][0],
    )


def _suite_digest(payload: dict) -> str:
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def build_evaluation_suite(results) -> dict:
    """Seal one model's held-out, HellaSwag and LAMBADA results together."""
    results = tuple(results)
    expected = {
        "heldout_likelihood",
        "hellaswag_validation",
        "lambada_last_word",
    }
    if (
        len(results) != 3
        or any(not isinstance(result, EvaluationResult) for result in results)
        or {result.name for result in results} != expected
    ):
        raise EvaluationSuiteError("suite requires exactly three named evaluations")
    models = {result.model_sha256 for result in results}
    if len(models) != 1:
        raise EvaluationSuiteError("suite evaluations must share one model identity")
    ordered = sorted(results, key=lambda result: result.name)
    complete = all(result.eligible for result in ordered)
    payload = {
        "schema_version": 1,
        "model_sha256": next(iter(models)),
        "evaluations": [result.to_mapping() for result in ordered],
        "claim_status": (
            "complete_bounded_three_axis"
            if complete
            else "blocked_ineligible_evaluation"
        ),
        "limitations": SUITE_LIMITATIONS.copy(),
    }
    payload["evidence_sha256"] = _suite_digest(payload)
    return verify_evaluation_suite(payload)


def verify_evaluation_suite(payload: dict) -> dict:
    """Verify suite integrity, completeness, model identity and claim status."""
    fields = {
        "schema_version",
        "model_sha256",
        "evaluations",
        "claim_status",
        "limitations",
        "evidence_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or payload["schema_version"] != 1
        or payload["limitations"] != SUITE_LIMITATIONS
    ):
        raise EvaluationSuiteError("evaluation suite schema does not match")
    unsigned = {
        key: value for key, value in payload.items() if key != "evidence_sha256"
    }
    if _suite_digest(unsigned) != payload["evidence_sha256"]:
        raise EvaluationSuiteError("evaluation suite digest does not match")
    if not isinstance(payload["evaluations"], list):
        raise EvaluationSuiteError("evaluation suite results must be a list")
    try:
        results = [EvaluationResult(**result) for result in payload["evaluations"]]
    except (TypeError, EvaluationSuiteError) as error:
        raise EvaluationSuiteError("evaluation suite results are invalid") from error
    expected = {
        "heldout_likelihood",
        "hellaswag_validation",
        "lambada_last_word",
    }
    if (
        len(results) != 3
        or {result.name for result in results} != expected
        or [result.name for result in results] != sorted(expected)
        or {result.model_sha256 for result in results} != {payload["model_sha256"]}
    ):
        raise EvaluationSuiteError("evaluation suite coverage does not match")
    complete = all(result.eligible for result in results)
    expected_status = (
        "complete_bounded_three_axis" if complete else "blocked_ineligible_evaluation"
    )
    if payload["claim_status"] != expected_status:
        raise EvaluationSuiteError("evaluation suite claim status is inconsistent")
    return payload
