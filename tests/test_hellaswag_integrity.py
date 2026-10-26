"""Resealed reports must retain semantic consistency with their scored rows."""

from copy import deepcopy

import pytest

from ai_journey.hellaswag import (
    HellaSwagError,
    _payload_digest,
    summarize_scores,
    verify_evidence,
)
from test_hellaswag_evidence import evidence_fixture


@pytest.mark.parametrize(
    "corruption",
    [
        "outcome",
        "prediction",
        "confidence",
        "normalization",
        "token_count",
        "rows_digest",
        "row_slice",
        "model_digest",
    ],
)
def test_resealed_inconsistent_scoring_is_rejected(corruption):
    evidence = deepcopy(evidence_fixture())
    run = evidence["run"]
    row = run["evaluation"]["records"][0]
    if corruption == "outcome":
        row["correct"] = False
    elif corruption == "prediction":
        row.update(prediction=1, correct=False)
    elif corruption == "confidence":
        row["confidence"] = 0.99
    elif corruption == "normalization":
        row["normalization"] = "prompt_and_completion"
    elif corruption == "token_count":
        row["scores"][0]["tokens"] += 1
    elif corruption == "rows_digest":
        run["records_sha256"] = "0" * 64
    elif corruption == "row_slice":
        row["row_index"] += 1
    else:
        run["model_sha256"] = "invalid-model-digest"
    protocol = evidence["protocol"]
    summary = summarize_scores(
        [row],
        bootstrap_samples=protocol["bootstrap_samples"],
        bootstrap_seed=protocol["bootstrap_seed"],
    )
    run["evaluation"] = {**summary, "records": [row]}
    evidence["evidence_sha256"] = _payload_digest(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    with pytest.raises(HellaSwagError):
        verify_evidence(evidence)
