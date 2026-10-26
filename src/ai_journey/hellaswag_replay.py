"""Repeat an evaluation only when its model and CPU runtime identity agree."""

import torch

from .gpt2_training import gpt2_model_fingerprint
from .hellaswag import (
    HellaSwagError,
    HellaSwagProtocol,
    evaluation_runtime,
    run_evaluation,
    seal_evidence,
    verify_evidence,
)
from .training_models import model_config_mapping


def verify_evaluation_replay(model, response: bytes, evidence: dict) -> dict:
    """Rerun pinned scoring and require every regenerated evidence field to match."""
    expected = verify_evidence(evidence)
    if expected["schema_version"] != 2 or expected["run"]["schema_version"] != 2:
        raise HellaSwagError(
            "legacy evidence lacks the runtime metadata for exact replay"
        )
    run = expected["run"]
    if (
        model_config_mapping(model.config) != run["model_config"]
        or gpt2_model_fingerprint(model) != run["model_sha256"]
    ):
        raise HellaSwagError("evaluation replay model identity does not match")
    dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if (
        len(dtypes) != 1
        or evaluation_runtime(
            model_dtype=next(iter(dtypes)), torch_threads=torch.get_num_threads()
        )
        != expected["runtime"]
    ):
        raise HellaSwagError(
            "evaluation replay runtime does not match the recorded CPU environment"
        )
    protocol = HellaSwagProtocol.from_mapping(expected["protocol"])
    actual = seal_evidence(run_evaluation(model, response, protocol), protocol)
    if actual != expected:
        raise HellaSwagError(
            "evaluation replay does not match the recorded candidate scores"
        )
    return actual
