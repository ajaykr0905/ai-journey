"""Evaluate and verify a bounded HellaSwag validation slice."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from .ddp_pipeline import load_serving_model
from .gpt2_training import load_training_spec
from .hellaswag import HellaSwagError, HellaSwagProtocol, compare_evaluations
from .hellaswag import run_evaluation, seal_evidence
from .hellaswag_evidence import read_evidence, write_evidence
from .hellaswag_io import (
    read_evaluation_bytes,
    parse_evaluation_json,
    validate_evaluation_output,
)
from .training_artifact import load_training_model_checkpoint
from .training_models import build_training_model, expected_training_parameters


def _read_json(path: Path, *, maximum: int) -> dict:
    return parse_evaluation_json(
        read_evaluation_bytes(path, maximum=maximum, name=path.name), name=path.name
    )


def _load_model(
    source: Path,
    spec_path: Path,
    checkpoint: Path | None,
    *,
    checkpoint_format: str = "distributed",
):
    spec = load_training_spec(spec_path)
    source_bytes = read_evaluation_bytes(
        source, maximum=16 * 1024 * 1024, name="training source"
    )
    if sha256(source_bytes).hexdigest() != spec.source.sha256:
        raise HellaSwagError("model source does not match the training specification")
    if expected_training_parameters(spec.model) > 1_000_000:
        raise HellaSwagError("evaluation model exceeds the CPU parameter budget")
    if checkpoint is not None:
        if checkpoint_format == "local":
            return load_training_model_checkpoint(
                checkpoint, spec=spec, source=source_bytes, max_parameters=1_000_000
            )
        if checkpoint_format != "distributed":
            raise HellaSwagError("unsupported evaluation checkpoint format")
        return load_serving_model(checkpoint, spec, source_bytes)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.training.seed)
        return (
            build_training_model(spec.model, max_parameters=1_000_000).double().eval()
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--dataset-response", type=Path, required=True)
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--source", type=Path, required=True)
    evaluate.add_argument("--spec", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument(
        "--checkpoint-format", choices=("distributed", "local"), default="distributed"
    )
    evaluate.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "compare":
            baseline = read_evidence(arguments.baseline)
            candidate = read_evidence(arguments.candidate)
            left, right = baseline["run"], candidate["run"]
            if any(
                left[field] != right[field]
                for field in ("protocol_sha256", "sample_sha256", "records_sha256")
            ):
                raise HellaSwagError(
                    "comparison evidence does not cover the same protocol rows"
                )
            comparison = {
                "baseline_evidence_sha256": baseline["evidence_sha256"],
                "candidate_evidence_sha256": candidate["evidence_sha256"],
                "baseline_model_sha256": left["model_sha256"],
                "candidate_model_sha256": right["model_sha256"],
                "protocol_sha256": left["protocol_sha256"],
                "records_sha256": left["records_sha256"],
                "metrics": compare_evaluations(left["evaluation"], right["evaluation"]),
            }
            print(json.dumps(comparison, sort_keys=True, allow_nan=False))
            return 0
        if arguments.command == "verify":
            evidence = read_evidence(arguments.report)
        else:
            inputs = (
                arguments.dataset_response,
                arguments.protocol,
                arguments.source,
                arguments.spec,
            )
            if arguments.checkpoint is not None:
                inputs += (arguments.checkpoint,)
            validate_evaluation_output(arguments.output, inputs)
            protocol = HellaSwagProtocol.from_mapping(
                _read_json(arguments.protocol, maximum=128 * 1024)
            )
            response = read_evaluation_bytes(
                arguments.dataset_response,
                maximum=2 * 1024 * 1024,
                name="dataset response",
            )
            model = _load_model(
                arguments.source,
                arguments.spec,
                arguments.checkpoint,
                checkpoint_format=arguments.checkpoint_format,
            )
            evidence = seal_evidence(
                run_evaluation(model, response, protocol), protocol
            )
            write_evidence(arguments.output, evidence)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    receipt = {
        "evidence_sha256": evidence["evidence_sha256"],
        "examples": evidence["run"]["evaluation"]["examples"],
        "accuracy": evidence["run"]["evaluation"]["accuracy"],
        "model_sha256": evidence["run"]["model_sha256"],
    }
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
