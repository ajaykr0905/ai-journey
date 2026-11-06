"""Evaluate and verify a bounded LAMBADA test slice."""

import argparse
import json
from pathlib import Path

from .evaluation_model import load_evaluation_model
from .hellaswag_io import (
    parse_evaluation_json,
    read_evaluation_bytes,
    validate_evaluation_output,
)
from .lambada import LambadaProtocol, compare_evaluations
from .lambada import run_evaluation, seal_evidence
from .lambada_evidence import read_evidence, write_evidence


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
                for field in ("protocol_sha256", "sample_sha256", "examples_sha256")
            ):
                raise ValueError("comparison evidence does not cover identical rows")
            comparison = {
                "baseline_evidence_sha256": baseline["evidence_sha256"],
                "candidate_evidence_sha256": candidate["evidence_sha256"],
                "baseline_model_sha256": left["model_sha256"],
                "candidate_model_sha256": right["model_sha256"],
                "protocol_sha256": left["protocol_sha256"],
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
            response = read_evaluation_bytes(
                arguments.dataset_response,
                maximum=2 * 1024 * 1024,
                name="dataset response",
            )
            protocol = LambadaProtocol.from_mapping(
                parse_evaluation_json(
                    read_evaluation_bytes(
                        arguments.protocol,
                        maximum=128 * 1024,
                        name="protocol input",
                    ),
                    name="protocol input",
                )
            )
            model, training_source = load_evaluation_model(
                arguments.source,
                arguments.spec,
                arguments.checkpoint,
                checkpoint_format=arguments.checkpoint_format,
                include_source=True,
            )
            evidence = seal_evidence(
                run_evaluation(model, response, training_source, protocol), protocol
            )
            write_evidence(arguments.output, evidence)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    receipt = {
        "evidence_sha256": evidence["evidence_sha256"],
        "examples": evidence["run"]["evaluation"]["examples"],
        "exact_match_accuracy": evidence["run"]["evaluation"]["exact_match_accuracy"],
        "mean_nll": evidence["run"]["evaluation"]["mean_nll"],
        "contamination_passed": evidence["run"]["contamination"]["passed"],
        "model_sha256": evidence["run"]["model_sha256"],
    }
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
