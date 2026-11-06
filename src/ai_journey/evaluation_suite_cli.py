"""Build and verify a three-axis model evaluation suite."""

import argparse
import json
from pathlib import Path

from .ddp_evaluation import documents_from_lines
from .evaluation_model import load_evaluation_model
from .evaluation_suite import (
    build_evaluation_suite,
    evaluate_heldout_likelihood,
    result_from_hellaswag,
    result_from_lambada,
)
from .evaluation_suite_evidence import read_suite, write_suite
from .hellaswag_evidence import read_evidence as read_hellaswag_evidence
from .hellaswag_io import read_evaluation_bytes, validate_evaluation_output
from .lambada_evidence import read_evidence as read_lambada_evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--checkpoint", type=Path)
    build.add_argument(
        "--checkpoint-format", choices=("distributed", "local"), default="distributed"
    )
    build.add_argument("--heldout-data", type=Path, required=True)
    build.add_argument("--hellaswag-report", type=Path, required=True)
    build.add_argument("--lambada-report", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "verify":
            suite = read_suite(arguments.report)
        else:
            inputs = (
                arguments.source,
                arguments.spec,
                arguments.heldout_data,
                arguments.hellaswag_report,
                arguments.lambada_report,
            )
            if arguments.checkpoint is not None:
                inputs += (arguments.checkpoint,)
            validate_evaluation_output(arguments.output, inputs)
            model = load_evaluation_model(
                arguments.source,
                arguments.spec,
                arguments.checkpoint,
                checkpoint_format=arguments.checkpoint_format,
            )
            heldout = documents_from_lines(
                read_evaluation_bytes(
                    arguments.heldout_data,
                    maximum=16 * 1024 * 1024,
                    name="held-out data",
                )
            )
            suite = build_evaluation_suite(
                [
                    evaluate_heldout_likelihood(model, heldout),
                    result_from_hellaswag(
                        read_hellaswag_evidence(arguments.hellaswag_report)
                    ),
                    result_from_lambada(
                        read_lambada_evidence(arguments.lambada_report)
                    ),
                ]
            )
            write_suite(arguments.output, suite)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    receipt = {
        "evidence_sha256": suite["evidence_sha256"],
        "model_sha256": suite["model_sha256"],
        "claim_status": suite["claim_status"],
        "evaluations": {
            item["name"]: {
                "metric": item["primary_metric"],
                "value": item["value"],
                "sample_count": item["sample_count"],
            }
            for item in suite["evaluations"]
        },
    }
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
