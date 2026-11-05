"""Build and verify bounded scaling experiment evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .scaling_experiment_analysis import ScalingAcceptancePolicy
from .scaling_experiment_identity import ExperimentInputs
from .scaling_experiment_io import load_experiment_report, save_experiment_report
from .scaling_experiment_observation import ScalingObservation
from .scaling_experiment_protocol import ExperimentPoint, ScalingExperimentProtocol
from .scaling_experiment_report import build_experiment_report
from .scaling_experiment_verify import seal_experiment_report
from .scaling_laws import ScalingLawError

MAX_EXPERIMENT_PROTOCOL_BYTES = 1024 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ScalingLawError("duplicate scaling experiment protocol field")
        result[key] = value
    return result


def _load_protocol(path: Path):
    try:
        payload = path.read_bytes()
        if len(payload) > MAX_EXPERIMENT_PROTOCOL_BYTES:
            raise ScalingLawError("scaling experiment protocol exceeds 1 MiB")
        data = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
        protocol_data = data["protocol"]
        protocol = ScalingExperimentProtocol(
            name=protocol_data["name"],
            parameter_counts=tuple(protocol_data["parameter_counts"]),
            train_token_counts=tuple(protocol_data["train_token_counts"]),
            seeds=tuple(protocol_data["seeds"]),
        )
        inputs = ExperimentInputs(**data["inputs"])
        observations = tuple(
            ScalingObservation(
                run_id=row["run_id"],
                point=ExperimentPoint(**row["point"]),
                validation_loss=row["validation_loss"],
                trained_tokens=row["trained_tokens"],
                optimizer_steps=row["optimizer_steps"],
                elapsed_seconds=row["elapsed_seconds"],
            )
            for row in data["observations"]
        )
        return (
            protocol,
            inputs,
            observations,
            data["target_compute"],
            ScalingAcceptancePolicy(**data["policy"]),
        )
    except ScalingLawError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ScalingLawError("unable to read scaling experiment protocol") from error


def _aliases(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and verify supplied scaling observations without authenticating execution."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a sealed experiment report")
    build.add_argument("--protocol", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify a sealed experiment report")
    verify.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        report = load_experiment_report(args.report)
        print(
            f"verified_scaling_experiment={report['integrity']['digest']} "
            f"runs={len(report['observations'])} "
            f"admitted={str(report['acceptance']['admitted']).lower()} "
            "execution_authenticated=false"
        )
        return 0
    if _aliases(args.protocol, args.output):
        raise ScalingLawError("experiment output must not alias its protocol")
    protocol, inputs, observations, target_compute, policy = _load_protocol(
        args.protocol
    )
    report = seal_experiment_report(
        build_experiment_report(
            protocol,
            inputs,
            observations,
            target_compute=target_compute,
            policy=policy,
        )
    )
    save_experiment_report(args.output, report)
    print(
        f"scaling_experiment={report['integrity']['digest']} "
        f"runs={len(report['observations'])} "
        f"admitted={str(report['acceptance']['admitted']).lower()} "
        "execution_authenticated=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
