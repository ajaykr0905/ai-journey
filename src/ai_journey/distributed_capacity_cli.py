"""Build and verify analytical distributed-training capacity evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .distributed_capacity import ClusterTopology, LinkProfile
from .distributed_capacity_io import (
    MAX_DISTRIBUTED_CAPACITY_BYTES,
    load_distributed_capacity_report,
    save_distributed_capacity_report,
)
from .distributed_capacity_report import (
    DistributedCapacityPlan,
    DistributedCapacityReportError,
    build_distributed_capacity_report,
    seal_distributed_capacity_report,
)
from .resource_accounting import DecoderSpec


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DistributedCapacityReportError("duplicate capacity plan field")
        result[key] = value
    return result


def load_distributed_capacity_plan(path: Path) -> DistributedCapacityPlan:
    """Load a bounded JSON plan into validated model and topology contracts."""

    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_DISTRIBUTED_CAPACITY_BYTES + 1)
        if len(payload) > MAX_DISTRIBUTED_CAPACITY_BYTES:
            raise DistributedCapacityReportError("capacity plan exceeds 1 MiB")
        raw = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
        model = raw.pop("model")
        topology = raw.pop("topology")
        if not isinstance(model, dict) or not isinstance(topology, dict):
            raise TypeError("model and topology must be objects")
        intra_node = topology.pop("intra_node")
        inter_node = topology.pop("inter_node")
        if not isinstance(intra_node, dict) or not isinstance(inter_node, dict):
            raise TypeError("topology links must be objects")
        return DistributedCapacityPlan(
            model=DecoderSpec(**model),
            topology=ClusterTopology(
                **topology,
                intra_node=LinkProfile(**intra_node),
                inter_node=LinkProfile(**inter_node),
            ),
            **raw,
        )
    except DistributedCapacityReportError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DistributedCapacityReportError("unable to load capacity plan") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", allow_abbrev=False)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", allow_abbrev=False)
    verify.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        report = load_distributed_capacity_report(args.report)
        print(
            f"verified_capacity={report['integrity']['digest']} "
            f"world_size={report['topology']['world_size']} measured=false"
        )
        return 0
    if args.config.resolve() == args.output.resolve() or (
        args.config.exists()
        and args.output.exists()
        and args.config.samefile(args.output)
    ):
        raise DistributedCapacityReportError(
            "capacity output must not alias its protected config"
        )
    plan = load_distributed_capacity_plan(args.config)
    report = seal_distributed_capacity_report(build_distributed_capacity_report(plan))
    save_distributed_capacity_report(args.output, report)
    print(
        f"capacity={report['integrity']['digest']} "
        f"world_size={plan.topology.world_size} measured=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
