"""Generate bounded hybrid-parallel planning and CPU/Gloo evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .activation_handoff import ActivationHandoffContract
from .hybrid_parallel_io import save_hybrid_parallel_report
from .hybrid_parallel_report import (
    HybridParallelReportError,
    build_hybrid_parallel_report,
    seal_hybrid_parallel_report,
)
from .hybrid_topology import HybridParallelTopology
from .parallel_communication import CommunicationAssumptions
from .pipeline_handoff_probe import (
    PipelineHandoffProbeError,
    run_pipeline_handoff_probe,
)
from .pipeline_partition import partition_transformer_layers
from .tensor_sharding import LogicalTensor


MAX_CLI_WORLD_SIZE = 4096
MAX_CLI_LAYER_COUNT = 4096
MAX_CLI_MICROBATCH_COUNT = 4096


def _bounded_positive(value: int, name: str, maximum: int) -> int:
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return value


def _integer_list(value: str, *, name: str, maximum_items: int) -> tuple[int, ...]:
    parts = value.split(",")
    if any(part == "" or part.strip() != part for part in parts):
        raise ValueError(f"{name} must be a comma-separated integer list")
    try:
        items = tuple(int(item) for item in parts)
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not items or len(items) > maximum_items:
        raise ValueError(f"{name} must contain between 1 and {maximum_items} items")
    return items


def _positive_shape(value: str, *, name: str) -> tuple[int, ...]:
    shape = _integer_list(value, name=name, maximum_items=8)
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"{name} dimensions must be positive")
    return shape


def _payload(value: str) -> bytes:
    try:
        payload = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(
            "payload-hex must contain an even-length hexadecimal value"
        ) from error
    if not payload:
        raise ValueError("payload-hex must not be empty")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--data-parallel-size", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, required=True)
    parser.add_argument("--sequence-parallel-size", type=int, required=True)
    parser.add_argument("--tensor-shape", required=True)
    parser.add_argument("--tensor-values", required=True)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--schedule-policy", choices=("gpipe", "1f1b"), required=True)
    parser.add_argument("--microbatch-count", type=int, required=True)
    parser.add_argument("--element-size-bytes", type=int, required=True)
    parser.add_argument("--pipeline-payload-bytes", type=int, required=True)
    parser.add_argument("--source-rank", type=int, required=True)
    parser.add_argument("--destination-rank", type=int, required=True)
    parser.add_argument("--activation-shape", required=True)
    parser.add_argument(
        "--activation-dtype",
        choices=("bfloat16", "float16", "float32", "float64"),
        required=True,
    )
    parser.add_argument("--payload-hex", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _generate(args: argparse.Namespace) -> dict[str, object]:
    world_size = _bounded_positive(args.world_size, "world_size", MAX_CLI_WORLD_SIZE)
    dimensions = {
        "data_parallel_size": args.data_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "sequence_parallel_size": args.sequence_parallel_size,
    }
    for name, value in dimensions.items():
        _bounded_positive(value, name, MAX_CLI_WORLD_SIZE)
    topology = HybridParallelTopology(world_size=world_size, **dimensions)
    if topology.pipeline_parallel_size < 2:
        raise ValueError("pipeline_parallel_size must be at least 2 for the probe")

    tensor_shape = _positive_shape(args.tensor_shape, name="tensor-shape")
    tensor_values = _integer_list(
        args.tensor_values, name="tensor-values", maximum_items=1_000_000
    )
    tensor = LogicalTensor(tensor_shape, tensor_values)
    token_ids = _integer_list(args.token_ids, name="token-ids", maximum_items=1_000_000)
    layer_count = _bounded_positive(
        args.layer_count, "layer_count", MAX_CLI_LAYER_COUNT
    )
    microbatch_count = _bounded_positive(
        args.microbatch_count, "microbatch_count", MAX_CLI_MICROBATCH_COUNT
    )
    partition = partition_transformer_layers(topology, layer_count=layer_count)

    payload = _payload(args.payload_hex)
    activation_shape = _positive_shape(args.activation_shape, name="activation-shape")
    if args.pipeline_payload_bytes != len(payload):
        raise ValueError(
            "pipeline_payload_bytes must equal the explicit payload length"
        )
    contract = ActivationHandoffContract.admit(
        topology=topology,
        partition=partition,
        source_rank=args.source_rank,
        destination_rank=args.destination_rank,
        tensor_shape=activation_shape,
        dtype=args.activation_dtype,
        payload=payload,
    )
    assumptions = CommunicationAssumptions(
        element_size_bytes=args.element_size_bytes,
        pipeline_payload_bytes=args.pipeline_payload_bytes,
    )
    probe = run_pipeline_handoff_probe(
        topology=topology,
        partition=partition,
        contract=contract,
        payload=payload,
    )
    return seal_hybrid_parallel_report(
        build_hybrid_parallel_report(
            topology=topology,
            tensor=tensor,
            token_ids=token_ids,
            layer_count=layer_count,
            schedule_policy=args.schedule_policy,
            microbatch_count=microbatch_count,
            communication_assumptions=assumptions,
            probe_result=probe,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output must be a new path")
        report = _generate(args)
        save_hybrid_parallel_report(args.output, report)
    except (
        HybridParallelReportError,
        PipelineHandoffProbeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"hybrid parallel generation failed: {error}", file=sys.stderr)
        return 2
    observation = report["executed_observations"]["pipeline_handoff"]
    print(
        f"hybrid_parallel={report['integrity']['digest']} "
        f"backend={observation['backend']} device={observation['device']} "
        "performance=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
