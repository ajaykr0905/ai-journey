"""Bounded offline replay verification for hybrid-parallel reports."""

from __future__ import annotations

from collections.abc import Mapping

from .hybrid_parallel_report import assemble_hybrid_parallel_report
from .hybrid_parallel_schema import (
    REPORT_KIND,
    REPORT_SCHEMA,
    TOP_LEVEL_FIELDS,
    HybridParallelReportError,
    admit_report_inputs,
)
from .hybrid_parallel_seal import seal_hybrid_parallel_report


def verify_hybrid_parallel_report(
    report: dict[str, object],
) -> dict[str, object]:
    """Verify integrity and internal consistency, not execution authenticity."""

    if not isinstance(report, dict):
        raise TypeError("hybrid parallel report must be a dictionary")
    if set(report) != TOP_LEVEL_FIELDS:
        raise HybridParallelReportError(
            "hybrid parallel report fields do not match the canonical schema"
        )
    if report.get("schema") != REPORT_SCHEMA:
        raise HybridParallelReportError("unsupported hybrid parallel report schema")
    if report.get("kind") != REPORT_KIND:
        raise HybridParallelReportError("unexpected hybrid parallel report kind")
    expected_integrity = seal_hybrid_parallel_report(report)["integrity"]
    if report.get("integrity") != expected_integrity:
        raise HybridParallelReportError("hybrid parallel report integrity mismatch")
    inputs = report.get("inputs")
    observations = report.get("executed_observations")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(observations, Mapping)
        or set(observations) != {"pipeline_handoff"}
        or observations["pipeline_handoff"] != inputs.get("probe_observation")
    ):
        raise HybridParallelReportError(
            "recorded probe observation drifted from primary evidence"
        )
    try:
        (
            topology,
            tensor,
            token_ids,
            layer_count,
            schedule_policy,
            microbatch_count,
            assumptions,
            probe,
        ) = admit_report_inputs(inputs)
        expected = seal_hybrid_parallel_report(
            assemble_hybrid_parallel_report(
                topology=topology,
                tensor=tensor,
                token_ids=token_ids,
                layer_count=layer_count,
                schedule_policy=schedule_policy,
                microbatch_count=microbatch_count,
                communication_assumptions=assumptions,
                probe_observation=probe,
            )
        )
    except HybridParallelReportError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
        message = str(error)
        if "report admission bound" in message:
            raise HybridParallelReportError(message) from error
        raise HybridParallelReportError(
            "hybrid parallel report cannot be rebuilt from primary inputs"
        ) from error
    if report != expected:
        raise HybridParallelReportError(
            "hybrid parallel report derived fields drifted from primary inputs"
        )
    return report
