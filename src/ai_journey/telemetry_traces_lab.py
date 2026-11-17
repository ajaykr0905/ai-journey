"""Resource conventions shared by every lab trace export.

Each exported run becomes one OTLP resource. Its attributes use OpenTelemetry
semantic-convention names where one applies (``service.name``,
``service.namespace``, ``deployment.environment.name``) and the ``ai_journey.``
namespace for lab facts: the run identity, the sealed report and trace-head
digests the spans were derived from, the lab's own evidence boundary and the
monotonic-to-epoch mapping of its timestamps. Every resource is labeled as
local CPU lab telemetry.
"""

from __future__ import annotations

from .telemetry_traces_ids import LABS
from .telemetry_traces_io import TelemetryError
from .telemetry_traces_otlp import (
    MONOTONIC_CLOCK,
    TIME_MAPPING_FORMULA,
    MonotonicEpochMapping,
    ResourceTelemetry,
    TelemetrySpan,
)
from .telemetry_traces_validate import (
    BOUNDARY_ATTRIBUTE,
    ENVIRONMENT_ATTRIBUTE,
    LOCAL_CPU_LAB,
)

SERVICE_NAMESPACE = "ai-journey"
SCOPE_NAME = "ai_journey.telemetry_traces"
SCOPE_VERSION = "1"
TELEMETRY_BOUNDARY = (
    "Local CPU lab telemetry: spans derived from sealed, independently verified "
    "recovery-lab evidence recorded on one CPU host. This is not production, "
    "GPU, CUDA, distributed or multi-host monitoring. Durations are measured "
    "monotonic-clock differences; absolute timestamps come from a declared "
    "anchor and are synthetic."
)


def lab_resource(
    *,
    lab: str,
    service_name: str,
    run_id: str,
    report_sha256: str,
    trace_head_sha256: str,
    lab_boundary: str,
    mapping: MonotonicEpochMapping,
    spans: tuple[TelemetrySpan, ...],
) -> ResourceTelemetry:
    """Describe one verified run as a labeled OTLP resource holding its spans."""

    if lab not in LABS:
        raise TelemetryError(f"lab must be one of {', '.join(LABS)}")
    if not isinstance(mapping, MonotonicEpochMapping):
        raise TypeError("mapping must be MonotonicEpochMapping")
    return ResourceTelemetry(
        attributes={
            "service.name": service_name,
            "service.namespace": SERVICE_NAMESPACE,
            ENVIRONMENT_ATTRIBUTE: LOCAL_CPU_LAB,
            BOUNDARY_ATTRIBUTE: TELEMETRY_BOUNDARY,
            "ai_journey.lab": lab,
            "ai_journey.run.id": run_id,
            "ai_journey.evidence.lab_boundary": lab_boundary,
            "ai_journey.evidence.report_sha256": report_sha256,
            "ai_journey.evidence.trace_head_sha256": trace_head_sha256,
            "ai_journey.clock.source": MONOTONIC_CLOCK,
            "ai_journey.clock.mapping": TIME_MAPPING_FORMULA,
            "ai_journey.clock.origin_monotonic_ns": mapping.origin_monotonic_ns,
            "ai_journey.clock.anchor_unix_nano": mapping.anchor_unix_nano,
        },
        scope_name=SCOPE_NAME,
        scope_version=SCOPE_VERSION,
        spans=spans,
    )
