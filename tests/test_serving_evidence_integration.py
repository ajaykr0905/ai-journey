import json
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from ai_journey.serving_analysis import (
    GoodputTarget,
    failure_fraction,
    goodput_fraction,
    output_throughput_per_second,
    successful_ttft_summary,
)
from ai_journey.serving_compare import (
    ComparisonKey,
    RegressionThresholds,
    ServingSnapshot,
    compare_snapshots,
    serving_comparison_digest,
)
from ai_journey.serving_contract import (
    ClaimBoundary,
    EngineIdentity,
    HardwareManifest,
    ModelIdentity,
    RunLimits,
    ServingRunContract,
    SoftwareManifest,
    WorkloadShape,
    serving_contract_digest,
)
from ai_journey.serving_faults import (
    FaultEvent,
    FaultKind,
    RecoveryObservation,
    fault_recovery_digest,
)
from ai_journey.serving_observation import (
    RequestObservation,
    RequestTiming,
    ServingOutcome,
    serving_observation_digest,
    validate_observation_for_request,
)
from ai_journey.serving_report import (
    EvidenceReferences,
    ServingEvidenceReport,
    seal_serving_report,
)
from ai_journey.serving_report_io import read_sealed_report, write_sealed_report
from ai_journey.serving_trace import constant_rate_trace, serving_trace_digest


def _digest_payload(value) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_complete_serving_evidence_pipeline_is_ci_verifiable(tmp_path: Path):
    contract = ServingRunContract(
        protocol_version=1,
        workload=WorkloadShape(2, 16, 4, 1),
        engine=EngineIdentity("fixture", "1.0", "local"),
        model=ModelIdentity("public/tiny", "model-rev", "tokenizer-rev", 128),
        hardware=HardwareManifest("cpu", 1, 1_048_576, "memory"),
        software=SoftwareManifest("test-os", "3.11", "torch-2", "none"),
        limits=RunLimits(10, 7),
        claims=ClaimBoundary(
            "cpu_fixture", ("determinism", "failure_recovery", "schema_validation")
        ),
    )
    trace = constant_rate_trace(
        request_count=2,
        interval_ms=1000,
        prompt_tokens=16,
        max_output_tokens=4,
        prompt_sha256=sha256(b"content-private-fixture").hexdigest(),
    )
    observations = (
        RequestObservation(
            trace[0].request_id,
            ServingOutcome.SUCCESS,
            RequestTiming(0, 10, 20, 500_000_000),
            16,
            4,
            False,
            0,
        ),
        RequestObservation(
            trace[1].request_id,
            ServingOutcome.SUCCESS,
            RequestTiming(1_000_000_000, 1_000_000_010, 1_000_000_020, 1_500_000_000),
            16,
            4,
            False,
            0,
        ),
    )
    for request, observation in zip(trace, observations, strict=True):
        validate_observation_for_request(observation, request)
    events = (
        FaultEvent("restart-1", FaultKind.ENGINE_RESTART, "engine", 2_000_000_000, 10),
    )
    recoveries = (
        RecoveryObservation("restart-1", 2_000_000_001, 2_000_000_010, 2_000_000_020),
    )
    trace_sha256 = serving_trace_digest(trace)
    key = ComparisonKey(
        protocol_version=contract.protocol_version,
        workload_sha256=_digest_payload(asdict(contract.workload)),
        model_sha256=_digest_payload(asdict(contract.model)),
        trace_sha256=trace_sha256,
    )
    snapshot = ServingSnapshot(
        key=key,
        engine_label=contract.engine.name,
        p95_ttft_ns=successful_ttft_summary(observations).p95_ns,
        output_tokens_per_second=output_throughput_per_second(observations),
        goodput=goodput_fraction(observations, GoodputTarget(100, 600_000_000)),
        failure_rate=failure_fraction(observations),
        peak_memory_bytes=1024,
    )
    zero = Fraction(0)
    comparison = compare_snapshots(
        snapshot,
        snapshot,
        RegressionThresholds(zero, zero, zero, zero, zero),
    )
    report = ServingEvidenceReport(
        report_version=1,
        references=EvidenceReferences(
            serving_contract_digest(contract),
            trace_sha256,
            serving_observation_digest(observations),
            fault_recovery_digest(events, recoveries),
        ),
        boundary=contract.claims,
        claims=("determinism", "failure_recovery", "schema_validation"),
        snapshot=snapshot,
        comparison_sha256=serving_comparison_digest(comparison),
    )
    sealed = seal_serving_report(report)
    path = write_sealed_report(tmp_path / "serving-evidence.json", sealed)

    assert comparison.passed
    assert read_sealed_report(path) == sealed
