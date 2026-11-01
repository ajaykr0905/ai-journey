from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_gates_isolated_hybrid_parallel_commands_and_retains_evidence():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = workflow.split("  hybrid-parallel-cpu:\n", 1)[1].split(
        "  distributed-capacity-contracts:\n", 1
    )[0]
    for required in (
        "python -m build",
        "ai_journey.__file__.startswith",
        "ai-journey-hybrid-parallel",
        "ai-journey-hybrid-parallel-verify",
        "hybrid-parallel-evidence",
        "actions/upload-artifact@v4",
        "if: always()",
    ):
        assert required in job


def test_readiness_requires_complete_hybrid_parallel_workflow_surface():
    required = json.loads(
        (ROOT / "config" / "deployment_readiness.json").read_text(encoding="utf-8")
    )["required_files"]
    expected = {
        "docs/hybrid-parallelism.md",
        "docs/ai-infrastructure-program-ledger.md",
        "src/ai_journey/hybrid_topology.py",
        "src/ai_journey/tensor_sharding.py",
        "src/ai_journey/sequence_sharding.py",
        "src/ai_journey/pipeline_partition.py",
        "src/ai_journey/pipeline_schedule.py",
        "src/ai_journey/pipeline_metrics.py",
        "src/ai_journey/parallel_communication.py",
        "src/ai_journey/activation_handoff.py",
        "src/ai_journey/pipeline_handoff_probe.py",
        "src/ai_journey/hybrid_parallel_report.py",
        "src/ai_journey/hybrid_parallel_schema.py",
        "src/ai_journey/hybrid_parallel_seal.py",
        "src/ai_journey/hybrid_parallel_analysis.py",
        "src/ai_journey/hybrid_parallel_report_verification.py",
        "src/ai_journey/hybrid_parallel_io.py",
        "src/ai_journey/hybrid_parallel_cli.py",
        "src/ai_journey/hybrid_parallel_verify.py",
        "tests/test_hybrid_topology.py",
        "tests/test_tensor_sharding.py",
        "tests/test_sequence_sharding.py",
        "tests/test_pipeline_partition.py",
        "tests/test_pipeline_schedule.py",
        "tests/test_pipeline_metrics.py",
        "tests/test_parallel_communication.py",
        "tests/test_activation_handoff.py",
        "tests/test_pipeline_handoff_probe.py",
        "tests/test_hybrid_parallel_report.py",
        "tests/test_hybrid_parallel_io.py",
        "tests/test_hybrid_parallel_cli.py",
        "tests/test_hybrid_parallel_package.py",
        "tests/test_hybrid_parallel_ci_contract.py",
        "tests/test_hybrid_parallel_docs.py",
    }
    assert expected <= set(required)
    assert all((ROOT / path).is_file() for path in expected)
