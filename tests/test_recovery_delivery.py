import json
from pathlib import Path

from ai_journey.recovery_config import config_from_mapping

ROOT = Path(__file__).parents[1]


def test_checked_in_day70_config_pins_reproducible_fault_boundary():
    payload = json.loads((ROOT / "config/day-70-worker-recovery.json").read_bytes())
    config = config_from_mapping(payload)

    assert config.experiment_name == "day-70-worker-recovery"
    assert config.work_items == 20
    assert config.checkpoint_interval == 4
    assert config.fault_after_items == 11
    assert config.fault_after_items - 8 == 3


def test_release_docs_disclose_cost_evaluations_failures_and_boundaries():
    readme = (ROOT / "README.md").read_text()
    guide = (ROOT / "docs/project1-release.md").read_text()

    for phrase in (
        "₹0",
        "HellaSwag",
        "LAMBADA",
        "What went wrong",
        "real NVIDIA multi-GPU",
        "ai-journey-recovery run",
        "ai-journey-recovery verify",
    ):
        assert phrase in readme + guide
    assert "GPU, distributed-scale, production-readiness" in readme


def test_ci_executes_installed_recovery_operator_and_retains_evidence():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    for command in (
        "ruff check src/ai_journey/recovery_*.py tests/test_recovery_*.py",
        "python -m pytest -q tests/test_recovery_*.py",
        "ai-journey-recovery run --source",
        "ai-journey-recovery verify --run-dir",
        "name: worker-recovery-evidence",
    ):
        assert command in workflow
