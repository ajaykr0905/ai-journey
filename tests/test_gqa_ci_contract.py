"""The new architecture has an executable installed operational CI path."""

import json
import importlib
from pathlib import Path
import tomllib

import pytest

from ai_journey.gpt2_training import load_training_spec, training_spec_from_mapping
from ai_journey.moe_config import MoEConfig
from ai_journey.gqa_config import GQAConfig

ROOT = Path(__file__).resolve().parents[1]


def test_grouped_training_schema_is_explicit_and_old_schemas_do_not_change():
    spec = load_training_spec(ROOT / "config/gqa-cpu-lab.json")
    payload = spec.to_mapping()
    assert payload["schema_version"] == 3
    assert training_spec_from_mapping(payload).fingerprint() == spec.fingerprint()
    assert (
        load_training_spec(ROOT / "config/gpt2-training.json").to_mapping()[
            "schema_version"
        ]
        == 1
    )
    for alias in (True, 1, 2, 3.0):
        with pytest.raises((ValueError, TypeError)):
            training_spec_from_mapping({**payload, "schema_version": alias})


def test_plain_ce_spec_rejects_grouped_experts_without_auxiliary_objective():
    spec = load_training_spec(ROOT / "config/gqa-cpu-lab.json")
    payload = spec.to_mapping()
    dense = spec.model.decoder
    expert = MoEConfig(dense.gpt2, position_kind="rope")
    payload["model"] = {
        "kind": "grouped_query",
        "config": GQAConfig(expert, 1).to_mapping(),
    }
    with pytest.raises(ValueError, match="auxiliary objective"):
        training_spec_from_mapping(payload)


def test_ci_executes_installed_grouped_training_replay_and_real_crash_verification():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    job = workflow.split("  grouped-query-cpu:\n", 1)[1].split("  moe-cpu-lab:\n", 1)[0]
    for command in (
        "python -m pytest -q tests/test_gqa*.py",
        "python -m build",
        "ai-journey-gpt2-train train",
        "ai-journey-gpt2-train verify",
        "ai-journey-gpt2-train replay",
        "ai-journey-ddp recover-crash",
        "ai-journey-ddp verify-crash-recovery",
    ):
        assert command in job
    assert "ai_journey.__file__.startswith" in job
    assert 'export PATH="$RUNNER_TEMP/gqa-installed/bin:$PATH"' in job
    required = json.loads((ROOT / "config/deployment_readiness.json").read_text())[
        "required_files"
    ]
    for path in (
        "config/gqa-cpu-lab.json",
        "docs/grouped-query-cpu.md",
        "src/ai_journey/gqa_model.py",
        "src/ai_journey/grouped_rotary_attention.py",
        "tests/test_gqa_cli_integration.py",
    ):
        assert path in required and (ROOT / path).is_file()


def test_public_experiment_console_and_installed_cache_replay_are_required(capsys):
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())
    entry = package["project"]["scripts"]["ai-journey-gqa-experiment"]
    module, attribute = entry.split(":")
    with pytest.raises(SystemExit) as result:
        getattr(importlib.import_module(module), attribute)(["--help"])
    assert result.value.code == 0
    assert "cache-replay" in capsys.readouterr().out
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    job = workflow.split("  grouped-query-cpu:\n", 1)[1].split("  moe-cpu-lab:\n", 1)[0]
    for operation in (
        "plan",
        "public",
        "verify",
        "replay",
        "cache",
        "cache-verify",
        "cache-replay",
    ):
        assert f"ai-journey-gqa-experiment {operation} " in job
    assert "6e328e8195c04a297c1e375d21fe663be1ce6178/en_pud-ud-test.conllu" in job
    required = json.loads((ROOT / "config/deployment_readiness.json").read_text())[
        "required_files"
    ]
    for pattern in ("src/ai_journey/gqa*.py", "tests/test_gqa*.py"):
        for path in ROOT.glob(pattern):
            assert path.relative_to(ROOT).as_posix() in required
