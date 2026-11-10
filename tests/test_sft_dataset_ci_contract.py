from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_lints_compiles_and_tests_every_sft_dataset_contract():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert (
        "ruff check src/ai_journey/sft_dataset_*.py tests/test_sft_dataset_*.py"
        in workflow
    )
    assert (
        "ruff format --check src/ai_journey/sft_dataset_*.py tests/test_sft_dataset_*.py"
        in workflow
    )
    assert (
        "python -m compileall -q src/ai_journey/sft_dataset_*.py tests/test_sft_dataset_*.py"
        in workflow
    )
    assert "python -m pytest -q tests/test_sft_dataset_*.py" in workflow


def test_ci_fetches_the_pinned_public_snapshot_and_operates_installed_commands():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert (
        "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl"
        in workflow
    )
    assert "ai-journey-sft-dataset build --source" in workflow
    assert "ai-journey-sft-dataset verify --evidence" in workflow
    assert "ai-journey-sft-dataset replay --source" in workflow
    assert "${{ runner.temp }}/sft-dataset-evidence/" in workflow


def test_ci_builds_the_wheel_before_running_sft_dataset_operations():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    wheel = workflow.index(
        'python -m build --wheel --outdir "$RUNNER_TEMP/data-pipeline-dist"'
    )
    operation = workflow.index("ai-journey-sft-dataset build --source")
    assert wheel < operation
