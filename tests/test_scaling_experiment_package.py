from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_exposes_the_scaling_experiment_command():
    project = (ROOT / "pyproject.toml").read_text()

    assert (
        'ai-journey-scaling-experiment = "ai_journey.scaling_experiment_cli:main"'
        in project
    )


def test_scaling_experiment_command_has_no_new_runtime_dependency():
    project = (ROOT / "pyproject.toml").read_text()
    dependency_block = project.split("dependencies = [", 1)[1].split("]", 1)[0]

    assert "scaling" not in dependency_block.lower()
