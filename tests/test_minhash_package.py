from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_package_installs_minhash_build_verify_and_replay_command():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["ai-journey-minhash"] == (
        "ai_journey.minhash_cli:main"
    )


def test_minhash_command_adds_no_runtime_dependency():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["dependencies"] == [
        "numpy>=1.24,<2",
        "matplotlib>=3.7,<4",
        "torch>=2.2,<3",
    ]
