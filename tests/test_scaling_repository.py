from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_surfaces_scaling_work_without_claiming_learning_completion():
    readme = (ROOT / "README.md").read_text()
    progress = (ROOT / "PROGRESS.md").read_text()

    assert "selected verified code through Day 62" in readme
    assert (
        "[Day 62 scaling experiment guide](docs/scaling-experiment-protocol.md)"
        in readme
    )
    assert "[scaling-law planning guide](docs/scaling-law-planning.md)" in readme
    assert "| 60 | 2026-11-03 | Analytical Chinchilla target" in readme
    assert "| 62 | 2026-11-05 | Bounded scaling sweep protocol" in readme
    assert "| 60 | Declared 100M-model token" in progress
    assert "| 62 | Complete model/token/seed grid" in progress
    assert "Lecture 9 and LoRA support video pending user confirmation" in progress
    assert "Lecture 11 and RLHF support video pending user confirmation" in progress


def test_repository_keeps_day60_as_analytical_not_executed_evidence():
    readme = (ROOT / "README.md").read_text()
    progress = (ROOT / "PROGRESS.md").read_text()

    assert "explicit non-execution boundaries" in readme
    assert "real-run data/hardware/cost approval pending" in progress
