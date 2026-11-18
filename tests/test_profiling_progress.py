"""The progress table records the Day 75 labs without claiming learner work."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def day_75_row():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 75 |")]
    return [cell.strip() for cell in row.strip("|").split("|")]


def test_day_75_row_names_the_merged_labs():
    _, code, _, _ = day_75_row()

    for part in ("torch.profiler", "MPS event timing", "named bottleneck", "hotspot"):
        assert part in code


def test_day_75_row_keeps_lecture_and_learner_evidence_pending():
    _, _, lecture, manual = day_75_row()

    assert lecture.endswith("pending user confirmation")
    assert "GPU MODE Lecture 1" in lecture
    assert manual.endswith("pending")
    assert "Learner-run profiler capture" in manual


def test_every_day_75_guide_exists():
    for guide in (
        "profile-trace-lab.md",
        "gpu-timing-lab.md",
        "matmul-bottleneck-lab.md",
        "simulator-profiling.md",
    ):
        assert (ROOT / "docs" / guide).is_file()
