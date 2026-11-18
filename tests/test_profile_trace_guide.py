import re
import shlex
import subprocess
import sys
from pathlib import Path

from ai_journey import profile_trace_cli
from ai_journey.profile_trace_config import load_config
from ai_journey.profile_trace_layout import ITERATION, phase_region, side_region
from ai_journey.profile_trace_report import MEASURED_LABEL
from test_profile_trace_recorded import write_recorded

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/profile-trace-lab.md"
COMMAND = ["python", "-m", "ai_journey.profile_trace_cli"]
PINNED = "config/day-75-profile-trace.json"
# A None entry in sys.modules makes every import of that name fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.profile_trace_cli import main
sys.exit(main(sys.argv[1:]))
"""


def flat(text):
    return " ".join(text.split())


def documented_commands():
    commands = []
    text = GUIDE.read_text(encoding="utf-8")
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(COMMAND)] == COMMAND:
                commands.append(words[len(COMMAND) :])
    return commands


def test_documented_commands_parse_and_use_the_pinned_config():
    parser = profile_trace_cli.build_parser()
    commands = documented_commands()
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["capture", "verify", "analyze"]
    assert str(parsed[0].config) == str(parsed[1].config) == PINNED
    assert parsed[1].report == parsed[0].output_dir / "report.json"
    assert parsed[2].trace.parent == parsed[0].output_dir
    assert f"the program name `{parser.prog}`" in flat(GUIDE.read_text())


def test_the_guide_states_exit_codes_limits_and_layout_names():
    text = flat(GUIDE.read_text(encoding="utf-8"))
    for code in (
        profile_trace_cli.EXIT_OK,
        profile_trace_cli.EXIT_REJECTED,
        profile_trace_cli.EXIT_INVALID,
        profile_trace_cli.EXIT_UNAVAILABLE,
    ):
        assert f"- {code}: " in text
    assert "It records no GPU kernel and no GPU kernel time." in text
    assert "It is not a reading of hardware counters." in text
    assert MEASURED_LABEL in text
    for name in (
        ITERATION,
        side_region("torch_matmul").replace("torch_matmul", "<side>"),
    ):
        assert f"`{name}`" in text
    assert f"`{phase_region('matmul_x_rows', 'synchronize')}`" in text
    assert (
        "Nothing here runs a CUDA kernel, uses an NVIDIA GPU, or uses Nsight." in text
    )
    assert (
        "There is no Nsight, no CUDA toolkit and no access to hardware counters."
        in text
    )


def test_the_pinned_controls_match_the_table():
    config = load_config(ROOT / PINNED)
    text = GUIDE.read_text(encoding="utf-8")
    assert f"| `seed` | {config.seed} |" in text
    assert f"| `warmup` | {config.warmup} |" in text
    assert f"| `iterations` | {config.iterations} |" in text
    assert f"| `analysis.top_n` | {config.analysis.top_n} |" in text
    assert f"| `analysis.timeline_width` | {config.analysis.timeline_width} |" in text
    ceilings = config.ceilings
    assert (
        f"{ceilings.fma_gflops_per_second} GFLOP/s, "
        f"{ceilings.bandwidth_gb_per_second} GB/s, "
        f"{ceilings.torch_matmul_gflops_per_second} GFLOP/s"
    ) in text


def test_analyze_and_verify_run_without_torch(tmp_path):
    write_recorded(tmp_path)
    env = {"PYTHONPATH": str(ROOT / "src")}
    for argv in (
        ["verify", "--report", str(tmp_path / "report.json")],
        ["analyze", "--trace", str(tmp_path / "trace-1-256x192x320.json")],
    ):
        completed = subprocess.run(
            [sys.executable, "-c", WITHOUT_TORCH, *argv],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
    assert "top 1 name=matmul_x_rows/synchronize" in completed.stdout
