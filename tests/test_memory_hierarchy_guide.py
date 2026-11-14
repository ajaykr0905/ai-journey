import json
import re
import shlex
from pathlib import Path

from ai_journey import memory_hierarchy_cli
from ai_journey.memory_access_benchmark import PERCENTILE_METHOD
from ai_journey.memory_hierarchy_access import ACCESS_PATTERNS
from ai_journey.memory_hierarchy_report import MEASUREMENT_BOUNDARY
from ai_journey.memory_hierarchy_stream import STREAM_KERNELS

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/memory-hierarchy-lab.md"
COMMAND = "ai-journey-memory-hierarchy"
PINNED = "config/day-71-memory-hierarchy.json"


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def documented_commands(text):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words and words[0] == COMMAND:
                commands.append(words[1:])
    return commands


def test_documented_commands_parse_and_operate_the_pinned_config():
    commands = documented_commands(guide())
    parsed = [memory_hierarchy_cli.build_parser().parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["run", "verify"]
    assert {str(arguments.config) for arguments in parsed} == {PINNED}
    assert parsed[1].report == parsed[0].output
    assert (ROOT / PINNED).is_file()
    assert memory_hierarchy_cli.build_parser().prog == COMMAND


def test_guide_quotes_the_evidence_boundary_and_the_statistics_method():
    text = flat(guide())

    assert flat(MEASUREMENT_BOUNDARY["statement"]) in text
    assert "sorted samples[ceil(p n / 100) - 1]" in text
    assert "nearest rank" in PERCENTILE_METHOD and "no interpolation" in text
    for phrase in (
        "Nothing here runs on, measures, or predicts a GPU.",
        "Exit code 0 means success, 1 a readable report that failed verification",
        "`latency_measured: false`",
        "`synthetic_injected_clock`",
        "It cannot re-observe a timing.",
        "names no cache level and detects no cache size",
    ):
        assert phrase in text, phrase
    assert (
        memory_hierarchy_cli.EXIT_OK,
        memory_hierarchy_cli.EXIT_REJECTED,
        memory_hierarchy_cli.EXIT_INVALID,
    ) == (0, 1, 2)


def test_guide_tables_match_the_kernel_calls_and_pinned_controls():
    text = guide()
    pinned = json.loads((ROOT / PINNED).read_text(encoding="utf-8"))

    for kernel in STREAM_KERNELS.values():
        for call in kernel.to_mapping()["calls"]:
            assert f"`{call}`" in text, call
    for pattern in ("copy", "sequential", "strided", "random"):
        assert f"`{pattern}`" in text
    assert "mode='clip'" in ACCESS_PATTERNS["random"].call
    for field, value in pinned["timing"].items():
        assert f"| `timing.{field}` | {value} |" in text, field
    assert f"| `access.elements` | {pinned['access']['elements']} |" in text
    assert (
        f"| `littles_law.latency_ns` | {pinned['littles_law']['latency_ns']} |" in text
    )
