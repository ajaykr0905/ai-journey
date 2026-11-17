import json
import re
import shlex
from pathlib import Path

from ai_journey import gpu_ceiling_cli, gpu_ceiling_roofline
from ai_journey.gpu_ceiling_analysis import FINDINGS, FIXED_SHARE_LIMIT, SLOWDOWN_LIMIT
from ai_journey.gpu_ceiling_config import (
    MAX_RUN_ACCESSES,
    MAX_RUN_BYTES,
    MAX_RUN_CALLS,
    MAX_RUN_FLOPS,
    load_config,
)
from ai_journey.gpu_ceiling_reference import (
    FMA_ADDEND,
    FMA_MULTIPLIER,
    MAX_REDUCE_VALUE,
)
from ai_journey.gpu_ceiling_report import (
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_SCOPE,
)
from ai_journey.gpu_ceiling_source import (
    BANK_LOADS_PER_TRIP,
    BANK_TILE_ROWS,
    FMA_CHAIN_COUNTS,
    FMA_STEPS_PER_TRIP,
)
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS
from test_gpu_ceiling_recorded import RECORDED

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/gpu-ceilings-lab.md"
COMMAND = ["python", "-m", "ai_journey.gpu_ceiling_cli"]
PINNED = "config/day-74-gpu-ceilings.json"


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def documented_commands(text):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(COMMAND)] == COMMAND:
                commands.append(words[len(COMMAND) :])
    return commands


def test_documented_commands_parse_and_operate_the_pinned_config():
    commands = documented_commands(guide())
    parser = gpu_ceiling_cli.build_parser()
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["run", "verify"]
    assert {str(arguments.config) for arguments in parsed} == {PINNED}
    assert parsed[1].report == parsed[0].output
    assert (ROOT / PINNED).is_file()
    assert f"the program name `{parser.prog}`" in flat(guide())


def test_guide_defines_peak_as_a_ceiling_measured_on_this_host():
    text = flat(guide())

    assert 'In this lab, "peak" means a ceiling measured on this host.' in text
    assert "It is never a vendor number." in text
    assert "It is not a vendor peak and not a hardware limit." in text
    assert "Nothing here runs a CUDA kernel or uses an NVIDIA GPU." in text
    assert flat(MEASUREMENT_BOUNDARY["statement"]) in text
    assert MEASUREMENT_BOUNDARY["vendor_peak_used"] is False
    for name in (MEASUREMENT_LABEL, MEASUREMENT_SCOPE, SYNTHETIC_SCOPE):
        assert f"`{name}`" in text, name


def test_guide_states_the_kernel_constants_of_the_source():
    text = flat(guide())

    assert float(FMA_MULTIPLIER) == (2**24 - 4173) * 2.0**-24
    assert float(FMA_ADDEND) == 12561923 * 2.0**-35
    assert "`a = (2^24 - 4173) * 2^-24`" in text
    assert "`b = 12561923 * 2^-35`" in text
    counts = ", ".join(map(str, FMA_CHAIN_COUNTS[:-1])) + f" and {FMA_CHAIN_COUNTS[-1]}"
    assert f"for `c` in {counts}." in text
    assert FMA_STEPS_PER_TRIP == 8 and "eight times to each chain" in text
    assert f"The tile has {BANK_TILE_ROWS} rows" in text
    assert f"then reads {BANK_LOADS_PER_TRIP} rows" in text
    assert f"whole numbers below {MAX_REDUCE_VALUE + 1}" in text


def test_guide_states_the_thresholds_and_every_finding():
    text = flat(guide())

    assert FIXED_SHARE_LIMIT == 0.05 and "at most 5% of" in text
    assert SLOWDOWN_LIMIT == 1.25 and "A ratio of 1.25 or more" in text
    for finding in FINDINGS:
        assert f"- `{finding}`:" in guide(), finding


def test_guide_states_the_exit_codes_and_every_missing_requirement():
    text = flat(guide())

    for phrase in (
        "0: the command succeeded.",
        "1: the command read a report, but the report failed the checks.",
        "2: a usage error",
        "3: this host cannot run Metal kernels through PyTorch",
    ):
        assert phrase in text, phrase
    assert (
        gpu_ceiling_cli.EXIT_OK,
        gpu_ceiling_cli.EXIT_REJECTED,
        gpu_ceiling_cli.EXIT_INVALID,
        gpu_ceiling_cli.EXIT_UNAVAILABLE,
    ) == (0, 1, 2, 3)
    for reason in UNAVAILABLE_REASONS:
        assert f"`{reason}`" in text, reason
    for prefix in (
        "`apple_gpu_metal_measurement`",
        "`gpu_ceiling_measurement=`",
        "`synthetic_timing`",
        "`gpu_ceiling_synthetic_timing=`",
        "`verified_`",
    ):
        assert prefix in text, prefix


def test_guide_controls_and_planned_work_match_the_pinned_config():
    text = guide()
    pinned = json.loads((ROOT / PINNED).read_text(encoding="utf-8"))
    config = load_config(ROOT / PINNED)

    assert f"| `seed` | {pinned['seed']} |" in text
    for section, fields in (
        ("correctness", ("threads", "fma_steps", "stream_float4", "bank_trips")),
        ("fma", ("group_size", "threads")),
        (
            "bandwidth",
            (
                "group_size",
                "copy_float4_per_thread",
                "reduce_float4_per_thread",
                "target_bytes_per_sample",
            ),
        ),
        ("bank", ("group_size", "threads", "trips")),
        ("timing", ("warmup", "repeats", "max_runtime_seconds")),
    ):
        for field in fields:
            row = f"| `{section}.{field}` | {pinned[section][field]} |"
            assert row in text, row
    work = config.planned_work()
    assert (
        f"the counts are {work.correctness_cases} correctness cases, "
        f"{work.fma_points} FMA points, {work.stream_points} stream points and "
        f"{work.bank_points} bank points. The run has {work.timed_samples} timed "
        f"samples and {work.calls} kernel calls. The work is {work.flops} FLOPs, "
        f"{work.stream_bytes} stream bytes and {work.bank_accesses} threadgroup loads."
    ) in flat(text)
    exponents = [
        limit.bit_length() - 1
        for limit in (MAX_RUN_CALLS, MAX_RUN_FLOPS, MAX_RUN_BYTES, MAX_RUN_ACCESSES)
    ]
    assert (
        "A run can have at most 2^{} calls, 2^{} FLOPs, 2^{} stream bytes and 2^{} "
        "loads.".format(*exponents)
    ) in flat(text)


def test_guide_describes_the_host_of_the_recorded_report():
    text = flat(guide())
    environment = RECORDED["environment"]

    assert (
        f"macOS {environment['macos_version']}, CPython "
        f"{environment['python_version']}, NumPy {environment['numpy_version']} "
        f"and PyTorch {environment['torch_version']}."
    ) in text
    assert f"Darwin {environment['platform_machine']} host" in text


def test_guide_names_the_roofline_helpers_that_compare_a_kernel():
    text = flat(guide())
    assert "`gpu_ceiling_roofline.py`" in text
    for name in ("load_roofline", "roofline_from_report"):
        assert callable(getattr(gpu_ceiling_roofline, name))
        assert f"`{name}`" in text, name
    for name in ("attainable_gflops_per_second", "bound", "below_ceiling"):
        assert callable(getattr(gpu_ceiling_roofline.MeasuredRoofline, name))
        assert f"`{name}`" in text, name
    assert "A tampered report gives no roofline" in text
