import copy
import json
from hashlib import sha256

import pytest

from ai_journey.kernel_metal_report import (
    ENVIRONMENT_FIELDS,
    MEASUREMENT_LABEL,
    SYNTHETIC_LABEL,
)
from ai_journey.matmul_bottleneck_config import MatmulBottleneckError
from ai_journey.matmul_bottleneck_timing import (
    MAX_CALL_NS,
    MAX_TIMING_BYTES,
    TIMING_KIND,
    load_timing,
    timing_from_mapping,
)
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.memory_access_plan import canonical_json_bytes

ENVIRONMENT = {
    "macos_version": "27.2",
    "numpy_version": "1.26.4",
    "platform_machine": "arm64",
    "platform_system": "Darwin",
    "python_implementation": "CPython",
    "python_version": "3.11.14",
    "torch_version": "2.15.0",
}
# Synthetic one-call medians for the two shapes of the recorded Day 74
# matmul-metal fixture: (gpu, host, dispatch) nanoseconds per kernel.
SYNTHETIC_TIMES = {
    (256, 256, 256): {
        "matmul_x_columns": (110_000, 130_000, 20_000),
        "matmul_x_rows": (270_000, 290_000, 20_000),
        "torch_matmul": (40_000, 60_000, 15_000),
    },
    (129, 65, 1000): {
        "matmul_x_columns": (100_000, 900_000, 40_000),
        "matmul_x_rows": (110_000, 950_000, 40_000),
        "torch_matmul": (90_000, 700_000, 40_000),
    },
}


def synthetic_timing_mapping(times=SYNTHETIC_TIMES, environment=ENVIRONMENT):
    """A synthetic timing input in the declared schema, never a measurement."""

    return {
        "schema_version": 1,
        "kind": TIMING_KIND,
        "measured": False,
        "label": SYNTHETIC_LABEL,
        "producer": "synthetic test values",
        "environment": dict(environment),
        "repeats": 5,
        "load_average_before": [1.0, 1.5, 2.0],
        "load_average_after": [1.25, 1.5, 2.0],
        "points": [
            {
                "m": m,
                "n": n,
                "k": k,
                "kernel": kernel,
                "median_gpu_ns": gpu,
                "median_host_ns": host,
                "median_dispatch_ns": dispatch,
            }
            for (m, n, k), kernels in times.items()
            for kernel, (gpu, host, dispatch) in kernels.items()
        ],
    }


def changed(change):
    mapping = copy.deepcopy(synthetic_timing_mapping())
    change(mapping)
    return mapping


def test_a_synthetic_input_parses_round_trips_and_finds_its_entries():
    mapping = synthetic_timing_mapping()
    timing = timing_from_mapping(mapping)

    assert timing.to_mapping() == mapping
    assert (timing.measured, timing.label) == (False, SYNTHETIC_LABEL)
    assert tuple(name for name, _ in timing.environment) == ENVIRONMENT_FIELDS
    assert timing.digest() == sha256(canonical_json_bytes(mapping)).hexdigest()
    entry = timing.entry(matmul_shape(129, 65, 1000), "matmul_x_rows")
    assert (entry.gpu_ns, entry.host_ns, entry.dispatch_ns) == (
        110_000,
        950_000,
        40_000,
    )
    assert timing.entry(matmul_shape(128, 128, 128), "matmul_x_rows") is None


def test_a_measurement_needs_its_label_and_macos():
    measured = changed(lambda m: m.update(measured=True, label=MEASUREMENT_LABEL))
    assert timing_from_mapping(measured).measured is True

    linux = copy.deepcopy(measured)
    linux["environment"]["platform_system"] = "Linux"
    with pytest.raises(MatmulBottleneckError, match="must come from macOS"):
        timing_from_mapping(linux)
    unlabeled = changed(lambda m: m.update(measured=True))
    with pytest.raises(MatmulBottleneckError, match="label does not match"):
        timing_from_mapping(unlabeled)


def first(mapping):
    return mapping["points"][0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m.update(kind="other"), "timing kind must be"),
        (lambda m: m.update(schema_version=2), "schema_version must be 1"),
        (lambda m: m.update(extra=1), "timing fields do not match"),
        (lambda m: m.update(measured=1), "measured must be a boolean"),
        (lambda m: m.update(producer=""), "producer must be"),
        (lambda m: m.update(producer="x" * 301), "producer must be"),
        (lambda m: m.update(producer="tab\there"), "producer must be"),
        (lambda m: m.update(repeats=0), "repeats must be"),
        (lambda m: m.update(points=[]), "1 to 96 entries"),
        (lambda m: m.update(load_average_before=[1, 2, 3]), "three nonnegative"),
        (lambda m: m["environment"].pop("torch_version"), "environment fields"),
        (lambda m: m["environment"].update(torch_version="a b"), "Metal labs' tokens"),
        (lambda m: first(m).update(kernel="matmul_tiled"), "timing kernel must be"),
        (lambda m: first(m).update(median_gpu_ns=0), "median_gpu_ns must be"),
        (lambda m: first(m).update(median_host_ns=1.5), "median_host_ns must be"),
        (
            lambda m: first(m).update(median_dispatch_ns=MAX_CALL_NS + 1),
            "median_dispatch_ns must be",
        ),
        (lambda m: first(m).update(m=0), "timing point 0: m must be in"),
        (lambda m: first(m).pop("k"), "timing point 0 fields"),
        (lambda m: m["points"].append(dict(first(m))), "repeat a kernel and shape"),
    ],
)
def test_invalid_inputs_are_refused_with_a_named_reason(change, message):
    with pytest.raises(MatmulBottleneckError, match=message):
        timing_from_mapping(changed(change))


def test_the_file_reader_takes_any_layout_but_refuses_bad_files(tmp_path):
    mapping = synthetic_timing_mapping()
    pretty = tmp_path / "timing.json"
    pretty.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    assert load_timing(pretty).digest() == timing_from_mapping(mapping).digest()

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"kind": "a", "kind": "b"}', encoding="utf-8")
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (MAX_TIMING_BYTES + 1))
    for path, message in ((duplicate, "duplicate JSON field"), (large, "exceeds")):
        with pytest.raises(MatmulBottleneckError, match=message):
            load_timing(path)
