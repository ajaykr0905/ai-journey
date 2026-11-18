import json
from contextlib import contextmanager

import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.matmul_metal_runtime import SIDES, MatmulWorkload, WorkloadCheck
from ai_journey.matmul_metal_source import KERNELS
from ai_journey.profile_trace_capture import (
    REDACTED_HOST,
    CapturedShape,
    ProfileTraceCheckError,
    TorchProfilerSession,
    capture_shape,
    capture_traces,
    redact_export,
)
from ai_journey.profile_trace_config import config_from_mapping
from ai_journey.profile_trace_events import (
    ProfileTraceError,
    decode_trace_json,
    parse_trace_bytes,
)
from ai_journey.profile_trace_layout import ITERATION, check_layout
from test_kernel_metal_runtime import requires_metal
from test_profile_trace_config import make_config_mapping

LIMITS = PipelineLimits(
    max_threads_per_threadgroup=1024,
    thread_execution_width=32,
    static_threadgroup_memory_bytes=0,
)
# Host nanoseconds each stand-in step takes: a dispatch, then the wait in
# synchronize, per side. The naive waits are longest, as on the GPU.
DISPATCH_NS = {
    "matmul_x_columns": 9_000,
    "matmul_x_rows": 11_000,
    "torch_matmul": 60_000,
}
WAIT_NS = {
    "matmul_x_columns": 400_000,
    "matmul_x_rows": 900_000,
    "torch_matmul": 50_000,
}
PRIVATE_HOST = "PRIVATE-HOST-NAME"


class FakeSession:
    """A torch-free stand-in that writes the torch export layout on a fake clock.

    Every step moves the clock by a fixed amount, so the trace and its
    analysis are exact. It is never a measurement.
    """

    measured = False
    torch_version = "2.15.0"
    float32_matmul_precision = "highest"

    def __init__(self, *, same_bits=True, disagreement=0.0, nested=None):
        self.limits = dict.fromkeys(KERNELS, LIMITS)
        self.now = 1_000_000_000
        self.spans = None
        self.same_bits = same_bits
        self.disagreement = disagreement
        self.nested = nested
        self.current = None

    def tick(self, nanoseconds):
        self.now += nanoseconds

    def workload(self, left, right, launches):
        assert set(launches) == set(KERNELS)

        def side(name):
            def call():
                self.current = name
                self.tick(DISPATCH_NS[name])

            return call

        def synchronize():
            if self.current is not None:
                self.tick(WAIT_NS[self.current])
                self.current = None

        return MatmulWorkload(
            sides={name: side(name) for name in SIDES},
            synchronize=synchronize,
            check=lambda: WorkloadCheck(self.same_bits, self.disagreement),
        )

    @contextmanager
    def annotate(self, name):
        start = self.now
        self.tick(1_000)
        yield None
        self.tick(1_000)
        if self.spans is not None:
            self.spans.append((name, start, self.now - start))
        if self.nested is not None and name == self.nested[0]:
            self.spans.append((self.nested[1], start, 0))

    def device_events(self):
        session = self

        class Event:
            def record(self):
                self.at = session.now

        return Event(), Event()

    @staticmethod
    def elapsed_ns(start, stop):
        return stop.at - start.at

    @contextmanager
    def profile(self):
        self.spans = []
        exported = []
        window = self.now
        yield lambda: exported[0]
        events = [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": name,
                "pid": 7,
                "tid": 8,
                "ts": f"@{start / 1000:.3f}@",
                "dur": f"@{duration / 1000:.3f}@",
                "args": {"External id": number},
            }
            for number, (name, start, duration) in enumerate(self.spans)
        ]
        events.append(
            {
                "ph": "X",
                "cat": "Trace",
                "name": "PyTorch Profiler (0)",
                "pid": "Spans",
                "tid": "PyTorch Profiler",
                "ts": f"@{window / 1000:.3f}@",
                "dur": f"@{(self.now - window) / 1000:.3f}@",
                "args": {"Op count": 0},
            }
        )
        document = {
            "schemaVersion": 1,
            "host_name": PRIVATE_HOST,
            "displayTimeUnit": "ms",
            "traceEvents": events,
            "traceName": "/private/tmp/profile-trace-x/trace.json",
        }
        text = json.dumps(document, indent=1).replace('"@', "").replace('@"', "")
        exported.append(text.encode("ascii"))
        self.spans = None


def small_config(**changes):
    return config_from_mapping(make_config_mapping(**changes))


def test_a_capture_profiles_each_shape_in_the_layout_and_redacts_the_host():
    config = small_config()
    captured = capture_traces(config, FakeSession())

    assert [item.file_name for item in captured] == [
        "trace-0-64x48x32.json",
        "trace-1-33x17x9.json",
    ]
    for item in captured:
        assert isinstance(item, CapturedShape)
        assert PRIVATE_HOST.encode() not in item.trace_bytes
        assert b"/private/tmp" not in item.trace_bytes
        document = decode_trace_json(item.trace_bytes)
        assert document["host_name"] == REDACTED_HOST
        assert document["traceName"] == item.file_name
        trace = parse_trace_bytes(item.trace_bytes)
        check_layout(trace, config.iterations)
        assert sum(span.name == ITERATION for span in trace.spans) == 3
        # The events wrap each dispatch span: two annotation ticks and the call.
        assert item.device_ns == {
            side: (DISPATCH_NS[side] + 2_000,) * 3 for side in SIDES
        }
        assert item.naive_same_bits is True
        assert item.torch_disagreement == 0.0
    assert captured[0].agreement_bound > captured[1].agreement_bound


def test_warmup_iterations_stay_out_of_the_profile():
    config = small_config(warmup=4)
    item = capture_shape(FakeSession(), config, 0)
    assert len(item.device_ns["torch_matmul"]) == config.iterations


@pytest.mark.parametrize(
    "session, message",
    [
        (FakeSession(same_bits=False), "naive kernels differ"),
        (FakeSession(disagreement=1.0), "the bound is"),
        (FakeSession(disagreement=-1.0), "the bound is"),
    ],
)
def test_outputs_that_disagree_stop_the_capture(session, message):
    with pytest.raises(ProfileTraceCheckError, match=message):
        capture_shape(session, small_config(), 0)


def test_a_trace_without_the_layout_is_refused():
    session = FakeSession(nested=("matmul_x_rows/dispatch", "aten::copy_"))
    with pytest.raises(ProfileTraceError, match="must not hold child spans"):
        capture_shape(session, small_config(), 0)


def test_redaction_changes_only_the_host_fields():
    payload = (
        b'{"host_name": "a\\"b", "traceName": "/x/y.json", '
        b'"traceEvents": [{"ph": "i", "name": "host_name", "pid": 1, "tid": 1, '
        b'"ts": 1}]}'
    )
    redacted = redact_export(payload, "trace-0-1x1x1.json")
    assert decode_trace_json(redacted) == {
        "host_name": REDACTED_HOST,
        "traceName": "trace-0-1x1x1.json",
        "traceEvents": [{"ph": "i", "name": "host_name", "pid": 1, "tid": 1, "ts": 1}],
    }
    assert redact_export(b'{"traceEvents": []}', "t") == b'{"traceEvents": []}'
    with pytest.raises(ProfileTraceError, match="exactly once"):
        redact_export(b'{"host_name":"a", "traceEvents": []}', "t")


def test_the_capture_needs_a_config():
    with pytest.raises(TypeError):
        capture_traces({}, FakeSession())


@requires_metal
def test_a_real_capture_has_the_layout_and_device_times():
    config = small_config(iterations=2, warmup=1)
    session = TorchProfilerSession()
    item = capture_shape(session, config, 1)

    trace = parse_trace_bytes(item.trace_bytes)
    check_layout(trace, 2)
    names = {span.name for span in trace.spans}
    assert {"aten::matmul", "PyTorch Profiler (0)"} <= names
    assert not any("matmul_x_rows" in name and "aten" in name for name in names)
    assert all(len(values) == 2 for values in item.device_ns.values())
    assert all(value > 0 for values in item.device_ns.values() for value in values)
    assert decode_trace_json(item.trace_bytes)["host_name"] == REDACTED_HOST
    assert session.measured is True
