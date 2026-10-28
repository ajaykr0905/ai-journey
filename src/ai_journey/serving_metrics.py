"""Fixed-cardinality serving metrics in Prometheus text exposition format."""

from bisect import bisect_left
from itertools import accumulate
import math
import threading


OUTCOMES = ("success", "cancelled", "deadline", "error", "rejected")
DURATION_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_MAX_COUNTER = 2**63 - 1


def _duration_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        duration = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return duration


def _histogram_snapshot(bins, total):
    cumulative = tuple(accumulate(bins))
    return {
        "buckets": tuple(zip((*DURATION_BUCKETS, math.inf), cumulative, strict=True)),
        "sum": total,
        "count": cumulative[-1],
    }


class ServingMetrics:
    """Record completed operations atomically without retaining request content.

    ``generated_tokens`` is the emitted output length. Cached responses count as
    cache hits, but do not add newly generated tokens. Cancelled and failed fresh
    operations can report tokens generated before they stopped.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._requests = dict.fromkeys(OUTCOMES, 0)
        self._duration_bins = [0] * (len(DURATION_BUCKETS) + 1)
        self._duration_sum = 0.0
        self._queue_wait_bins = [0] * (len(DURATION_BUCKETS) + 1)
        self._queue_wait_sum = 0.0
        self._generated_tokens = 0
        self._cache_hits = 0

    def record(self, *, outcome, duration_seconds, generated_tokens, cached=False):
        """Reject invalid observations before changing any counter or bucket."""
        if not isinstance(outcome, str) or outcome not in OUTCOMES:
            raise ValueError("outcome must be a fixed serving outcome")
        duration = _duration_seconds(duration_seconds, "duration_seconds")
        if (
            type(generated_tokens) is not int
            or not 0 <= generated_tokens <= _MAX_COUNTER
        ):
            raise ValueError("generated_tokens must be a nonnegative 64-bit integer")
        if type(cached) is not bool:
            raise ValueError("cached must be boolean")
        new_tokens = 0 if cached else generated_tokens
        bucket = bisect_left(DURATION_BUCKETS, duration)
        with self._lock:
            total_duration = self._duration_sum + duration
            if (
                not math.isfinite(total_duration)
                or self._generated_tokens + new_tokens > _MAX_COUNTER
                or sum(self._requests.values()) == _MAX_COUNTER
            ):
                raise ValueError("serving metric accumulation exceeds finite capacity")
            self._requests[outcome] += 1
            self._duration_bins[bucket] += 1
            self._duration_sum = total_duration
            self._generated_tokens += new_tokens
            self._cache_hits += int(cached)

    def observe_queue_wait(self, duration_seconds):
        """Observe each admitted wait ending in execution or queued interruption."""
        duration = _duration_seconds(duration_seconds, "queue wait")
        bucket = bisect_left(DURATION_BUCKETS, duration)
        with self._lock:
            total = self._queue_wait_sum + duration
            if not math.isfinite(total) or sum(self._queue_wait_bins) == _MAX_COUNTER:
                raise ValueError("serving metric accumulation exceeds finite capacity")
            self._queue_wait_bins[bucket] += 1
            self._queue_wait_sum = total

    def snapshot(self) -> dict:
        """Return an independent, coherent copy of all metric families."""
        with self._lock:
            return {
                "requests": self._requests.copy(),
                "duration_seconds": _histogram_snapshot(
                    self._duration_bins, self._duration_sum
                ),
                "queue_wait_seconds": _histogram_snapshot(
                    self._queue_wait_bins, self._queue_wait_sum
                ),
                "generated_tokens": self._generated_tokens,
                "cache_hits": self._cache_hits,
            }

    def render_prometheus(self, *, operational_state=None) -> str:
        """Render one coherent scrape without prompt or request-ID labels."""
        values = self.snapshot()
        lines = [
            "# HELP ai_journey_serving_requests_total Completed serving operations by outcome.",
            "# TYPE ai_journey_serving_requests_total counter",
        ]
        for outcome in OUTCOMES:
            lines.append(
                f'ai_journey_serving_requests_total{{outcome="{outcome}"}} {values["requests"][outcome]}'
            )
        for name, key, help_text in (
            (
                "generated_tokens",
                "generated_tokens",
                "New tokens generated excluding response-cache hits.",
            ),
            (
                "cache_hits",
                "cache_hits",
                "Serving operations that used the response cache.",
            ),
        ):
            metric = f"ai_journey_serving_{name}_total"
            lines.extend(
                (
                    f"# HELP {metric} {help_text}",
                    f"# TYPE {metric} counter",
                    f"{metric} {values[key]}",
                )
            )
        for name, key, help_text in (
            (
                "request_duration_seconds",
                "duration_seconds",
                "Completed serving operation duration in seconds.",
            ),
            (
                "queue_wait_seconds",
                "queue_wait_seconds",
                "Admitted wait until execution or queued interruption in seconds.",
            ),
        ):
            histogram = f"ai_journey_serving_{name}"
            lines.extend(
                (f"# HELP {histogram} {help_text}", f"# TYPE {histogram} histogram")
            )
            duration = values[key]
            for bound, count in duration["buckets"]:
                label = "+Inf" if math.isinf(bound) else str(bound)
                lines.append(f'{histogram}_bucket{{le="{label}"}} {count}')
            lines.extend(
                (
                    f'{histogram}_sum {duration["sum"]}',
                    f'{histogram}_count {duration["count"]}',
                )
            )
        if operational_state is not None:
            self._append_operational_gauges(lines, operational_state)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _append_operational_gauges(lines, state):
        names = (
            "active_requests",
            "running_requests",
            "queued_requests",
            "request_capacity",
        )
        if (
            not isinstance(state, dict)
            or any(
                type(state.get(name)) is not int or not 0 <= state[name] <= 64
                for name in names
            )
            or type(state.get("draining")) is not bool
        ):
            raise ValueError(
                "operational state requires bounded counts and boolean draining"
            )
        if not (
            state["active_requests"]
            == state["running_requests"] + state["queued_requests"]
            and state["active_requests"] <= state["request_capacity"]
            and state["request_capacity"] >= 1
        ):
            raise ValueError("operational state counts must fit admission capacity")
        for name, help_text in (
            ("active_requests", "Admitted requests currently running or waiting."),
            ("running_requests", "Requests currently holding an inference slot."),
            ("queued_requests", "Admitted requests waiting for an inference slot."),
            ("request_capacity", "Maximum running plus waiting request count."),
            ("draining", "Whether new inference admission has stopped."),
        ):
            metric = f"ai_journey_serving_{name}"
            lines.extend(
                (
                    f"# HELP {metric} {help_text}",
                    f"# TYPE {metric} gauge",
                    f"{metric} {int(state[name])}",
                )
            )
