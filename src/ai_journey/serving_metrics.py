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
        self._generated_tokens = 0
        self._cache_hits = 0

    def record(self, *, outcome, duration_seconds, generated_tokens, cached=False):
        """Reject invalid observations before changing any counter or bucket."""
        if not isinstance(outcome, str) or outcome not in OUTCOMES:
            raise ValueError("outcome must be a fixed serving outcome")
        if isinstance(duration_seconds, bool) or not isinstance(
            duration_seconds, (int, float)
        ):
            raise ValueError("duration_seconds must be finite and nonnegative")
        try:
            duration = float(duration_seconds)
        except OverflowError as error:
            raise ValueError(
                "duration_seconds must be finite and nonnegative"
            ) from error
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_seconds must be finite and nonnegative")
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

    def snapshot(self) -> dict:
        """Return an independent, coherent copy of all metric families."""
        with self._lock:
            cumulative = tuple(accumulate(self._duration_bins))
            return {
                "requests": self._requests.copy(),
                "duration_seconds": {
                    "buckets": tuple(
                        zip((*DURATION_BUCKETS, math.inf), cumulative, strict=True)
                    ),
                    "sum": self._duration_sum,
                    "count": cumulative[-1],
                },
                "generated_tokens": self._generated_tokens,
                "cache_hits": self._cache_hits,
            }

    def render_prometheus(self) -> str:
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
        histogram = "ai_journey_serving_request_duration_seconds"
        lines.extend(
            (
                f"# HELP {histogram} Completed serving operation duration in seconds.",
                f"# TYPE {histogram} histogram",
            )
        )
        duration = values["duration_seconds"]
        for bound, count in duration["buckets"]:
            label = "+Inf" if math.isinf(bound) else str(bound)
            lines.append(f'{histogram}_bucket{{le="{label}"}} {count}')
        lines.extend(
            (
                f'{histogram}_sum {duration["sum"]}',
                f'{histogram}_count {duration["count"]}',
            )
        )
        return "\n".join(lines) + "\n"
