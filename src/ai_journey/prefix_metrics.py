"""Fixed-cardinality prefix observations, separate from whole-response replay."""

import threading


class PrefixMetrics:
    def __init__(self, store):
        self.store = store
        self._lock = threading.Lock()
        self._attempts = dict.fromkeys(("hit", "miss", "bypass"), 0)
        self._reused = self._forwarded = 0

    def record(self, observation):
        outcome = observation.get("outcome", "unused")
        if outcome == "unused":
            return
        reused, forwarded = (
            observation.get("reused_prompt_tokens", 0),
            observation.get("forwarded_prefill_tokens", 0),
        )
        if outcome not in self._attempts or any(
            type(value) is not int or not 0 <= value <= 4096
            for value in (reused, forwarded)
        ):
            raise ValueError("invalid bounded prefix observation")
        with self._lock:
            self._attempts[outcome] += 1
            self._reused += reused
            self._forwarded += forwarded

    def snapshot(self):
        with self._lock:
            result = {
                "attempts": dict(self._attempts),
                "reused_prompt_tokens": self._reused,
                "forwarded_prefill_tokens": self._forwarded,
            }
        # Pool gauges are a separate instantaneous snapshot, not a transactional
        # claim about requests currently executing between prefill and recording.
        return {**result, "pool": self.store.snapshot()}

    def render_prometheus(self):
        state = self.snapshot()
        lines = ["# TYPE ai_journey_prefix_attempts_total counter"]
        lines.extend(
            f'ai_journey_prefix_attempts_total{{outcome="{outcome}"}} {count}'
            for outcome, count in state["attempts"].items()
        )
        for field in ("reused_prompt_tokens", "forwarded_prefill_tokens"):
            name = f"ai_journey_prefix_{field}_total"
            lines += [f"# TYPE {name} counter", f"{name} {state[field]}"]
        for field in (
            "resident_bytes",
            "leased_bytes",
            "reserved_bytes",
            "active_leases",
            "entries",
            "max_bytes",
            "max_entries",
        ):
            name = f"ai_journey_prefix_{field}"
            lines += [f"# TYPE {name} gauge", f"{name} {state['pool'][field]}"]
        return "\n".join(lines) + "\n"
