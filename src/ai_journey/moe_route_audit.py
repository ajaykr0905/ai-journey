"""Recompute bounded top-two routing evidence without trusting derived counts.

This validates the declared token-major capacity policy and internal
conservation. It does not authenticate submitted observations or logits.
"""

from collections.abc import Mapping
import math

MAX_ROUTING_TOKENS = 65_536
_FIELDS = {
    "layer_index",
    "valid_tokens",
    "capacity",
    "attempted_counts",
    "accepted_counts",
    "dropped_counts",
    "probability_sums",
    "capacity_applied",
    "accepted_per_token",
    "token_indices",
    "selected_experts",
    "accepted_routes",
}


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"routing {name} must be an integer in [{minimum}, {maximum}]")
    return value


def _rows(value, name, count, width):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"routing {name} has an invalid row count")
    if any(not isinstance(row, (list, tuple)) or len(row) != width for row in value):
        raise ValueError(f"routing {name} has invalid row dimensions")
    return value


def validate_moe_routing(stats, *, expert_count, top_k=2):
    """Validate one model layer's JSON-safe route evidence and return totals."""
    _integer(expert_count, "expert_count", 2, 64)
    if type(top_k) is not int or top_k != 2:
        raise ValueError("routing evidence requires top_k=2")
    if not isinstance(stats, Mapping) or set(stats) != _FIELDS:
        raise ValueError("routing evidence fields do not match")
    layer = _integer(stats["layer_index"], "layer_index", 0, 63)
    tokens = _integer(stats["valid_tokens"], "valid_tokens", 0, MAX_ROUTING_TOKENS)
    capacity = _integer(stats["capacity"], "capacity", 0, 2_000_000)
    if (tokens == 0) != (capacity == 0):
        raise ValueError("routing empty-set capacity is inconsistent")
    applied = stats["capacity_applied"]
    if type(applied) is not bool:
        raise ValueError("routing capacity_applied must be boolean")
    matrix = stats["accepted_per_token"]
    if not isinstance(matrix, (tuple, list)) or not 1 <= len(matrix) <= 64:
        raise ValueError("routing accepted_per_token must be a bounded token matrix")
    width = len(matrix[0]) if isinstance(matrix[0], (list, tuple)) else 0
    if not 1 <= width <= 4096 or width * len(matrix) > MAX_ROUTING_TOKENS:
        raise ValueError("routing accepted_per_token exceeds its cell budget")
    _rows(matrix, "accepted_per_token", len(matrix), width)
    flattened = [
        _integer(value, "accepted_per_token", 0, 2) for row in matrix for value in row
    ]
    indices = stats["token_indices"]
    if not isinstance(indices, (list, tuple)) or len(indices) != tokens:
        raise ValueError("routing token_indices must match valid_tokens")
    for index, value in enumerate(indices):
        _integer(value, "token_index", 0, len(flattened) - 1)
        if index and value <= indices[index - 1]:
            raise ValueError("routing token_indices must be sorted and unique")
    selected = _rows(stats["selected_experts"], "selected_experts", tokens, 2)
    flags = _rows(stats["accepted_routes"], "accepted_routes", tokens, 2)
    attempted, accepted = [0] * expert_count, [0] * expert_count
    per_token = [0] * len(flattened)
    for position, expert_row, accepted_row in zip(
        indices, selected, flags, strict=True
    ):
        if expert_row[0] == expert_row[1]:
            raise ValueError("routing must select two distinct experts per token")
        for expert, flag in zip(expert_row, accepted_row, strict=True):
            _integer(expert, "selected_expert", 0, expert_count - 1)
            if type(flag) is not bool:
                raise ValueError("routing accepted route flags must be boolean")
            expected = not applied or attempted[expert] < capacity
            if flag != expected:
                raise ValueError("routing flags violate token-major capacity admission")
            attempted[expert] += 1
            accepted[expert] += int(flag)
            per_token[position] += int(flag)
    if per_token != flattened:
        raise ValueError("routing per-token accepted counts do not match routes")
    dropped = [
        before - after for before, after in zip(attempted, accepted, strict=True)
    ]
    for name, expected in (
        ("attempted_counts", attempted),
        ("accepted_counts", accepted),
        ("dropped_counts", dropped),
    ):
        values = stats[name]
        if not isinstance(values, (list, tuple)) or len(values) != expert_count:
            raise ValueError(f"routing {name} does not match expert_count")
        for value in values:
            _integer(value, name, 0, tokens)
        if list(values) != expected:
            raise ValueError(f"routing {name} does not match individual routes")
    probabilities = stats["probability_sums"]
    if (
        not isinstance(probabilities, (list, tuple))
        or len(probabilities) != expert_count
    ):
        raise ValueError("routing probability_sums do not match expert_count")
    mass = []
    for value in probabilities:
        if type(value) not in (int, float):
            raise ValueError("routing probability mass must be finite numeric values")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("routing probability mass exceeds numeric range") from exc
        if not math.isfinite(number) or not 0 <= number <= tokens + 2e-5:
            raise ValueError("routing probability mass is outside valid-token exposure")
        mass.append(number)
    if not math.isclose(math.fsum(mass), tokens, rel_tol=2e-5, abs_tol=2e-5):
        raise ValueError("routing probability mass must sum to valid-token exposure")
    valid_counts = [per_token[index] for index in indices]
    return {
        "layer_index": layer,
        "valid_tokens": tokens,
        "attempted_routes": sum(attempted),
        "accepted_routes": sum(accepted),
        "dropped_routes": sum(dropped),
        "unrouted_tokens": valid_counts.count(0),
        "partially_routed_tokens": valid_counts.count(1),
        "fully_routed_tokens": valid_counts.count(2),
        "capacity_applied": applied,
    }
