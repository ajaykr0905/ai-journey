"""Day 11 executable Python fluency drills."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from functools import wraps
from typing import TypeVar

R = TypeVar("R")


def even_squares(values: Iterable[int]) -> list[int]:
    """List-comprehension drill: square only even inputs."""

    return [value * value for value in values if value % 2 == 0]


def running_total(values: Iterable[int]) -> Generator[int, None, None]:
    """Generator drill: yield each cumulative sum without storing all results."""

    total = 0
    for value in values:
        total += value
        yield total


def count_calls(function: Callable[..., R]) -> Callable[..., R]:
    """Typed decorator drill: expose a ``calls`` counter on the wrapper."""

    @wraps(function)
    def wrapper(*args: object, **kwargs: object) -> R:
        wrapper.calls += 1
        return function(*args, **kwargs)

    wrapper.calls = 0  # type: ignore[attr-defined]
    return wrapper


@count_calls
def typed_add(left: int, right: int) -> int:
    """Typing drill used by the all-days runner and unit tests."""

    return left + right


def run_drills() -> dict[str, object]:
    """Execute each drill and return a compact, inspectable result."""

    before = typed_add.calls  # type: ignore[attr-defined]
    sum_result = typed_add(3, 4)
    after = typed_add.calls  # type: ignore[attr-defined]
    return {
        "even_squares": even_squares(range(7)),
        "running_total": list(running_total([1, 2, 3, 4])),
        "typed_add": sum_result,
        "decorator_increment": after - before,
    }
