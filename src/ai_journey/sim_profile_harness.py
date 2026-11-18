"""Deterministic cProfile runs of the simulator, with stable function names.

``profile_call`` runs one callable under ``cProfile`` and reads the result
with ``pstats``. Each function gets a name that survives line edits:
``module:qualname`` for Python code, such as
``ai_journey.kernel_launch_buffer:DeviceBuffer.__getitem__``, and
``builtins:<built-in method _struct.pack>`` for C functions. The qualname
comes from the module's own syntax tree, because Python 3.11 profiles record
only the bare function name and the first line.

Call counts are exact and repeat from run to run, so tests pin them. The
times are the profiler's own clock readings. They include its overhead on
every call, which inflates small functions, so they only rank functions
within one profile. Speedups come from unprofiled ``perf_counter_ns``
timings instead.
"""

from __future__ import annotations

import ast
import cProfile
import functools
import pstats
import sysconfig
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .kernel_launch_config import require_int
from .sim_profile_config import MAX_TOP_FUNCTIONS, SimProfileError

PACKAGE = "ai_journey"
PACKAGE_DIR = Path(__file__).resolve().parent
NS_PER_SECOND = 1_000_000_000
PPM = 1_000_000
RANKING = (
    "functions sorted by total (self) time, largest first, ties by name; "
    "share_ppm = floor(total_ns * 1000000 / sum of total_ns of every function)"
)
# Code objects without a name of their own, as Python 3.11 names them.
_ANONYMOUS = {
    ast.Lambda: "<lambda>",
    ast.ListComp: "<listcomp>",
    ast.SetComp: "<setcomp>",
    ast.DictComp: "<dictcomp>",
    ast.GeneratorExp: "<genexpr>",
}
# The profiler records its own disable call; it is not simulator work.
_PROFILER_ENTRY = "<method 'disable' of '_lsprof.Profiler' objects>"


def _module_name(path: Path) -> str:
    if path.parent == PACKAGE_DIR:
        return f"{PACKAGE}.{path.stem}"
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    try:
        relative = path.relative_to(stdlib)
    except ValueError:
        return path.stem
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or path.stem


@functools.lru_cache(maxsize=256)
def _qualnames(filename: str) -> dict[tuple[int, str], str]:
    """Map (first line, name) of every function in a file to its qualname.

    A decorated function starts at its first decorator in a code object, so
    both that line and the ``def`` line map to it. Lambdas and comprehensions
    get the ``<locals>`` qualname that Python 3.11 gives their code objects.
    """

    try:
        tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return {}
    names: dict[tuple[int, str], str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}{child.name}"
                lines = {child.lineno, *(item.lineno for item in child.decorator_list)}
                for line in lines:
                    names[(line, child.name)] = qualname
                visit(child, f"{qualname}.<locals>.")
            elif type(child) in _ANONYMOUS:
                tag = _ANONYMOUS[type(child)]
                names[(child.lineno, tag)] = f"{prefix}{tag}"
                visit(child, f"{prefix}{tag}.<locals>.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return names


def function_name(filename: str, line: int, name: str) -> str:
    """Return the stable ``module:qualname`` name of one pstats entry."""

    if filename == "~":
        return f"builtins:{name}"
    path = Path(filename)
    if path.suffix != ".py" or not path.is_absolute():
        # Code compiled from a string, such as a dataclass __init__.
        return f"{filename}:{name}"
    resolved = path.resolve()
    qualname = _qualnames(str(resolved)).get((line, name), name)
    return f"{_module_name(resolved)}:{qualname}"


@dataclass(frozen=True)
class FunctionProfile:
    """The exact call counts and profiler times of one function in one run."""

    function: str
    calls: int
    primitive_calls: int
    total_ns: int
    cumulative_ns: int

    def __post_init__(self) -> None:
        if type(self.function) is not str or not 1 <= len(self.function) <= 256:
            raise SimProfileError("a function name must be 1 to 256 characters")
        require_int(self.calls, "calls", minimum=1, maximum=2**62)
        require_int(
            self.primitive_calls, "primitive_calls", minimum=1, maximum=self.calls
        )
        for field in ("total_ns", "cumulative_ns"):
            require_int(getattr(self, field), field, minimum=0, maximum=2**62)

    def to_mapping(self) -> dict[str, object]:
        return {
            "function": self.function,
            "calls": self.calls,
            "primitive_calls": self.primitive_calls,
            "total_ns": self.total_ns,
            "cumulative_ns": self.cumulative_ns,
        }


def _ns(seconds: float) -> int:
    return max(0, round(seconds * NS_PER_SECOND))


def merge_functions(records: Iterable[FunctionProfile]) -> tuple[FunctionProfile, ...]:
    """Sum the entries that share a name; return them sorted by name.

    Two entries share a name when one function runs in several workloads, or
    when one file defines two functions with one qualname.
    """

    totals: dict[str, list[int]] = {}
    for record in records:
        if type(record) is not FunctionProfile:
            raise TypeError("records must be FunctionProfile values")
        entry = totals.setdefault(record.function, [0, 0, 0, 0])
        entry[0] += record.calls
        entry[1] += record.primitive_calls
        entry[2] += record.total_ns
        entry[3] += record.cumulative_ns
    return tuple(FunctionProfile(name, *totals[name]) for name in sorted(totals))


def profile_call(
    call: Callable[[], object],
) -> tuple[tuple[FunctionProfile, ...], object]:
    """Run ``call`` once under cProfile; return its functions and its result."""

    if not callable(call):
        raise TypeError("call must be callable")
    profiler = cProfile.Profile()
    result = profiler.runcall(call)
    stats = pstats.Stats(profiler).stats
    records = [
        FunctionProfile(
            function_name(filename, line, name),
            calls,
            primitive,
            _ns(total),
            _ns(cumulative),
        )
        for (filename, line, name), (primitive, calls, total, cumulative, _) in (
            stats.items()
        )
        if not (filename == "~" and name == _PROFILER_ENTRY)
    ]
    return merge_functions(records), result


@dataclass(frozen=True)
class Hotspot:
    """One function of a ranked hotspot list and its share of the self time."""

    rank: int
    function: str
    calls: int
    total_ns: int
    cumulative_ns: int
    share_ppm: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "function": self.function,
            "calls": self.calls,
            "total_ns": self.total_ns,
            "cumulative_ns": self.cumulative_ns,
            "share_ppm": self.share_ppm,
        }


def rank_hotspots(
    records: tuple[FunctionProfile, ...], top: int
) -> tuple[Hotspot, ...]:
    """Rank functions by self time and keep the first ``top``; see ``RANKING``."""

    require_int(top, "top", minimum=1, maximum=MAX_TOP_FUNCTIONS)
    if type(records) is not tuple or not records:
        raise SimProfileError("a ranking needs at least one function")
    if any(type(record) is not FunctionProfile for record in records):
        raise TypeError("records must be FunctionProfile values")
    grand = sum(record.total_ns for record in records)
    if grand <= 0:
        raise SimProfileError("the profile recorded no time")
    ordered = sorted(records, key=lambda record: (-record.total_ns, record.function))
    return tuple(
        Hotspot(
            rank=rank,
            function=record.function,
            calls=record.calls,
            total_ns=record.total_ns,
            cumulative_ns=record.cumulative_ns,
            share_ppm=record.total_ns * PPM // grand,
        )
        for rank, record in enumerate(ordered[:top], start=1)
    )
