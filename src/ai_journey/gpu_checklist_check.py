"""Compile-free checks of the Day 77 coarsened and register-blocked CUDA texts.

No compiler exists on this host, so the checker reads a text the way the Day
76 checker does: comments as a compiler reads them, no line splice, trigraph
or space inside an operator, and only the three pinned definitions. It then
parses the kernel body into a tree of statements, ``for`` loops and ``if``
blocks, and it parses the pinned text of the same kernel with the same
parser. Each named check compares one part of the two trees, so a changed
statement fails the check of its part.

Some checks also apply a rule that does not use the pinned text:

* shared memory: the extents of each ``__shared__`` tile, read from the text,
  must give the declared size, inside the 48 KB static limit of compute
  capability 8.0;
* barriers: no barrier is inside an ``if``, and each loop around a barrier
  uses only ``k``, ``m``, ``n``, constants and the variables of such loops,
  so every thread of a block reaches each barrier equally often. A static
  scan also runs each loop body twice, so the last access of one iteration
  meets the first access of the next, and it requires a barrier between each
  write of a shared tile and a later read of it, and between each read and a
  later write. The scan reads each assignment to a shared element as a write
  and each other use of a shared tile as a read;
* fma order and stores: no statement other than the pinned ones names an
  accumulator, and only the guarded store touches ``c``;
* grid: the host ``dim3`` expressions are evaluated with C unsigned division,
  and they must give the launch of the kernel's model for the shape.

The checker proves nothing about ``nvcc``: a text that passes can still fail
to compile, and only a later NVIDIA run can show that it does not.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass

from .gpu_checklist_cuda import (
    MAX_STATIC_SHARED_BYTES,
    BlockedKernel,
    ChecklistKernel,
    CoarsenedKernel,
    GpuChecklistError,
    cuda_source,
    require_kernel,
    require_shape,
)
from .gpu_tiled_config import PINNED_PLAN
from .gpu_tiled_cuda import _SHARED, FLOAT_BYTES, _comparable_code, _definitions
from .kernel_launch_config import Dim3, KernelLaunchError, LaunchConfig
from .kernel_launch_cuda import _CDIV, _DECIMAL, _braced, _cdiv, _normalize
from .memory_access_tiling import MatmulShape

COARSENED_CHECK_NAMES = (
    "cdiv_definition",
    "coarsened_signature",
    "coarsened_shared_memory",
    "coarsened_index",
    "coarsened_phase_loop",
    "coarsened_loads",
    "coarsened_barriers",
    "coarsened_inner_loop",
    "coarsened_fma_order",
    "coarsened_store_guard",
    "coarsened_grid",
)
BLOCKED_CHECK_NAMES = (
    "cdiv_definition",
    "blocked_signature",
    "blocked_shared_memory",
    "blocked_index",
    "blocked_vector_alignment",
    "blocked_phase_loop",
    "blocked_loads",
    "blocked_barriers",
    "blocked_inner_loop",
    "blocked_fma_order",
    "blocked_store_guard",
    "blocked_grid",
)
# The shapes of config/day-74-matmul-metal.json: the correctness cases, which
# the Day 76 plan repeats, and the benchmark points that a later run times.
CORRECTNESS_SHAPES = PINNED_PLAN.shapes
BENCHMARK_SHAPES = tuple(
    MatmulShape(*triple)
    for triple in (
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 1024, 1024),
        (1024, 4096, 2048),
        (2000, 3000, 1000),
    )
)
CHECK_SHAPES = (*CORRECTNESS_SHAPES, *BENCHMARK_SHAPES)
BARRIER = "__syncthreads();"
_ACCUMULATORS = {CoarsenedKernel: ("sum",), BlockedKernel: ("sum", "a_reg", "b_reg")}
_SIGNATURE = "(const float* a, const float* b, float* c, int m, int n, int k)"
_OPENERS = (("for(", "for"), ("if(", "if"))
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LOOP_VARIABLE = re.compile(r"for\(int ([A-Za-z_][A-Za-z0-9_]*)=")
_UNIFORM_NAMES = frozenset({"int", "k", "m", "n", "min"})
_EXIT = re.compile(
    r"(?<![A-Za-z0-9_])(return|break|continue|goto|while|do|switch)(?![A-Za-z0-9_])"
)
_FLOAT4_LOAD = "reinterpret_cast<const float4*>"
_C_ELEMENT = re.compile(r"(?<![A-Za-z0-9_])c\[")
_ALIGNMENT_STATEMENTS = (
    "int a_c=",
    "int b_c=",
    "int a_row=",
    "int b_col=",
    "int a_col=",
    "int b_row=",
)
_LAUNCHER = re.compile(
    rf"dim3 block\({_DECIMAL},{_DECIMAL}\);"
    rf"dim3 grid\(cdiv\(n,{_DECIMAL}\),cdiv\(m,{_DECIMAL}\)\);"
    r"([A-Za-z_][A-Za-z0-9_]*)<<<grid,block>>>\(a,b,c,m,n,k\);"
)


@dataclass(frozen=True)
class Node:
    """One statement, ``for`` loop or ``if`` block of a normalized kernel body.

    ``head`` is the statement, or the loop or guard header. ``orelse`` holds
    the ``else`` block of an ``if``, and it is ``None`` when there is none.
    """

    kind: str
    head: str
    body: tuple[Node, ...] = ()
    orelse: tuple[Node, ...] | None = None


def _close_parenthesis(code: str, start: int) -> int | None:
    depth = 0
    for position in range(start, len(code)):
        if code[position] == "(":
            depth += 1
        elif code[position] == ")":
            depth -= 1
            if depth == 0:
                return position + 1
    return None


def _statement_end(code: str, start: int) -> int | None:
    """Return the end of a simple statement; a brace in it is refused."""

    depth = 0
    for position in range(start, len(code)):
        character = code[position]
        if character in "{}":
            return None
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == ";" and depth == 0:
            return position + 1
    return None


def parse_body(code: str) -> tuple[Node, ...] | None:
    """Parse normalized code into nodes, or return ``None`` for other code.

    Every loop and guard must have a braced body, and an ``else`` must have
    one too. Any other brace, such as a ``while`` body or an initializer
    list, refuses the whole body.
    """

    nodes: list[Node] = []
    position = 0
    while position < len(code):
        kind = next(
            (name for opener, name in _OPENERS if code.startswith(opener, position)),
            None,
        )
        if kind is None:
            end = _statement_end(code, position)
            if end is None:
                return None
            nodes.append(Node("statement", code[position:end]))
            position = end
            continue
        close = _close_parenthesis(code, position + len(kind))
        if close is None or not code.startswith("{", close):
            return None
        inside = _braced(code, close)
        body = None if inside is None else parse_body(inside)
        if body is None:
            return None
        head = code[position:close]
        position = close + len(inside) + 2
        orelse = None
        if kind == "if" and code.startswith("else{", position):
            other = _braced(code, position + 4)
            orelse = None if other is None else parse_body(other)
            if orelse is None:
                return None
            position += len("else") + len(other) + 2
        nodes.append(Node(kind, head, body, orelse))
    return tuple(nodes)


def _children(node: Node) -> tuple[Node, ...]:
    return node.body + (() if node.orelse is None else node.orelse)


def _texts(nodes: tuple[Node, ...]) -> tuple[str, ...]:
    """Return every statement and header, depth first, in program order."""

    texts: list[str] = []
    for node in nodes:
        texts.append(node.head)
        texts.extend(_texts(_children(node)))
    return tuple(texts)


def _names(text: str, name: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])", text) is not None


def _single(nodes: tuple[Node, ...], prefix: str) -> Node | None:
    """Return the one ``for`` loop whose header starts with ``prefix``."""

    found = [
        node for node in nodes if node.kind == "for" and node.head.startswith(prefix)
    ]
    return found[0] if len(found) == 1 else None


def _statements(nodes: tuple[Node, ...]) -> tuple[str, ...]:
    """Return the simple statements of one block; the barrier check owns barriers."""

    return tuple(
        node.head for node in nodes if node.kind == "statement" and node.head != BARRIER
    )


def _layout(nodes: tuple[Node, ...]) -> tuple[int, int] | None:
    """Return (statements, loops) if every statement comes before every loop."""

    kinds = [node.kind for node in nodes]
    statements = kinds.count("statement")
    if "if" in kinds or kinds != ["statement"] * statements + ["for"] * (
        len(kinds) - statements
    ):
        return None
    return statements, len(kinds) - statements


def _skeleton(nodes: tuple[Node, ...]) -> tuple[object, ...]:
    """Return the barriers and the blocks that hold one, without other code."""

    items: list[object] = []
    for node in nodes:
        if node.kind == "statement":
            if node.head == BARRIER:
                items.append("barrier")
            continue
        inner = _skeleton(node.body)
        other = () if node.orelse is None else _skeleton(node.orelse)
        if inner or other:
            items.append((node.kind, inner, other))
    return tuple(items)


def _has_barrier(nodes: tuple[Node, ...]) -> bool:
    return any(BARRIER in text for text in _texts(nodes))


def _uniform_barriers(nodes: tuple[Node, ...], uniform: frozenset[str]) -> bool:
    """Return whether every thread of a block reaches each barrier equally often."""

    for node in nodes:
        if node.kind == "statement" or not _has_barrier(_children(node)):
            continue
        if node.kind == "if":
            return False
        variable = _LOOP_VARIABLE.match(node.head)
        if variable is None:
            return False
        own = uniform | {variable.group(1)}
        if not set(_IDENTIFIER.findall(node.head[len("for") :])) <= own:
            return False
        if not _uniform_barriers(node.body, own):
            return False
    return True


def _write_target(text: str) -> tuple[str, bool] | None:
    """Return the array that a statement assigns and whether it reads it too."""

    name = _IDENTIFIER.match(text)
    position = 0 if name is None else name.end()
    if name is None or not text.startswith("[", position):
        return None
    while text.startswith("[", position):
        depth = 0
        for index in range(position, len(text)):
            if text[index] == "[":
                depth += 1
            elif text[index] == "]":
                depth -= 1
                if depth == 0:
                    break
        else:
            return None
        position = index + 1
    rest = text[position:]
    if rest.startswith("=") and not rest.startswith("=="):
        return name.group(), False
    if rest[:2] in ("+=", "-=", "*=", "/="):
        return name.group(), True
    return None


def _accesses(text: str, shared: tuple[str, ...]) -> list[tuple[str, str]]:
    """Return the reads of shared tiles in a statement, then its shared write.

    A declaration of a tile accesses no slot.
    """

    if "__shared__" in text:
        return []
    target = _write_target(text)
    written = None if target is None or target[0] not in shared else target[0]
    rest = text if written is None else text[len(written) :]
    events = [
        ("read", name)
        for name in shared
        if re.search(rf"(?<![A-Za-z0-9_]){name}\[", rest)
        or (name == written and target[1])
    ]
    if written is not None:
        events.append(("write", written))
    return events


def _events(nodes: tuple[Node, ...], shared: tuple[str, ...]) -> list[tuple[str, str]]:
    """Return the shared accesses and barriers in program order.

    A loop body appears twice, so the end of one iteration meets the start
    of the next. An ``if`` gives its condition and then both branches.
    """

    events: list[tuple[str, str]] = []
    for node in nodes:
        if node.kind == "statement":
            if node.head == BARRIER:
                events.append(("barrier", ""))
            else:
                events.extend(_accesses(node.head, shared))
            continue
        events.extend(_accesses(node.head, shared))
        inner = _events(node.body, shared)
        if node.kind == "for":
            events.extend(inner + inner)
        else:
            events.extend(inner)
            events.extend(() if node.orelse is None else _events(node.orelse, shared))
    return events


def shared_hazards(nodes: tuple[Node, ...], shared: tuple[str, ...]) -> int:
    """Count the write-read and read-write pairs of a tile with no barrier between.

    A write that follows a write is not counted: in each phase, each slot of a
    tile has one writer, which the CPU block simulator confirms.
    """

    hazards = 0
    written: set[str] = set()
    read: set[str] = set()
    for kind, name in _events(nodes, shared):
        if kind == "barrier":
            written.clear()
            read.clear()
        elif kind == "read":
            hazards += name in written
            read.add(name)
        else:
            hazards += name in read
            written.add(name)
    return hazards


def _shared_tiles(code: str) -> tuple[tuple[str, int, int], ...] | None:
    """Return each ``__shared__ float`` tile and its extents, or ``None``.

    Any other ``__shared__`` declaration, such as an ``extern`` array, makes
    the size unknown.
    """

    tiles = tuple(
        (name, int(rows), int(cols)) for name, rows, cols in _SHARED.findall(code)
    )
    return tiles if len(tiles) == code.count("__shared__") else None


def _shared_names(nodes: tuple[Node, ...]) -> tuple[str, ...]:
    """Return the name of each array that a ``__shared__`` declaration makes."""

    names = []
    for text in _texts(nodes):
        found = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\[", text)
        if "__shared__" in text and found is not None:
            names.append(found.group(1))
    return tuple(dict.fromkeys(names))


def _tile_extent(kernel: ChecklistKernel) -> tuple[int, int]:
    """Return the (columns, rows) of C that one block computes."""

    if type(kernel) is CoarsenedKernel:
        return kernel.width, kernel.tile
    return kernel.block_n, kernel.block_m


def _host_launch(
    host: str, kernel: ChecklistKernel, shape: MatmulShape
) -> tuple[LaunchConfig | None, bool]:
    """Evaluate a launcher that is one block, one grid and one launch.

    The grid divisors must be the C tile of one block, so a divisor that
    agrees with the model for one shape only still fails.
    """

    found = _LAUNCHER.fullmatch(host)
    if found is None or found.group(5) != kernel.kernel_name:
        return None, False
    block_x, block_y, divisor_x, divisor_y = (int(item) for item in found.groups()[:4])
    try:
        launch = LaunchConfig(
            Dim3(_cdiv(shape.n, divisor_x), _cdiv(shape.m, divisor_y)),
            Dim3(block_x, block_y),
        )
    except (KernelLaunchError, TypeError):
        return None, False
    return launch, (divisor_x, divisor_y) == _tile_extent(kernel)


def _signatures(kernel: ChecklistKernel) -> tuple[str, str, str]:
    return (
        _CDIV,
        f"__global__ void {kernel.kernel_name}{_SIGNATURE}",
        f"void {kernel.launcher_name}{_SIGNATURE}",
    )


def _read(text: str, kernel: ChecklistKernel) -> tuple | None:
    """Return the cdiv, kernel and launcher code and the kernel tree, or ``None``."""

    if type(text) is not str:
        raise TypeError("CUDA source must be a string")
    code = _comparable_code(text)
    bodies = (
        None if code is None else _definitions(_normalize(code), _signatures(kernel))
    )
    nodes = None if bodies is None else parse_body(bodies[1])
    return None if nodes is None else (*bodies, nodes)


def parse_kernel(text: str, *, kernel: ChecklistKernel) -> tuple[Node, ...] | None:
    """Return the tree of the kernel body of a text, or ``None`` if it fails to read."""

    parts = _read(text, require_kernel(kernel))
    return None if parts is None else parts[3]


@functools.cache
def _pinned_parts(kernel: ChecklistKernel) -> tuple[tuple[Node, ...], str]:
    """Return the tree of the pinned kernel body, and its code."""

    parts = _read(cuda_source(kernel), kernel)
    if parts is None:
        raise GpuChecklistError(f"the pinned text of {kernel.name} does not parse")
    return parts[3], parts[1]


def _accumulator_uses(nodes: tuple[Node, ...], names: tuple[str, ...]) -> tuple:
    """Return each text that names an accumulator, except the stores to ``c``."""

    return tuple(
        text
        for text in _texts(nodes)
        if any(_names(text, name) for name in names) and not _names(text, "c")
    )


def _float4_sites(nodes: tuple[Node, ...], guards: tuple[str, ...] = ()) -> tuple:
    """Return each float4 load with the guards and branches around it."""

    sites: list[object] = []
    for node in nodes:
        if node.kind == "statement":
            if _FLOAT4_LOAD in node.head:
                sites.append((guards, node.head))
            continue
        if node.kind == "if":
            sites.extend(_float4_sites(node.body, (*guards, f"{node.head}:then")))
            sites.extend(
                _float4_sites(node.orelse or (), (*guards, f"{node.head}:else"))
            )
        else:
            sites.extend(_float4_sites(node.body, guards))
    return tuple(sites)


def _tile_bytes(tiles: tuple[tuple[str, int, int], ...] | None) -> int | None:
    if tiles is None:
        return None
    return sum(rows * cols * FLOAT_BYTES for _, rows, cols in tiles)


def _phase(nodes: tuple[Node, ...]) -> Node | None:
    return _single(nodes, "for(int phase=")


@dataclass(frozen=True)
class _Pair:
    """The tree of a kernel body next to the tree of its pinned text."""

    kernel: ChecklistKernel
    top: tuple[Node, ...]
    code: str

    @property
    def pinned(self) -> tuple[Node, ...]:
        return _pinned_parts(self.kernel)[0]

    @property
    def pinned_code(self) -> str:
        return _pinned_parts(self.kernel)[1]

    @property
    def accumulators(self) -> tuple[str, ...]:
        return _ACCUMULATORS[type(self.kernel)]

    def same(self, part: Callable[[tuple[Node, ...]], object]) -> bool:
        """Return whether one part of the text equals that part of the pinned text."""

        return part(self.top) == part(self.pinned)


def _shared_declarations(nodes: tuple[Node, ...]) -> tuple[str, ...]:
    return tuple(item for item in _statements(nodes) if item.startswith("__shared__"))


def _shared_memory_check(pair: _Pair) -> bool:
    """The two tiles: the pinned declarations, and the size read from the text."""

    size = _tile_bytes(_shared_tiles(pair.code))
    return pair.same(_shared_declarations) and (
        size == pair.kernel.shared_memory_bytes <= MAX_STATIC_SHARED_BYTES
    )


def _index_check(pair: _Pair) -> bool:
    """The top-level index statements: not a tile, a flag or an accumulator."""

    def index(nodes: tuple[Node, ...]) -> tuple[str, ...]:
        return tuple(
            item
            for item in _statements(nodes)
            if not item.startswith(("__shared__", "bool "))
            and not any(_names(item, name) for name in pair.accumulators)
        )

    return pair.same(index)


def _phase_loop_check(pair: _Pair) -> bool:
    """One phase loop with the pinned header, between the setup and the stores."""

    phase, layout = _phase(pair.top), _layout(pair.top)
    return (
        phase is not None
        and phase.head == _phase(pair.pinned).head
        and layout is not None
        and layout[1] == 3
        and pair.top[-2] is phase
    )


def _barrier_check(pair: _Pair) -> bool:
    """The pinned barriers, reached by every thread, with no shared hazard."""

    tiles = tuple(dict.fromkeys(_shared_names(pair.top) + _shared_names(pair.pinned)))
    return (
        pair.same(_skeleton)
        and _uniform_barriers(pair.top, _UNIFORM_NAMES)
        and shared_hazards(pair.top, tiles) == 0
        and pair.code.count("__syncthreads") == pair.pinned_code.count("__syncthreads")
        and _EXIT.search(pair.code) is None
    )


def _store_check(pair: _Pair) -> bool:
    """The pinned store loop last, the only text that touches ``c``, no exit."""

    def stores(nodes: tuple[Node, ...]) -> tuple[str, ...]:
        return tuple(text for text in _texts(nodes) if _C_ELEMENT.search(text))

    return (
        pair.top[-1:] == pair.pinned[-1:]
        and pair.same(stores)
        and _EXIT.search(pair.code) is None
    )


def _accumulator_check(pair: _Pair) -> bool:
    """Each use of an accumulator outside the stores, and the +0.0 start loop."""

    def uses(nodes: tuple[Node, ...]) -> tuple:
        return _accumulator_uses(nodes, pair.accumulators)

    def start(nodes: tuple[Node, ...]) -> tuple[Node, ...]:
        return tuple(node for node in nodes if node.kind == "for")[:1]

    return pair.same(uses) and pair.same(start)


def _coarsened_parts(pair: _Pair) -> dict[str, bool]:
    """The A load per phase, the loop over the B tiles, and its inner loop."""

    phase, pinned_phase = _phase(pair.top), _phase(pair.pinned)
    split = None if phase is None else _single(phase.body, "for(int j=")
    pinned_split = _single(pinned_phase.body, "for(int j=")
    inner = None if split is None else _single(split.body, "for(int i=")
    pinned_inner = _single(pinned_split.body, "for(int i=")
    # The loop over the B tiles sets which accumulators get loads and fmas.
    loads = (
        split is not None
        and phase.body[-1] is split
        and split.head == pinned_split.head
        and _statements(phase.body) == _statements(pinned_phase.body)
        and _statements(split.body) == _statements(pinned_split.body)
    )
    return {
        "coarsened_loads": loads,
        "coarsened_inner_loop": inner is not None and inner.head == pinned_inner.head,
        "coarsened_fma_order": _accumulator_check(pair)
        and inner is not None
        and inner.body == pinned_inner.body,
    }


def _alignment(nodes: tuple[Node, ...]) -> tuple:
    """Return the flags, each float4 load in its branch, and its offset terms.

    A float4 address is 16-byte aligned when the flag holds, the group
    starts at a multiple of 4 and the row length is a multiple of 4. These
    statements fix all three.
    """

    phase = _phase(nodes)
    inside = () if phase is None else _statements(phase.body)
    return (
        tuple(item for item in _statements(nodes) if item.startswith("bool ")),
        _float4_sites(nodes),
        tuple(
            item
            for item in (*_statements(nodes), *inside)
            if item.startswith(_ALIGNMENT_STATEMENTS)
        ),
    )


def _blocked_parts(pair: _Pair) -> dict[str, bool]:
    """The float4 rule, the guarded loads, the inner loop and the micro-tile."""

    phase, pinned_phase = _phase(pair.top), _phase(pair.pinned)
    inner = None if phase is None else _single(phase.body, "for(int i=")
    pinned_inner = _single(pinned_phase.body, "for(int i=")

    def loads(nodes: tuple[Node, ...]) -> tuple | None:
        found = _phase(nodes)
        if found is None:
            return None
        guards = tuple(node for node in found.body if node.kind == "if")
        return _statements(found.body), guards

    return {
        "blocked_vector_alignment": pair.same(_alignment)
        and pair.code.count(_FLOAT4_LOAD) == pair.pinned_code.count(_FLOAT4_LOAD),
        "blocked_loads": phase is not None and pair.same(loads),
        "blocked_inner_loop": inner is not None and inner.head == pinned_inner.head,
        "blocked_fma_order": _accumulator_check(pair)
        and inner is not None
        and inner.body == pinned_inner.body,
    }


def _kernel_checks(pair: _Pair, prefix: str) -> dict[str, bool]:
    checks = {
        f"{prefix}_shared_memory": _shared_memory_check(pair),
        f"{prefix}_index": _index_check(pair),
        f"{prefix}_phase_loop": _phase_loop_check(pair),
        f"{prefix}_barriers": _barrier_check(pair),
        f"{prefix}_store_guard": _store_check(pair),
    }
    if type(pair.kernel) is CoarsenedKernel:
        checks.update(_coarsened_parts(pair))
    else:
        checks.update(_blocked_parts(pair))
    return checks


@dataclass(frozen=True)
class ChecklistCudaCheck:
    """What a compiler-free reading of one checklist kernel text confirmed."""

    kernel: str
    checks: tuple[tuple[str, bool], ...]
    shared_memory_bytes: int | None
    launch: LaunchConfig | None

    @property
    def passed(self) -> bool:
        return all(result for _, result in self.checks)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(name for name, result in self.checks if not result)

    def to_mapping(self) -> dict[str, object]:
        return {
            "kernel": self.kernel,
            "checks": dict(self.checks),
            "passed": self.passed,
            "shared_memory_bytes": self.shared_memory_bytes,
            "launch": None if self.launch is None else self.launch.to_mapping(),
        }


def check_names(kernel: ChecklistKernel) -> tuple[str, ...]:
    """Return the names of the checks of one kernel, in report order."""

    if type(require_kernel(kernel)) is CoarsenedKernel:
        return COARSENED_CHECK_NAMES
    return BLOCKED_CHECK_NAMES


def check_cuda_source(
    text: str, *, kernel: ChecklistKernel, shape: MatmulShape
) -> ChecklistCudaCheck:
    """Check one kernel text and its host grid for one shape.

    Text with code beyond the three definitions, a line splice, a trigraph,
    a space inside an operator or a body that does not parse fails every
    check. Otherwise each check passes or fails on its own.
    """

    checked = require_kernel(kernel)
    case = require_shape(shape)
    names = check_names(checked)
    parts = _read(text, checked)
    if parts is None:
        return ChecklistCudaCheck(
            kernel=checked.name,
            checks=tuple((name, False) for name in names),
            shared_memory_bytes=None,
            launch=None,
        )
    cdiv, body, host, top = parts
    prefix = names[1].removesuffix("_signature")
    checks = _kernel_checks(_Pair(checked, top, body), prefix)
    launch, divisors = _host_launch(host, checked, case)
    checks["cdiv_definition"] = cdiv == _normalize("return (a + b - 1) / b;")
    # _definitions found each signature exactly once.
    checks[f"{prefix}_signature"] = True
    checks[f"{prefix}_grid"] = divisors and launch == checked.launch(case)
    return ChecklistCudaCheck(
        kernel=checked.name,
        checks=tuple((name, bool(checks[name])) for name in names),
        shared_memory_bytes=_tile_bytes(_shared_tiles(body)),
        launch=launch,
    )


def require_cuda_source(
    text: str, *, kernel: ChecklistKernel, shape: MatmulShape
) -> ChecklistCudaCheck:
    """Return the check of ``text``, or raise naming every failed check."""

    result = check_cuda_source(text, kernel=kernel, shape=shape)
    if not result.passed:
        raise GpuChecklistError(
            f"{result.kernel} text fails structural checks: "
            + ", ".join(result.failures)
        )
    return result
