"""Compile-free checks of the Day 78 CUDA reduction texts.

No compiler exists on this host, so the checker reads a text the way the Day
76 and 77 checkers do: comments as a compiler reads them, no line splice,
trigraph or space inside an operator, and only the four pinned definitions:
``cdiv``, the pass-1 kernel, the combine kernel and the launcher. It parses
each kernel body into the Day 77 tree of statements, ``for`` loops and ``if``
blocks, and it parses the pinned text with the same parser.

Each statement and each loop or guard header belongs to one part of a
kernel, and each part has one check. A check compares its part of the two
trees, with the loops and guards above each item. An item names each of
those by its part and its place, so a changed loop header moves no item of
another part. Thus a changed or added statement fails the check of its
part. The ``order`` check compares the order of the parts, so it fails
when a statement moves before or after a statement of another part.

Each check also applies a rule that does not use the pinned text:

* shared memory: one ``__shared__ float`` array, one float for each thread
  of the block, inside the 48 KB static limit of compute capability 8.0;
* loads: each read of ``x`` is ``i < n ? x[i] : 0.0f`` for its own index, and
  the loads of one thread are one block apart, so the grid loads each
  element once;
* tree: the tree loop, run with the block of the launch, gives the declared
  strides;
* barriers: no barrier is inside an ``if``, only tree loops hold a barrier,
  and their headers use only ``blockDim.x``, constants and their own
  variable. No thread leaves the kernel early, and each thread reaches the
  declared count of barriers;
* warp stage: each shuffle is ``__shfl_down_sync(0xffffffff, w, d)`` under
  the one guard ``warp == 0u``, which selects the 32 threads of warp 0, so
  each lane that the mask names calls each shuffle. The offsets give lane 0
  each of the 32 values once, and the stage holds no barrier;
* stores: one guarded store, by thread 0 of the block, at the end of the
  kernel;
* grid-stride loop: for the partials of the size, the two loops of the
  combine, run for the 1024 threads of its launch, read each partial once and
  read nothing at or past the count;
* launch: the host ``dim3`` expressions, with C unsigned division, give the
  launch of the kernel model. The combine runs as one block of 1024 threads,
  and its count is ``grid.x``, the number of partials.

The checker proves nothing about ``nvcc``: a text that passes can still fail
to compile, and only a later NVIDIA run can show that it does not.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy

from .gpu_checklist_check import BARRIER, Node, parse_body
from .gpu_reduce_cuda import (
    COMBINE_BARRIERS,
    COMBINE_KERNEL,
    COMBINE_LAUNCH,
    COMBINE_SHARED_BYTES,
    COMBINE_SIGNATURE,
    COMBINE_STRIDES,
    COMBINE_THREADS,
    MAX_STATIC_SHARED_BYTES,
    CudaReduceKernel,
    GpuReduceCudaError,
    cuda_source,
    launcher_signature,
    pass_one_signature,
    require_kernel,
    require_n,
)
from .gpu_tiled_cuda import FLOAT_BYTES, _comparable_code, _definitions
from .kernel_launch_config import WARP_SIZE, Dim3, KernelLaunchError, LaunchConfig
from .kernel_launch_cuda import _CDIV, _DECIMAL, _cdiv, _normalize

# The sizes of config/day-78-gpu-reduce.json: each correctness size once, in
# the order of the cases, and then the timing sizes. A test ties them to it.
CORRECTNESS_SIZES = (1, 31, 33, 257, 1000003, 2000003, 511, 513, 300007)
BENCHMARK_SIZES = (65536, 1048576, 16777216, 67108864, 10000019)
CHECK_SIZES = (*CORRECTNESS_SIZES, *BENCHMARK_SIZES)
COMBINE_CHECK_NAMES = (
    "combine_grid_stride",
    "combine_tree",
    "combine_barriers",
    "combine_store",
)
_X_READ = re.compile(r"(?<![A-Za-z0-9_])x\[")
_GUARDED_READ = re.compile(
    r"(?<![A-Za-z0-9_])(i(?:\+([1-9][0-9]{0,9})u)?)<n\?x\[\1\]:0\.0f"
)
_LOAD_INDEX = re.compile(rf"unsigned int i=blockIdx\.x\*{_DECIMAL}u\+t;")
_SHARED_ARRAY = re.compile(rf"__shared__ float shared\[{_DECIMAL}\];")
_TREE = re.compile(r"for\(unsigned int s=blockDim\.x/2u;s(>=|>)([0-9]{1,10})u;s>>=1u\)")
_SHUFFLE = re.compile(
    r"w=__fadd_rn\(w,__shfl_down_sync\(0xffffffffu,w,([0-9]{1,10})u\)\);"
)
_LANE = re.compile(r"unsigned int lane=t%([0-9]{1,10})u;")
_WARP = re.compile(r"unsigned int warp=t/([0-9]{1,10})u;")
_GUARD = re.compile(r"if\((t|lane|warp)(==|<)([0-9]{1,10})u\)")
_UNROLLED = re.compile(r"for\(;e\+([0-9]{1,10})u<count;e\+=([0-9]{1,10})u\)")
_TAIL = re.compile(r"for\(;e<count;e\+=([0-9]{1,10})u\)")
_ACCUMULATE = re.compile(
    r"a([0-3])=__fadd_rn\(a\1,partials\[e(?:\+([0-9]{1,10})u)?\]\);"
)
_PARTIAL_STORE = re.compile(r"partials\[blockIdx\.x\]=(?:shared\[0\]|w);")
_OUT_STORE = "out[0]=shared[0];"
_EXIT = re.compile(
    r"(?<![A-Za-z0-9_])(return|break|continue|goto|while|do|switch)(?![A-Za-z0-9_])"
)
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])[0-9][0-9A-Za-z]*")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A tree loop reads only values that every thread of a block shares.
_UNIFORM = frozenset({"unsigned", "int", "s", "blockDim", "x"})
_LAUNCHER = re.compile(
    rf"dim3 block\({_DECIMAL}\);dim3 grid\(cdiv\(n,{_DECIMAL}\)\);"
    r"([A-Za-z_][A-Za-z0-9_]*)<<<grid,block>>>\(partials,x,n\);"
    rf"{COMBINE_KERNEL}<<<{_DECIMAL},{_DECIMAL}>>>\(out,partials,grid\.x\);"
)
Anchor = tuple[str, int]


@dataclass(frozen=True)
class Item:
    """One statement, or one loop or guard header, of a kernel body.

    ``anchor`` is the part of the item and its place among the siblings of
    that part. ``path`` holds the anchor of each loop and guard above the
    item, and ``guards`` holds their header texts. ``neighbors`` holds the
    anchors of the siblings before and after the item.
    """

    part: str
    anchor: Anchor
    path: tuple[Anchor, ...]
    guards: tuple[str, ...]
    neighbors: tuple[Anchor | None, Anchor | None]
    last: bool
    node: Node

    @property
    def text(self) -> str:
        return self.node.head


def _subtree(node: Node) -> Iterator[str]:
    yield node.head
    for child in node.body + (node.orelse or ()):
        yield from _subtree(child)


def _items(
    nodes: tuple[Node, ...],
    part_of: Callable[[Node, str | None], str],
    path: tuple[Anchor, ...] = (),
    guards: tuple[str, ...] = (),
    parent: str | None = None,
) -> list[Item]:
    """Return every item of a body, depth first, in program order."""

    parts = [part_of(node, parent) for node in nodes]
    seen: Counter[str] = Counter()
    anchors = []
    for part in parts:
        anchors.append((part, seen[part]))
        seen[part] += 1
    items = []
    for index, (node, part, anchor) in enumerate(zip(nodes, parts, anchors)):
        before = anchors[index - 1] if index else None
        after = anchors[index + 1] if index + 1 < len(nodes) else None
        last = index + 1 == len(nodes)
        items.append(Item(part, anchor, path, guards, (before, after), last, node))
        inner = ((*path, anchor), (*guards, node.head), part)
        items.extend(_items(node.body, part_of, *inner))
        if node.orelse is not None:
            otherwise = ((*path, (f"{part}:else", anchor[1])), (*guards, "else"), part)
            items.extend(_items(node.orelse, part_of, *otherwise))
    return items


def _pass_one_part(kernel: CudaReduceKernel) -> Callable[[Node, str | None], str]:
    """Return the part of each pass-1 item: by its text, else by its parent."""

    prefix = kernel.prefix

    def part_of(node: Node, parent: str | None) -> str:
        head, texts = node.head, tuple(_subtree(node))
        if head == BARRIER:
            return f"{prefix}_barriers"
        if head.startswith("__shared__"):
            return f"{prefix}_shared_memory"
        if kernel.warp_shuffle and any("__shfl" in text for text in texts):
            return f"{prefix}_warp_stage"
        if any("partials[" in text for text in texts):
            return f"{prefix}_partial_store"
        if parent == f"{prefix}_warp_stage":
            return parent
        if _X_READ.search(head) or head.startswith("unsigned int i="):
            return f"{prefix}_loads"
        if head == "shared[t]=v;":
            return f"{prefix}_loads"
        if head.startswith("for(unsigned int s="):
            return f"{prefix}_tree"
        return f"{prefix}_index" if parent is None else parent

    return part_of


def _combine_part(node: Node, parent: str | None) -> str:
    """Return the part of each combine item: by its text, else by its parent."""

    head = node.head
    if head == BARRIER:
        return "combine_barriers"
    if any("out[" in text for text in _subtree(node)):
        return "combine_store"
    if "partials[" in head or head.startswith(
        ("for(;", "float a0=", "unsigned int e=", "unsigned int t=")
    ):
        return "combine_grid_stride"
    return "combine_tree" if parent is None else parent


def check_names(kernel: CudaReduceKernel) -> tuple[str, ...]:
    """Return the names of the checks of one text, in report order."""

    checked = require_kernel(kernel)
    parts = ("shared_memory", "index", "loads", "tree", "barriers")
    parts += ("warp_stage",) if checked.warp_shuffle else ()
    return (
        "cdiv_definition",
        *(f"{checked.prefix}_{part}" for part in (*parts, "partial_store")),
        *COMBINE_CHECK_NAMES,
        "order",
        "launch",
    )


@dataclass(frozen=True)
class _Kernel:
    """One kernel body of a text: its code, its tree and its items."""

    code: str
    nodes: tuple[Node, ...]
    items: tuple[Item, ...]

    def part(self, name: str) -> tuple[object, ...]:
        """Return the items of one part with the anchors of the loops above them."""

        if name.endswith("_barriers"):
            return tuple(
                (item.path, item.neighbors) for item in self.items if item.part == name
            )
        return tuple(
            (item.path, item.node.kind, item.text)
            for item in self.items
            if item.part == name
        )

    def order(self) -> tuple[tuple[tuple[Anchor, ...], str], ...]:
        """Return the parts in program order, each run of one part once.

        Barriers are left out: their own check compares where they are.
        """

        sequence: list[tuple[tuple[Anchor, ...], str]] = []
        for item in self.items:
            key = (item.path, item.part)
            if not item.part.endswith("_barriers") and sequence[-1:] != [key]:
                sequence.append(key)
        return tuple(sequence)


@dataclass(frozen=True)
class _Text:
    cdiv: str
    pass_one: _Kernel
    combine: _Kernel
    host: str


def _read(text: str, kernel: CudaReduceKernel) -> _Text | None:
    """Return the four definitions of a text and the trees of its kernels."""

    if type(text) is not str:
        raise TypeError("CUDA source must be a string")
    code = _comparable_code(text)
    signatures = (
        _CDIV,
        pass_one_signature(kernel),
        COMBINE_SIGNATURE,
        launcher_signature(kernel),
    )
    bodies = None if code is None else _definitions(_normalize(code), signatures)
    if bodies is None:
        return None
    cdiv, pass_one, combine, host = bodies
    kernels = []
    for body, part_of in ((pass_one, _pass_one_part(kernel)), (combine, _combine_part)):
        nodes = parse_body(body)
        if nodes is None:
            return None
        kernels.append(_Kernel(body, nodes, tuple(_items(nodes, part_of))))
    return _Text(cdiv, *kernels, host)


@functools.cache
def _pinned(kernel: CudaReduceKernel) -> _Text:
    parts = _read(cuda_source(kernel), kernel)
    if parts is None:
        raise GpuReduceCudaError(f"the pinned text of {kernel.name} does not parse")
    return parts


def tree_trip(header: str, threads: int) -> tuple[int, ...] | None:
    """Return the strides of a tree loop header for a block, or ``None``.

    ``None`` means that the header is not a halving tree loop, or that the
    loop never ends, such as ``s >= 0u`` on an unsigned stride.
    """

    found = _TREE.fullmatch(header)
    if found is None:
        return None
    strict, bound = found.group(1) == ">", int(found.group(2))
    strides: list[int] = []
    stride = threads // 2
    while stride > bound if strict else stride >= bound:
        if stride == 0:
            return None
        strides.append(stride)
        stride >>= 1
    return tuple(strides)


def selected_threads(guards: tuple[str, ...], threads: int) -> tuple[int, ...] | None:
    """Return the threads of a block that pass every guard, or ``None``.

    A guard compares ``t``, ``lane = t % 32`` or ``warp = t / 32`` with a
    constant by ``==`` or ``<``; any other guard gives ``None``.
    """

    thread = numpy.arange(threads)
    values = {"t": thread, "lane": thread % WARP_SIZE, "warp": thread // WARP_SIZE}
    mask = numpy.ones(threads, dtype=bool)
    for guard in guards:
        found = _GUARD.fullmatch(guard)
        if found is None:
            return None
        name, operator, bound = found.group(1), found.group(2), int(found.group(3))
        value = values[name]
        mask &= value == bound if operator == "==" else value < bound
    return tuple(int(index) for index in numpy.flatnonzero(mask))


def shuffle_coverage(offsets: tuple[int, ...]) -> tuple[int, ...]:
    """Return how often the sum of lane 0 holds the value of each lane.

    ``__shfl_down_sync`` gives lane l the value of lane l + d. When l + d is
    32 or more, lane l gets its own value back: the CUDA C++ Programming Guide
    states that the upper d lanes stay unchanged.
    """

    lanes = numpy.arange(WARP_SIZE)
    counts = numpy.eye(WARP_SIZE, dtype=numpy.int64)
    for offset in offsets:
        source = numpy.where(lanes + offset < WARP_SIZE, lanes + offset, lanes)
        counts = counts + counts[source]
    return tuple(int(count) for count in counts[0])


@dataclass(frozen=True)
class GridStrideLoops:
    """The two loops of the combine, as a text writes them.

    ``offsets`` holds the accumulator and the offset of each read in the
    unrolled loop, and ``tail_offsets`` those of the tail loop.
    """

    guard: int
    step: int
    offsets: tuple[tuple[int, int], ...]
    tail_step: int
    tail_offsets: tuple[tuple[int, int], ...]


def _reads(body: tuple[Node, ...]) -> tuple[tuple[int, int], ...] | None:
    found = [
        _ACCUMULATE.fullmatch(node.head) if node.kind == "statement" else None
        for node in body
    ]
    if not found or any(match is None for match in found):
        return None
    return tuple((int(match.group(1)), int(match.group(2) or 0)) for match in found)


def grid_stride_loops(nodes: tuple[Node, ...]) -> GridStrideLoops | None:
    """Return the two grid-stride loops of a combine body, or ``None``."""

    loops = [node for node in nodes if node.kind == "for" and node.head[:5] == "for(;"]
    if len(loops) != 2:
        return None
    unrolled, tail = _UNROLLED.fullmatch(loops[0].head), _TAIL.fullmatch(loops[1].head)
    offsets, tail_offsets = _reads(loops[0].body), _reads(loops[1].body)
    if None in (unrolled, tail, offsets, tail_offsets):
        return None
    return GridStrideLoops(
        guard=int(unrolled.group(1)),
        step=int(unrolled.group(2)),
        offsets=offsets,
        tail_step=int(tail.group(1)),
        tail_offsets=tail_offsets,
    )


def combine_coverage(
    loops: GridStrideLoops, count: int, threads: int = COMBINE_THREADS
) -> numpy.ndarray | None:
    """Return how often the loops read each partial, or ``None`` for a bad read.

    Thread t starts at ``e = t``. A read at or past ``count`` gives ``None``,
    and so does a step of 0, because such a loop never ends.
    """

    if loops.step == 0 or loops.tail_step == 0:
        return None
    index = numpy.arange(threads, dtype=numpy.int64)
    reads = []
    for guard, step, offsets in (
        (loops.guard, loops.step, loops.offsets),
        (0, loops.tail_step, loops.tail_offsets),
    ):
        while (active := index + guard < count).any():
            reads.extend(index[active] + offset for _, offset in offsets)
            index[active] += step
    every = numpy.concatenate(reads) if reads else numpy.zeros(0, dtype=numpy.int64)
    if (every >= count).any():
        return None
    return numpy.bincount(every, minlength=count)


def _shared_floats(code: str) -> int | None:
    """Return the floats of the one ``__shared__`` array, or ``None``."""

    found = _SHARED_ARRAY.findall(code)
    if len(found) != 1 or code.count("__shared__") != 1:
        return None
    return int(found[0])


def _has_barrier(node: Node) -> bool:
    return any("__syncthreads" in text for text in _subtree(node))


def barriers_reached(nodes: tuple[Node, ...], threads: int) -> int | None:
    """Return the barriers that every thread of a block reaches, or ``None``.

    ``None`` means that some thread can skip a barrier: a barrier inside an
    ``if``, or a loop around a barrier that is not a halving tree loop over
    values that every thread shares.
    """

    count = 0
    for node in nodes:
        if node.kind == "statement":
            count += node.head == BARRIER
            continue
        if not _has_barrier(node):
            continue
        strides = tree_trip(node.head, threads) if node.kind == "for" else None
        names = set(_IDENTIFIER.findall(_NUMBER.sub(" ", node.head[len("for") :])))
        inner = None if strides is None else barriers_reached(node.body, threads)
        if inner is None or not names <= _UNIFORM:
            return None
        count += len(strides) * inner
    return count


def _barrier_rule(kernel: _Kernel, threads: int, expected: int) -> bool:
    statements = sum(item.text == BARRIER for item in kernel.items)
    return (
        barriers_reached(kernel.nodes, threads) == expected
        and kernel.code.count("__syncthreads") == statements
        and _EXIT.search(kernel.code) is None
    )


def _tree_rule(kernel: _Kernel, part: str, threads: int, strides: tuple) -> bool:
    loops = [
        item
        for item in kernel.items
        if item.part == part and not item.path and item.node.kind == "for"
    ]
    return (
        len(loops) == 1
        and loops[0].node.kind == "for"
        and tree_trip(loops[0].text, threads) == strides
    )


def _shared_rule(code: str, floats: int, limit: int) -> bool:
    found = _shared_floats(code)
    return found == floats and found * FLOAT_BYTES <= limit


def _last_in_kernel(kernel: _Kernel, item: Item) -> bool:
    """Return whether the item and each guard above it end their blocks."""

    above = {(item.path[:depth], item.path[depth]) for depth in range(len(item.path))}
    return item.last and all(
        other.last for other in kernel.items if (other.path, other.anchor) in above
    )


def _store_rule(kernel: _Kernel, marker: str, pattern: Callable, threads: int) -> bool:
    """One store that only thread 0 runs, as the last statement of the kernel."""

    stores = [item for item in kernel.items if marker in item.text]
    if len(stores) != 1 or kernel.code.count(marker) != 1:
        return False
    store = stores[0]
    return (
        pattern(store.text)
        and selected_threads(store.guards, threads) == (0,)
        and _last_in_kernel(kernel, store)
    )


def _loads_rule(kernel: CudaReduceKernel, code: str) -> bool:
    """Each read of x has its own guard, and a block loads its span once.

    Thread t of block b loads ``b * span + t + offset`` for each offset. With
    offsets 0, B, 2 B and so on for B threads, and a span of B times the
    loads, the loads of the grid are the elements ``0`` to ``grid * span - 1``,
    each once.
    """

    guarded = _GUARDED_READ.findall(code)
    spans = _LOAD_INDEX.findall(code)
    if not guarded or len(guarded) != len(_X_READ.findall(code)) or len(spans) != 1:
        return False
    offsets = tuple(int(offset or 0) for _, offset in guarded)
    threads, span = kernel.block_threads, int(spans[0])
    return span == threads * len(
        offsets
    ) == kernel.elements_per_block and offsets == tuple(range(0, span, threads))


def _warp_rule(kernel: _Kernel, threads: int) -> bool:
    """Five full-mask shuffles in warp 0 give lane 0 every value of the warp once."""

    shuffles = [item for item in kernel.items if "__shfl" in item.text]
    found = [_SHUFFLE.fullmatch(item.text) for item in shuffles]
    if not shuffles or None in found or kernel.code.count("__shfl") != len(found):
        return False
    stage = [
        item
        for item in kernel.items
        if item.part.endswith("_warp_stage") or "if(warp==0u)" in item.guards
    ]
    top = [item.text for item in kernel.items if not item.path]
    offsets = tuple(int(match.group(1)) for match in found)
    return (
        {item.guards for item in shuffles} == {("if(warp==0u)",)}
        and _LANE.findall(kernel.code) == [str(WARP_SIZE)]
        and _WARP.findall(kernel.code) == [str(WARP_SIZE)]
        and "unsigned int t=threadIdx.x;" in top
        and selected_threads(("if(warp==0u)",), threads) == tuple(range(WARP_SIZE))
        and shuffle_coverage(offsets) == (1,) * WARP_SIZE
        and not any("__syncthreads" in item.text for item in stage)
    )


def _pass_one_checks(checked: CudaReduceKernel, text: _Text) -> dict[str, bool]:
    kernel, pinned = text.pass_one, _pinned(checked).pass_one
    prefix, threads = checked.prefix, checked.block_threads

    def same(part: str) -> bool:
        return kernel.part(f"{prefix}_{part}") == pinned.part(f"{prefix}_{part}")

    checks = {
        f"{prefix}_shared_memory": same("shared_memory")
        and _shared_rule(kernel.code, threads, MAX_STATIC_SHARED_BYTES),
        f"{prefix}_index": same("index"),
        f"{prefix}_loads": same("loads") and _loads_rule(checked, kernel.code),
        f"{prefix}_tree": same("tree")
        and _tree_rule(kernel, f"{prefix}_tree", threads, checked.tree_strides),
        f"{prefix}_barriers": same("barriers")
        and _barrier_rule(kernel, threads, checked.barriers_per_block),
        f"{prefix}_partial_store": same("partial_store")
        and _store_rule(kernel, "partials[", _PARTIAL_STORE.fullmatch, threads),
    }
    if checked.warp_shuffle:
        checks[f"{prefix}_warp_stage"] = same("warp_stage") and _warp_rule(
            kernel, threads
        )
    return checks


def _combine_checks(text: _Text, kernel: CudaReduceKernel, n: int) -> dict[str, bool]:
    combine, pinned = text.combine, _pinned(kernel).combine
    loops = grid_stride_loops(combine.nodes)
    coverage = (
        None if loops is None else combine_coverage(loops, kernel.partial_count(n))
    )

    def same(name: str) -> bool:
        return combine.part(name) == pinned.part(name)

    return {
        "combine_grid_stride": same("combine_grid_stride")
        and coverage is not None
        and bool((coverage == 1).all())
        and loops.guard >= max(offset for _, offset in loops.offsets),
        "combine_tree": same("combine_tree")
        and _shared_rule(combine.code, COMBINE_THREADS, COMBINE_SHARED_BYTES)
        and _tree_rule(combine, "combine_tree", COMBINE_THREADS, COMBINE_STRIDES),
        "combine_barriers": same("combine_barriers")
        and _barrier_rule(combine, COMBINE_THREADS, COMBINE_BARRIERS),
        "combine_store": same("combine_store")
        and _store_rule(
            combine, "out[", lambda store: store == _OUT_STORE, COMBINE_THREADS
        ),
    }


def _host_launch(
    host: str, kernel: CudaReduceKernel, n: int
) -> tuple[LaunchConfig | None, LaunchConfig | None, bool]:
    """Evaluate a launcher that is one block, one grid and two launches.

    The grid divisor must be the elements of one block, so a divisor that
    agrees with the model for one n only still fails, and the pass-1 block
    must hold whole warps.
    """

    found = _LAUNCHER.fullmatch(host)
    if found is None or found.group(3) != kernel.kernel_name:
        return None, None, False
    block, divisor, combine_grid, combine_block = (
        int(found.group(index)) for index in (1, 2, 4, 5)
    )
    try:
        launch = LaunchConfig(Dim3(_cdiv(n, divisor)), Dim3(block))
        combine = LaunchConfig(Dim3(combine_grid), Dim3(combine_block))
    except (KernelLaunchError, TypeError):
        return None, None, False
    return (
        launch,
        combine,
        divisor == kernel.elements_per_block and block % WARP_SIZE == 0,
    )


@dataclass(frozen=True)
class ReduceCudaCheck:
    """What a compiler-free reading of one reduction text confirmed for one n."""

    kernel: str
    n: int
    checks: tuple[tuple[str, bool], ...]
    launch: LaunchConfig | None
    combine_launch: LaunchConfig | None
    shared_memory_bytes: int | None
    barriers_per_block: int | None

    @property
    def passed(self) -> bool:
        return all(result for _, result in self.checks)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(name for name, result in self.checks if not result)

    def to_mapping(self) -> dict[str, object]:
        return {
            "kernel": self.kernel,
            "n": self.n,
            "checks": dict(self.checks),
            "passed": self.passed,
            "launch": None if self.launch is None else self.launch.to_mapping(),
            "combine_launch": None
            if self.combine_launch is None
            else self.combine_launch.to_mapping(),
            "shared_memory_bytes": self.shared_memory_bytes,
            "barriers_per_block": self.barriers_per_block,
        }


def check_reduce_source(
    text: str, *, kernel: CudaReduceKernel, n: int
) -> ReduceCudaCheck:
    """Check one reduction text and its host launch for one input length.

    Text with code beyond the four definitions, a line splice, a trigraph, a
    space inside an operator or a body that does not parse fails every
    check. Otherwise each check passes or fails on its own.
    """

    checked = require_kernel(kernel)
    count = require_n(n)
    names = check_names(checked)
    text_parts = _read(text, checked)
    if text_parts is None:
        return ReduceCudaCheck(
            kernel=checked.name,
            n=count,
            checks=tuple((name, False) for name in names),
            launch=None,
            combine_launch=None,
            shared_memory_bytes=None,
            barriers_per_block=None,
        )
    pinned = _pinned(checked)
    checks = {
        "cdiv_definition": text_parts.cdiv == _normalize("return (a + b - 1) / b;")
    }
    checks.update(_pass_one_checks(checked, text_parts))
    checks.update(_combine_checks(text_parts, checked, count))
    checks["order"] = (text_parts.pass_one.order(), text_parts.combine.order()) == (
        pinned.pass_one.order(),
        pinned.combine.order(),
    )
    launch, combine, rule = _host_launch(text_parts.host, checked, count)
    checks["launch"] = (
        rule and launch == checked.launch(count) and combine == COMBINE_LAUNCH
    )
    floats = _shared_floats(text_parts.pass_one.code)
    return ReduceCudaCheck(
        kernel=checked.name,
        n=count,
        checks=tuple((name, bool(checks[name])) for name in names),
        launch=launch,
        combine_launch=combine,
        shared_memory_bytes=None if floats is None else floats * FLOAT_BYTES,
        barriers_per_block=barriers_reached(
            text_parts.pass_one.nodes, checked.block_threads
        ),
    )


def require_reduce_source(
    text: str, *, kernel: CudaReduceKernel, n: int
) -> ReduceCudaCheck:
    """Return the check of ``text``, or raise naming every failed check."""

    result = check_reduce_source(text, kernel=kernel, n=n)
    if not result.passed:
        raise GpuReduceCudaError(
            f"{result.kernel} text fails structural checks: "
            + ", ".join(result.failures)
        )
    return result
