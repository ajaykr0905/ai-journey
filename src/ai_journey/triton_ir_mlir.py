"""Read the TTIR and TTGIR texts that Triton wrote, with no triton import.

Triton lowers a ``@triton.jit`` function in steps. The Python AST becomes
TTIR, an MLIR module of ``tt.*`` ops on whole tiles with no GPU layout: one
``tt.load`` for each ``tl.load``, one ``tt.dot`` for each ``tl.dot``, and
``scf.for`` for a Python ``for`` loop. TTGIR is the same program after Triton
chose a GPU layout: each tensor type names an encoding such as ``#blocked``
(which thread holds which elements), ``#shared`` (a buffer in shared memory)
or ``#mma`` (the operands of a Tensor Core instruction), and the module names
``ttg.num-warps``, ``ttg.threads-per-warp`` and the target. The pipeliner
turns the loads of a loop into ``ttg.async_copy_global_to_local`` copies
into a ring of shared buffers and waits for them with ``ttg.async_wait``.

``read_mlir`` reads one module: its attributes, its encoding aliases, its ops
and its ``#loc`` table. Each op gets the line of the kernel source that its
location names. A ``callsite`` location of an inlined helper names the call
site, so the op gets the line of the call. ``ttir_summary`` and
``ttgir_summary`` count what a learner reads first.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .triton_ir_manifest import TritonIrError
from .triton_ir_ptx import SOURCE_PREFIX

_LOC_DEF = re.compile(r"^(#loc\d*) = loc\((.*)\)$")
_ALIAS = re.compile(r"^#(\w+) = #([\w.]+)(<.*>)?$")
_OP = re.compile(r"^(?:(%[^=]+?) = )?([a-z_]+(?:\.[a-z_0-9]+)+)(?=[ ({]|$)(.*)$")
_TRAILING = re.compile(r"\s+loc\((#loc\d*)\)$")
_CLOSE = re.compile(r"^\}\s*loc\((#loc\d*)\)$")
_FILE_LOC = re.compile(r'"([^"]*)":(\d+):(\d+)')
_NAMED = re.compile(r'"[^"]*"\((.*)\)')
_CALLSITE = re.compile(r"callsite\((#loc\d*) at (#loc\d*)\)")
_ATTRIBUTE = re.compile(r'^"?([\w.-]+)"?\s*=\s*(.+?)(?:\s*:\s*i\d+)?$')
_MODULE = re.compile(r"^module(?: attributes \{(.*)\})? \{$")
_FUNCTION = re.compile(r"^tt\.func (?:public |private )?@(\w+)\(")
_INLINE_ENCODING = re.compile(r"#ttg\.(\w+)<")
_RANGE = re.compile(r"\{end = (-?\d+) : i32, start = (-?\d+) : i32\}")
_ALLOC = re.compile(r"!ttg\.memdesc<([0-9x]+)x(f32|f16|bf16|i\d+)")
_WAIT = re.compile(r"\{num = (\d+) : i32\}")
_DOT_PARENT = re.compile(r"#ttg\.dot_op<\{opIdx = (\d), parent = (#\w+)\}>")


@dataclass(frozen=True)
class MlirOp:
    """One op of the kernel function: its MLIR line, name and source line."""

    line: int
    name: str
    body: str
    loc: str | None
    source_line: int | None


@dataclass(frozen=True)
class MlirModule:
    """One TTIR or TTGIR module: attributes, encoding aliases and ops."""

    function: str
    attributes: tuple[tuple[str, str], ...]
    aliases: tuple[tuple[str, str, str], ...]
    ops: tuple[MlirOp, ...]
    locations: int

    def count(self, name: str) -> int:
        return sum(op.name == name for op in self.ops)

    @property
    def histogram(self) -> dict[str, int]:
        counts = Counter(op.name for op in self.ops)
        return dict(sorted(counts.items()))


def _split_top(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "<{([":
            depth += 1
        elif char in ">})]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _attributes(text: str | None) -> tuple[tuple[str, str], ...]:
    if text is None:
        return ()
    found = []
    for part in _split_top(text):
        match = _ATTRIBUTE.match(part)
        if not match:
            raise TritonIrError(f"module attribute {part!r} is not name = value")
        found.append((match.group(1), match.group(2).strip('"')))
    return tuple(found)


def _resolve(
    name: str, table: dict[str, str], seen: frozenset[str] = frozenset()
) -> tuple[str, int] | None:
    """Return (file, line) of a location alias, following names and calls."""

    if name in seen:
        raise TritonIrError(f"location {name} refers to itself")
    if name not in table:
        raise TritonIrError(f"location {name} is not defined")
    return _resolve_expression(table[name], table, seen | {name})


def _resolve_expression(
    expression: str, table: dict[str, str], seen: frozenset[str]
) -> tuple[str, int] | None:
    if expression == "unknown":
        return None
    if expression.startswith("#loc"):
        return _resolve(expression, table, seen)
    match = _FILE_LOC.fullmatch(expression)
    if match:
        return match.group(1), int(match.group(2))
    match = _NAMED.fullmatch(expression)
    if match:
        return _resolve_expression(match.group(1), table, seen)
    match = _CALLSITE.fullmatch(expression)
    if match:
        # The caller is the second location: the line that called the helper.
        return _resolve(match.group(2), table, seen)
    raise TritonIrError(f"location {expression!r} has an unknown form")


def _source_line(loc: str | None, table: dict[str, str]) -> int | None:
    if loc is None:
        return None
    place = _resolve(loc, table)
    if place is None or not place[0].startswith(SOURCE_PREFIX):
        return None
    return place[1]


def read_mlir(text: str) -> MlirModule:
    """Parse one TTIR or TTGIR module with one ``tt.func``."""

    if type(text) is not str or not text.isascii():
        raise TritonIrError("an MLIR text must be an ASCII string")
    lines = text.splitlines()
    table: dict[str, str] = {}
    aliases: list[tuple[str, str, str]] = []
    modules = []
    for number, line in enumerate(lines, start=1):
        location = _LOC_DEF.match(line)
        if location:
            if location.group(1) in table:
                raise TritonIrError(f"location {location.group(1)} is defined twice")
            table[location.group(1)] = location.group(2)
            continue
        alias = _ALIAS.match(line)
        if alias:
            aliases.append((alias.group(1), alias.group(2), alias.group(3) or ""))
            continue
        module = _MODULE.match(line)
        if module:
            modules.append((number, module.group(1)))
    if len(modules) != 1:
        raise TritonIrError("an MLIR text must hold exactly one module")
    start = modules[0][0]
    # The module region is the bottom of the stack and must close last.
    pending: list[tuple[int, str, str]] = [(start, "module", "")]
    ops: list[tuple[int, str, str, str | None]] = []
    functions = []
    for number in range(start + 1, len(lines) + 1):
        line = lines[number - 1].strip()
        if not line or line.startswith("#"):
            continue
        if not pending:
            raise TritonIrError(f"MLIR line {number} is outside the module")
        close = _CLOSE.match(line)
        if close:
            opened = pending.pop()
            if pending:
                ops.append((*opened, close.group(1)))
            continue
        if line.startswith("}"):
            raise TritonIrError(f"MLIR line {number} closes a region with no loc")
        match = _OP.match(line)
        if not match:
            raise TritonIrError(f"MLIR line {number} is not an op")
        name, body = match.group(2), match.group(3)
        function = _FUNCTION.match(line)
        if function:
            functions.append(function.group(1))
        if line.endswith("{"):
            pending.append((number, name, body))
            continue
        trailing = _TRAILING.search(body)
        if not trailing:
            raise TritonIrError(f"MLIR line {number} has no trailing loc")
        ops.append((number, name, body[: trailing.start()], trailing.group(1)))
    if pending:
        raise TritonIrError("an MLIR region never closes")
    if len(functions) != 1:
        raise TritonIrError("an MLIR module must hold exactly one tt.func")
    return MlirModule(
        function=functions[0],
        attributes=_attributes(modules[0][1]),
        aliases=tuple(aliases),
        ops=tuple(
            MlirOp(number, name, body, loc, _source_line(loc, table))
            for number, name, body, loc in sorted(ops)
        ),
        locations=len(table),
    )


def _operands(op: MlirOp) -> list[str]:
    return [part for part in op.body.split(" : ", 1)[0].split(",") if part.strip()]


def common_summary(module: MlirModule) -> dict[str, object]:
    loads = [op for op in module.ops if op.name == "tt.load"]
    stores = [op for op in module.ops if op.name == "tt.store"]
    ranges = sorted(
        {
            (int(match.group(2)), int(match.group(1)))
            for op in module.ops
            if op.name == "tt.make_range"
            for match in [_RANGE.search(op.body)]
            if match
        }
    )
    return {
        "function": module.function,
        "ops": sum(module.histogram.values()),
        "histogram": module.histogram,
        "locations": module.locations,
        "ops_with_source_line": sum(op.source_line is not None for op in module.ops),
        "tt_load": len(loads),
        "tt_load_masked": sum(len(_operands(op)) >= 2 for op in loads),
        "tt_load_with_other": sum(len(_operands(op)) >= 3 for op in loads),
        "tt_store": len(stores),
        "tt_store_masked": sum(len(_operands(op)) >= 3 for op in stores),
        "tt_dot": module.count("tt.dot"),
        "tt_make_range": [{"start": start, "end": end} for start, end in ranges],
        "tt_splat": module.count("tt.splat"),
        "scf_for": module.count("scf.for"),
    }


def ttir_summary(module: MlirModule) -> dict[str, object]:
    """Count the ``tt.*`` ops of a TTIR module, which has no GPU layout."""

    if module.aliases or any(_INLINE_ENCODING.search(op.body) for op in module.ops):
        raise TritonIrError("a TTIR module must hold no GPU layout encoding")
    return common_summary(module)


def ttgir_summary(module: MlirModule) -> dict[str, object]:
    """Count the layouts, async copies and module attributes of a TTGIR module."""

    attributes = dict(module.attributes)
    for name in ("ttg.num-warps", "ttg.threads-per-warp", "ttg.target"):
        if name not in attributes:
            raise TritonIrError(f"a TTGIR module must name {name}")
    kinds = {name: kind for name, kind, _ in module.aliases}
    inline = Counter(
        match for op in module.ops for match in _INLINE_ENCODING.findall(op.body)
    )
    parents = sorted(
        {
            (int(index), parent)
            for op in module.ops
            if op.name == "tt.dot"
            for index, parent in _DOT_PARENT.findall(op.body)
        }
    )
    allocations = [
        match.group(1)
        for op in module.ops
        if op.name == "ttg.local_alloc"
        for match in [_ALLOC.search(op.body)]
        if match
    ]
    waits = [
        int(match.group(1))
        for op in module.ops
        if op.name == "ttg.async_wait"
        for match in [_WAIT.search(op.body)]
        if match
    ]
    every_kind = [*kinds.values(), *(f"ttg.{name}" for name in inline)]
    return {
        **common_summary(module),
        "attributes": dict(sorted(attributes.items())),
        "num_warps": int(attributes["ttg.num-warps"]),
        "threads_per_warp": int(attributes["ttg.threads-per-warp"]),
        "target": attributes["ttg.target"],
        "encodings": {
            name: {"kind": kind, "parameters": parameters}
            for name, kind, parameters in module.aliases
        },
        "inline_encodings": dict(sorted(inline.items())),
        "mma_encoding": any("mma" in kind for kind in every_kind),
        "dot_operands": [
            {"op_idx": index, "parent": parent, "parent_kind": kinds.get(parent[1:])}
            for index, parent in parents
        ],
        "local_alloc": allocations,
        "async_copy_global_to_local": module.count("ttg.async_copy_global_to_local"),
        "async_commit_group": module.count("ttg.async_commit_group"),
        "async_wait": waits,
        "local_load": module.count("ttg.local_load"),
        "convert_layout": module.count("ttg.convert_layout"),
    }
