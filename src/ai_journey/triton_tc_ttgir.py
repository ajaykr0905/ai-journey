"""Read the MMA, dot-operand and shared-memory encodings of a TTGIR text.

TTGIR gives every tensor an encoding: which thread holds which element. The
encodings that decide Tensor Core use are these:

- ``#ttg.nvidia_mma<{versionMajor = 2, ..., instrShape = [16, 8]}>``: the
  accumulator fragments of a Tensor Core instruction. Version 2 is the
  ``mma.sync`` layout of sm_80, and version 3 is the ``wgmma`` layout of
  sm_90. ``instrShape`` is the shape of one instruction, and
  ``warpsPerCTA`` places the warps on the tile;
- ``#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>``: the register
  fragments of operand A (``opIdx = 0``) or B (``opIdx = 1``) of a dot whose
  result has the ``parent`` layout. ``kWidth`` is the number of consecutive
  K elements that one thread holds. A ``#blocked`` parent means a dot with
  no Tensor Core: each thread runs scalar FMAs;
- ``#ttg.swizzled_shared<{vec, perPhase, maxPhase, order}>``: a buffer in
  shared memory whose rows are swizzled. ``swizzled_column`` applies the
  rule: each group of ``vec`` elements moves to group
  ``group XOR ((row // perPhase) % maxPhase)``. So rows that a warp reads
  together start in different banks;
- ``#ttg.nvmma_shared<{swizzlingByteWidth, transposed, elementBitWidth}>``:
  the shared layout that a ``wgmma`` descriptor reads. It names the swizzle
  by its byte width, for example 128, as the descriptor does.

``analyze_ttgir`` reads the encoding aliases, the inline encodings, the
module attributes, the dot ops and every ``ttg.local_alloc`` buffer with its
shape, element type, encoding and bytes. It needs no triton.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

from .triton_tc_kernel import TritonTcError

MMA_KINDS = ("nvidia_mma", "mma")
SHARED_KINDS = ("swizzled_shared", "nvmma_shared", "shared", "shared_linear")
DOT_OPS = ("tt.dot", "ttng.warp_group_dot")
COUNTED_OPS = (
    "tt.dot",
    "ttng.warp_group_dot",
    "ttng.warp_group_dot_wait",
    "ttg.local_alloc",
    "ttg.local_load",
    "ttg.local_store",
    "ttg.async_copy_global_to_local",
    "ttg.async_commit_group",
    "ttg.async_wait",
    "ttg.convert_layout",
    "ttng.fence_async_shared",
    "ttng.async_tma_copy_global_to_local",
    "scf.for",
)
_ALIAS = re.compile(r"^#(\w+) = #(?:ttg|triton_gpu|ttng)\.(\w+)(<.*>)?$")
_INLINE = re.compile(r"#(?:ttg|triton_gpu)\.(\w+)<")
_MODULE = re.compile(r"^module(?: attributes \{(.*)\})? \{$")
_OP = re.compile(r'^(?:%[^=]*=\s*)?"?([a-z_][a-z_0-9]*(?:\.[a-z_0-9]+)+)"?')
_MEMDESC = "!ttg.memdesc<"
_RESULT = re.compile(r"->\s*tensor<[^,>]*,\s*(#\w+)>")
_PRECISION = re.compile(r"inputPrecision = (\w+)")
_INTEGER = re.compile(r"-?\d+")
_ELEMENT_BITS = {"f64": 64, "f32": 32, "tf32": 32, "f16": 16, "bf16": 16}
_ELEMENT = re.compile(r"(?:[fiu](\d+)|f8\w+)")


def _closing(text: str, start: int) -> int:
    """Return the index of the ``>`` that closes the ``<`` at ``start``."""

    depth = 0
    for index in range(start, len(text)):
        if text[index] == "<":
            depth += 1
        elif text[index] == ">" and text[index - 1] != "-":
            depth -= 1
            if depth == 0:
                return index
    raise TritonTcError("an encoding never closes its <")


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


def _value(text: str) -> object:
    text = text.strip()
    if _INTEGER.fullmatch(text):
        return int(text)
    if text in ("true", "false"):
        return text == "true"
    if text.startswith("[") and text.endswith("]"):
        return [_value(part) for part in _split_top(text[1:-1])]
    return text.strip('"')


def parse_parameters(text: str) -> dict[str, object]:
    """Parse ``<{name = value, ...}>`` of one encoding into a mapping."""

    if not text:
        return {}
    body = text.strip()
    if not (body.startswith("<{") and body.endswith("}>")):
        raise TritonTcError(f"encoding parameters {text!r} are not <{{...}}>")
    found: dict[str, object] = {}
    unparsed = []
    for part in _split_top(body[2:-2]):
        name, equals, value = part.partition("=")
        if equals and name.strip().isidentifier():
            found[name.strip()] = _value(value)
        else:
            unparsed.append(part)
    if unparsed:
        # Kept as text, so a new parameter form stays visible and is not lost.
        found["unparsed"] = unparsed
    return found


@dataclass(frozen=True)
class Encoding:
    """One encoding: its alias (or None inline), its kind and its parameters."""

    alias: str | None
    kind: str
    parameters: Mapping[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "alias": self.alias,
            "kind": self.kind,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class SharedBuffer:
    """One ``ttg.local_alloc`` buffer: shape, element type, encoding, bytes."""

    line: int
    shape: tuple[int, ...]
    element: str
    element_bits: int
    encoding: str
    encoding_kind: str | None
    memory: str | None

    @property
    def bytes(self) -> int:
        return prod(self.shape) * self.element_bits // 8

    def to_mapping(self) -> dict[str, object]:
        return {
            "line": self.line,
            "shape": list(self.shape),
            "element": self.element,
            "element_bits": self.element_bits,
            "encoding": self.encoding,
            "encoding_kind": self.encoding_kind,
            "memory": self.memory,
            "bytes": self.bytes,
        }


def element_bits(element: str) -> int:
    """Return the bits of one MLIR element type such as f32, bf16 or i64."""

    if element in _ELEMENT_BITS:
        return _ELEMENT_BITS[element]
    match = _ELEMENT.fullmatch(element)
    if match is None:
        raise TritonTcError(f"element type {element!r} has no known width")
    return int(match.group(1)) if match.group(1) else 8


def parse_memdesc(text: str) -> tuple[tuple[int, ...], str, str, str | None]:
    """Read ``2x128x32xbf16, #shared, #smem, mutable`` of one memdesc type."""

    parts = _split_top(text)
    if len(parts) < 2:
        raise TritonTcError(f"memdesc {text!r} must name a shape and an encoding")
    *dimensions, element = parts[0].split("x")
    if not dimensions or not all(item.isdigit() for item in dimensions):
        raise TritonTcError(f"memdesc {text!r} must name a static shape")
    memory = parts[2] if len(parts) > 2 and parts[2].startswith("#") else None
    return tuple(int(item) for item in dimensions), element, parts[1], memory


@dataclass(frozen=True)
class TtgirAnalysis:
    """The encodings, dots, buffers and module facts of one TTGIR text."""

    target: str
    num_warps: int
    threads_per_warp: int
    encodings: tuple[Encoding, ...]
    inline: tuple[Encoding, ...]
    ops: Mapping[str, int]
    dots: tuple[Mapping[str, object], ...]
    buffers: tuple[SharedBuffer, ...]

    def kind(self, alias: str) -> str | None:
        name = alias.removeprefix("#")
        for encoding in self.encodings:
            if encoding.alias == name:
                return encoding.kind
        return None

    @property
    def mma(self) -> tuple[Encoding, ...]:
        return tuple(item for item in self.encodings if item.kind in MMA_KINDS)

    @property
    def shared(self) -> tuple[Encoding, ...]:
        return tuple(item for item in self.encodings if item.kind in SHARED_KINDS)

    @property
    def dot_operands(self) -> tuple[Encoding, ...]:
        return tuple(item for item in self.inline if item.kind == "dot_op")

    @property
    def mma_versions(self) -> tuple[int, ...]:
        return tuple(
            sorted({int(item.parameters.get("versionMajor", 0)) for item in self.mma})
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "target": self.target,
            "num_warps": self.num_warps,
            "threads_per_warp": self.threads_per_warp,
            "encodings": [item.to_mapping() for item in self.encodings],
            "mma": [item.to_mapping() for item in self.mma],
            "mma_versions": list(self.mma_versions),
            "shared": [item.to_mapping() for item in self.shared],
            "dot_operands": [
                {
                    **item.to_mapping(),
                    "parent_kind": self.kind(str(item.parameters.get("parent", ""))),
                }
                for item in self.dot_operands
            ],
            "ops": dict(self.ops),
            "dots": [dict(item) for item in self.dots],
            "buffers": [item.to_mapping() for item in self.buffers],
            "buffer_bytes": sum(item.bytes for item in self.buffers),
        }


def _inline(line: str) -> list[Encoding]:
    found = []
    for match in _INLINE.finditer(line):
        end = _closing(line, match.end() - 1)
        text = line[match.end() - 1 : end + 1]
        found.append(Encoding(None, match.group(1), parse_parameters(text)))
    return found


def _attributes(text: str | None) -> dict[str, str]:
    found = {}
    for part in _split_top(text or ""):
        name, _, value = part.partition("=")
        found[name.strip().strip('"')] = value.split(" : ")[0].strip().strip('"')
    return found


def analyze_ttgir(text: str) -> TtgirAnalysis:
    """Read one TTGIR module; raise ``TritonTcError`` when it is not one."""

    if type(text) is not str:
        raise TritonTcError("a TTGIR text must be a string")
    encodings: list[Encoding] = []
    inline: dict[str, Encoding] = {}
    modules = []
    ops: Counter = Counter()
    dots = []
    buffers = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        alias = _ALIAS.match(line)
        if alias:
            encodings.append(
                Encoding(
                    alias.group(1),
                    alias.group(2),
                    parse_parameters(alias.group(3) or ""),
                )
            )
            continue
        module = _MODULE.match(line)
        if module:
            modules.append(_attributes(module.group(1)))
            continue
        if not line or line.startswith(("#", "}", "^")):
            continue
        for item in _inline(line):
            inline.setdefault(repr(item.to_mapping()), item)
        match = _OP.match(line)
        if not match:
            continue
        name = match.group(1)
        ops[name] += 1
        if name in DOT_OPS:
            result = _RESULT.search(line)
            precision = _PRECISION.search(line)
            dots.append(
                {
                    "op": name,
                    "line": number,
                    "result_encoding": result.group(1) if result else None,
                    "input_precision": precision.group(1) if precision else None,
                }
            )
        if name == "ttg.local_alloc":
            start = line.rfind(_MEMDESC)
            if start < 0:
                raise TritonTcError(f"TTGIR line {number} allocates no memdesc")
            end = _closing(line, start + len(_MEMDESC) - 1)
            shape, element, encoding, memory = parse_memdesc(
                line[start + len(_MEMDESC) : end]
            )
            buffers.append((number, shape, element, encoding, memory))
    if len(modules) != 1:
        raise TritonTcError("a TTGIR text must hold exactly one module")
    attributes = modules[0]
    for name in ("ttg.num-warps", "ttg.threads-per-warp", "ttg.target"):
        if name not in attributes:
            raise TritonTcError(f"a TTGIR module must name {name}")
    kinds = {f"#{item.alias}": item.kind for item in encodings}
    for dot in dots:
        dot["result_kind"] = kinds.get(dot["result_encoding"])
    return TtgirAnalysis(
        target=attributes["ttg.target"],
        num_warps=int(attributes["ttg.num-warps"]),
        threads_per_warp=int(attributes["ttg.threads-per-warp"]),
        encodings=tuple(encodings),
        inline=tuple(inline.values()),
        ops={name: ops.get(name, 0) for name in COUNTED_OPS},
        dots=tuple(dots),
        buffers=tuple(
            SharedBuffer(
                line=number,
                shape=shape,
                element=element,
                element_bits=element_bits(element),
                encoding=encoding,
                encoding_kind=kinds.get(encoding),
                memory=memory,
            )
            for number, shape, element, encoding, memory in buffers
        ),
    )


def swizzled_column(
    row: int, column: int, vec: int, per_phase: int, max_phase: int
) -> int:
    """Return the stored column of element ``(row, column)`` of a swizzled row.

    The group of ``vec`` elements that holds the column moves to group
    ``group XOR ((row // per_phase) % max_phase)``; the element keeps its
    place inside the group. XOR is its own inverse, so the same rule maps a
    stored column back.
    """

    for name, value in (
        ("row", row),
        ("column", column),
        ("vec", vec),
        ("per_phase", per_phase),
        ("max_phase", max_phase),
    ):
        low = 0 if name in ("row", "column") else 1
        if type(value) is not int or value < low:
            raise TritonTcError(f"{name} must be an integer of at least {low}")
    phase = (row // per_phase) % max_phase
    return ((column // vec) ^ phase) * vec + column % vec


def swizzle_table(
    rows: int, columns: int, vec: int, per_phase: int, max_phase: int
) -> list[list[int]]:
    """Return, for each row, the stored column of each logical column."""

    swizzled_column(0, 0, vec, per_phase, max_phase)
    if type(rows) is not int or type(columns) is not int or min(rows, columns) < 1:
        raise TritonTcError("rows and columns must be positive integers")
    if columns % vec or (columns // vec) & (columns // vec - 1):
        raise TritonTcError("columns must be vec times a power of two")
    if max_phase > columns // vec:
        raise TritonTcError("max_phase must not move a group out of its row")
    return [
        [
            swizzled_column(row, column, vec, per_phase, max_phase)
            for column in range(columns)
        ]
        for row in range(rows)
    ]


def shared_bank(
    row: int,
    column: int,
    *,
    columns: int,
    element_bytes: int,
    vec: int,
    per_phase: int,
    max_phase: int,
    banks: int = 32,
    bank_bytes: int = 4,
) -> int:
    """Return the shared memory bank of one element of a swizzled row-major tile.

    Shared memory has 32 banks of 4 bytes. Elements that a warp reads in one
    step and that fall in one bank, at different addresses, wait for each
    other: a bank conflict.
    """

    stored = swizzled_column(row, column, vec, per_phase, max_phase)
    if stored >= columns:
        raise TritonTcError("the swizzle moved a group out of its row")
    return (row * columns + stored) * element_bytes // bank_bytes % banks
