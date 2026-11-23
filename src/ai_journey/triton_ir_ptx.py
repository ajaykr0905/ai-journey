"""Read a PTX text that Triton wrote, with no triton import and no ptxas.

PTX is the virtual instruction set that the CUDA driver or ``ptxas`` turns
into machine code for one GPU. Triton writes one ``.entry`` function per
kernel. ``read_ptx`` reads the parts that a learner needs first:

- the header: ``.version``, ``.target`` and ``.address_size``;
- the entry name and its ``.param`` list, ``.reqntid`` (threads per block);
- the ``.reg`` declarations by class. These are virtual registers, so a count
  is not the register use of the machine code: only ``ptxas`` decides that;
- each ``.shared`` declaration. Triton declares one ``extern`` array with no
  size and gives its size at launch, so the size comes from the compile
  metadata and not from the PTX;
- a histogram of every opcode and of the opcode families in ``FAMILIES``,
  for the whole entry and for each loop: a label that a later branch jumps
  back to.

Each instruction also gets the line of the Triton source that the last
``.loc`` names. A ``.loc`` of an inlined Triton helper names its call site
with ``inlined_at``, so the instruction gets the line of the call.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .triton_ir_manifest import TritonIrError

FAMILIES = (
    "ld.global",
    "st.global",
    "cp.async",
    "cp.async.commit_group",
    "cp.async.wait_group",
    "ld.shared",
    "ldmatrix",
    "st.shared",
    "bar.sync",
    "fma.rn.f32",
    "mma",
    "setp",
    "selp",
    "branch",
    "other",
)
SOURCE_PREFIX = "<ai_journey.triton_matmul_kernel.TRITON_SOURCE"
_HEADER = {
    "version": re.compile(r"^\.version\s+(\S+)$"),
    "target": re.compile(r"^\.target\s+(\S+)$"),
    "address_size": re.compile(r"^\.address_size\s+(\d+)$"),
}
_ENTRY = re.compile(r"^\.visible\s+\.entry\s+(\w+)\($")
_PARAM = re.compile(r"^\.param\s+(.+?)\s+(\w+),?$")
_REQNTID = re.compile(r"^\.reqntid\s+(\d+(?:\s*,\s*\d+)*)$")
_REG = re.compile(r"^\.reg\s+\.(\w+)\s+%(\w+)<(\d+)>;$")
_SHARED = re.compile(
    r"^(\.extern\s+)?\.shared\s+\.align\s+(\d+)\s+\.(\w+)\s+(\w+)\[(\d*)\];$"
)
_FILE = re.compile(r'^\.file\s+(\d+)\s+"([^"]*)"')
_LOC = re.compile(r"^\.loc\s+(\d+)\s+(\d+)\s+\d+(.*)$")
_INLINED = re.compile(r"inlined_at\s+(\d+)\s+(\d+)\s+\d+")
_LABEL = re.compile(r"^(\$\w+):")
_GUARD = re.compile(r"^@(!?)%(\w+)\s+")


def family(opcode: str) -> str:
    """Return the family in ``FAMILIES`` of one PTX opcode."""

    if opcode.startswith(("ld.global", "st.global", "ld.shared", "st.shared")):
        return opcode[:9]
    if opcode == "cp.async.commit_group":
        return opcode
    if opcode.startswith(("cp.async.wait_group", "cp.async.wait_all")):
        return "cp.async.wait_group"
    if opcode.startswith("cp.async."):
        return "cp.async"
    if opcode.startswith("ldmatrix."):
        return "ldmatrix"
    if opcode.startswith(("bar.sync", "barrier.sync")):
        return "bar.sync"
    if opcode == "fma.rn.f32":
        return opcode
    if opcode.startswith(("mma.", "wgmma.")):
        return "mma"
    if opcode.startswith("setp."):
        return "setp"
    if opcode.startswith("selp."):
        return "selp"
    if opcode == "bra" or opcode.startswith("bra."):
        return "branch"
    return "other"


def family_counts(opcodes: Counter) -> dict[str, int]:
    counts = dict.fromkeys(FAMILIES, 0)
    for opcode, count in opcodes.items():
        counts[family(opcode)] += count
    return counts


@dataclass(frozen=True)
class PtxInstruction:
    """One instruction of the entry body: its PTX line, opcode and source line."""

    line: int
    opcode: str
    guard: str | None
    operands: str
    source_line: int | None

    @property
    def family(self) -> str:
        return family(self.opcode)


@dataclass(frozen=True)
class PtxLoop:
    """A label that a later branch of the entry jumps back to, and its span."""

    label: str
    first_line: int
    last_line: int
    opcodes: dict[str, int]

    def to_mapping(self) -> dict[str, object]:
        return {
            "label": self.label,
            "first_line": self.first_line,
            "last_line": self.last_line,
            "instructions": sum(self.opcodes.values()),
            "families": family_counts(Counter(self.opcodes)),
        }


@dataclass(frozen=True)
class PtxSummary:
    """The header, entry, declarations and instruction counts of one PTX text."""

    version: str
    target: str
    address_size: int
    entry: str
    parameters: tuple[tuple[str, str], ...]
    reqntid: str | None
    registers: tuple[tuple[str, str, int], ...]
    shared: tuple[dict[str, object], ...]
    source_file: int | None
    instructions: tuple[PtxInstruction, ...]
    loops: tuple[PtxLoop, ...]

    @property
    def opcodes(self) -> dict[str, int]:
        counts = Counter(item.opcode for item in self.instructions)
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    @property
    def families(self) -> dict[str, int]:
        return family_counts(Counter(item.opcode for item in self.instructions))

    def to_mapping(self) -> dict[str, object]:
        return {
            "version": self.version,
            "target": self.target,
            "address_size": self.address_size,
            "entry": self.entry,
            "parameters": [
                {"name": name, "type": kind} for kind, name in self.parameters
            ],
            "reqntid": self.reqntid,
            "registers": [
                {"class": kind, "prefix": prefix, "declared": count}
                for kind, prefix, count in self.registers
            ],
            "shared": list(self.shared),
            "instructions": len(self.instructions),
            "predicated": sum(item.guard is not None for item in self.instructions),
            "families": self.families,
            "opcodes": self.opcodes,
            "loops": [loop.to_mapping() for loop in self.loops],
        }


def _strip(line: str) -> str:
    return line.split("//", 1)[0].strip()


def _header(lines: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in lines:
        text = _strip(line)
        for name, pattern in _HEADER.items():
            match = pattern.match(text)
            if match:
                if name in found:
                    raise TritonIrError(f"a PTX text declares .{name} twice")
                found[name] = match.group(1)
    missing = [name for name in _HEADER if name not in found]
    if missing:
        raise TritonIrError(f"a PTX text must declare .{missing[0]}")
    return found


def _source_file(lines: list[str]) -> int | None:
    files = [
        int(match.group(1))
        for match in (_FILE.match(line.strip()) for line in lines)
        if match and match.group(2).startswith(SOURCE_PREFIX)
    ]
    if len(files) > 1:
        raise TritonIrError("a PTX text names the kernel source in two .file lines")
    return files[0] if files else None


def _entry(lines: list[str]) -> tuple[str, list[tuple[str, str]], int]:
    """Return the entry name, its parameters and the index of its last line."""

    entries = [index for index, line in enumerate(lines) if _ENTRY.match(line.strip())]
    if len(entries) != 1:
        raise TritonIrError("a PTX text must hold exactly one .entry")
    start = entries[0]
    name = _ENTRY.match(lines[start].strip()).group(1)
    parameters = []
    for index in range(start + 1, len(lines)):
        text = _strip(lines[index])
        if text == ")":
            return name, parameters, index
        match = _PARAM.match(text)
        if not match:
            raise TritonIrError(f"PTX line {index + 1} is not an entry parameter")
        parameters.append((match.group(1), match.group(2)))
    raise TritonIrError("the .entry parameter list never closes")


def _body(lines: list[str], after: int) -> tuple[int, int]:
    opened = [index for index in range(after, len(lines)) if lines[index] == "{"]
    if not opened:
        raise TritonIrError("the .entry has no body")
    closed = [index for index in range(opened[0], len(lines)) if lines[index] == "}"]
    if not closed:
        raise TritonIrError("the .entry body never closes")
    return opened[0], closed[0]


def _loops(
    instructions: list[PtxInstruction], labels: dict[str, int]
) -> tuple[PtxLoop, ...]:
    loops = []
    for label, line in sorted(labels.items(), key=lambda item: item[1]):
        back = [
            item.line
            for item in instructions
            if item.family == "branch"
            and item.operands.rstrip(";").strip() == label
            and item.line > line
        ]
        if back:
            last = max(back)
            inside = Counter(
                item.opcode for item in instructions if line < item.line <= last
            )
            loops.append(PtxLoop(label, line, last, dict(sorted(inside.items()))))
    return tuple(loops)


def read_ptx(text: str) -> PtxSummary:
    """Parse one PTX text with one ``.entry``; raise ``TritonIrError`` if not."""

    if type(text) is not str or not text.isascii():
        raise TritonIrError("a PTX text must be an ASCII string")
    lines = text.splitlines()
    header = _header(lines)
    name, parameters, closing = _entry(lines)
    start, end = _body(lines, closing)
    source = _source_file(lines[end:])
    reqntid = None
    for line in lines[closing + 1 : start]:
        match = _REQNTID.match(_strip(line))
        if match:
            reqntid = match.group(1)
    shared = []
    for line in lines[:start]:
        match = _SHARED.match(_strip(line))
        if match:
            shared.append(
                {
                    "name": match.group(4),
                    "extern": match.group(1) is not None,
                    "align": int(match.group(2)),
                    "element": match.group(3),
                    "elements": int(match.group(5)) if match.group(5) else None,
                }
            )
    registers: list[tuple[str, str, int]] = []
    instructions: list[PtxInstruction] = []
    labels: dict[str, int] = {}
    current: int | None = None
    for index in range(start + 1, end):
        number = index + 1
        raw = lines[index].strip()
        loc = _LOC.match(raw)
        if loc:
            inlined = _INLINED.search(loc.group(3))
            file, line = (
                (int(inlined.group(1)), int(inlined.group(2)))
                if inlined
                else (int(loc.group(1)), int(loc.group(2)))
            )
            # LLVM writes line 0 for code that has no source line.
            current = line if file == source and line > 0 else None
            continue
        code = _strip(raw)
        if not code:
            continue
        label = _LABEL.match(code)
        if label:
            labels[label.group(1)] = number
            continue
        register = _REG.match(code)
        if register:
            registers.append(
                (register.group(1), register.group(2), int(register.group(3)))
            )
            continue
        if code.startswith(".") or code in ("{", "}"):
            continue
        guard = _GUARD.match(code)
        if guard:
            code = code[guard.end() :]
        opcode, _, operands = code.partition(" ")
        if not opcode.endswith(";") and not operands:
            raise TritonIrError(f"PTX line {number} is not an instruction")
        instructions.append(
            PtxInstruction(
                line=number,
                opcode=opcode.rstrip(";"),
                guard=None if guard is None else guard.group(1) + "%" + guard.group(2),
                operands=operands.strip(),
                source_line=current,
            )
        )
    return PtxSummary(
        version=header["version"],
        target=header["target"],
        address_size=int(header["address_size"]),
        entry=name,
        parameters=tuple(parameters),
        reqntid=reqntid,
        registers=tuple(registers),
        shared=tuple(shared),
        source_file=source,
        instructions=tuple(instructions),
        loops=_loops(instructions, labels),
    )
