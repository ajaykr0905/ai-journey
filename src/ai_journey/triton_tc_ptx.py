"""Count the Tensor Core and scalar math instructions of a PTX text by rules.

A Tensor Core runs one matrix multiply-accumulate (MMA) ``D = A x B + C`` on
small tiles. PTX names it in two forms:

- ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``: one warp of 32
  threads computes one 16 x 8 x 16 product together. Each thread holds a
  fixed part of A, B and C in its registers, a fragment. This is the form of
  sm_80;
- ``wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16``: one warpgroup
  of 4 warps computes one 64 x 128 x 16 product. A and B can come from
  shared memory through descriptors, and the instruction runs
  asynchronously until a ``wgmma.wait_group``. Only sm_90a has this form.

``read_instructions`` splits the text into statements at each ``;``, drops
comments, directives, labels and predicate guards, and takes the first word
as the opcode. Thus an instruction inside an inline-asm block counts, as
Triton writes each MMA as inline asm. ``RULES`` then gives each opcode one
class, first match wins. An MMA opcode must name its shape and its operand
types, or the analysis fails: a rule that the compiler output does not match
must not count silently.

``static_tile_products`` multiplies each count by the multiply-adds of its
instruction and by the threads, warps or warpgroups of one program. The
text holds one K-loop body, so a value of 1 says that the static code holds
exactly one ``BLOCK_M x BLOCK_N x BLOCK_K`` tile product. It counts the
instructions in the text, not the instructions that a GPU runs.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from .triton_tc_kernel import TC_CONFIG, TcConfig, TritonTcError

THREADS_PER_WARP = 32
WARPS_PER_WARPGROUP = 4
CLASSES = (
    "mma_sync",
    "wgmma_mma_async",
    "wgmma_fence",
    "wgmma_commit_group",
    "wgmma_wait_group",
    "ldmatrix",
    "stmatrix",
    "cp_async",
    "cp_async_commit_group",
    "cp_async_wait_group",
    "cp_async_bulk",
    "cvt_tf32",
    "fma_rn_f32",
)
TENSOR_CORE_CLASSES = ("mma_sync", "wgmma_mma_async")
# (class, kind, text): kind "exact" matches the whole opcode, "prefix" its
# start. First match wins, so the bulk copies come before the other copies.
RULES = (
    ("mma_sync", "prefix", "mma.sync.aligned."),
    ("wgmma_mma_async", "prefix", "wgmma.mma_async."),
    ("wgmma_fence", "exact", "wgmma.fence.sync.aligned"),
    ("wgmma_commit_group", "exact", "wgmma.commit_group.sync.aligned"),
    ("wgmma_wait_group", "exact", "wgmma.wait_group.sync.aligned"),
    ("ldmatrix", "prefix", "ldmatrix.sync.aligned."),
    ("stmatrix", "prefix", "stmatrix.sync.aligned."),
    ("cp_async_bulk", "prefix", "cp.async.bulk."),
    ("cp_async_commit_group", "exact", "cp.async.commit_group"),
    ("cp_async_wait_group", "exact", "cp.async.wait_group"),
    ("cp_async_wait_group", "exact", "cp.async.wait_all"),
    ("cp_async", "prefix", "cp.async.ca."),
    ("cp_async", "prefix", "cp.async.cg."),
    ("cvt_tf32", "suffix", ".tf32.f32"),
    ("fma_rn_f32", "exact", "fma.rn.f32"),
)
_SHAPE = re.compile(r"m(\d+)n(\d+)k(\d+)")
_TYPE = re.compile(r"[a-z]+[0-9]*(?:x[0-9]+)?")
_LAYOUTS = ("row", "col")
_QUALIFIERS = ("sync", "aligned", "satfinite", "sp")
_VERSION = re.compile(r"^\.version\s+(\d+\.\d+)$")
_TARGET = re.compile(r"^\.target\s+([\w,\s]+)$")
_LABEL = re.compile(r"^[A-Za-z_$][\w$]*:")
_GUARD = re.compile(r"^@!?%?\w+\s+")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/")


def classify(opcode: str) -> str | None:
    """Return the class in ``CLASSES`` of one opcode, or None for no class."""

    for name, kind, text in RULES:
        if kind == "exact" and opcode == text:
            return name
        if kind == "prefix" and opcode.startswith(text):
            return name
        if kind == "suffix" and opcode.startswith("cvt.") and opcode.endswith(text):
            return name
    return None


@dataclass(frozen=True)
class MmaForm:
    """One MMA opcode of the text: its shape, operand types and count."""

    opcode: str
    family: str
    m: int
    n: int
    k: int
    d_type: str
    a_type: str
    b_type: str
    c_type: str | None
    layouts: tuple[str, ...]
    count: int

    @property
    def shape(self) -> str:
        return f"m{self.m}n{self.n}k{self.k}"

    @property
    def multiply_adds(self) -> int:
        """Return the multiply-adds of one instruction: m x n x k."""

        return self.m * self.n * self.k

    def to_mapping(self) -> dict[str, object]:
        return {
            "opcode": self.opcode,
            "family": self.family,
            "shape": self.shape,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "d_type": self.d_type,
            "a_type": self.a_type,
            "b_type": self.b_type,
            "c_type": self.c_type,
            "layouts": list(self.layouts),
            "count": self.count,
            "multiply_adds": self.multiply_adds,
        }


def parse_mma(opcode: str, count: int = 1) -> MmaForm:
    """Read the shape and the operand types of one MMA opcode, or refuse it."""

    family = classify(opcode)
    if family not in TENSOR_CORE_CLASSES:
        raise TritonTcError(f"{opcode} is not an MMA opcode")
    words = opcode.split(".")
    shapes = [index for index, word in enumerate(words) if _SHAPE.fullmatch(word)]
    if len(shapes) != 1:
        raise TritonTcError(f"MMA opcode {opcode} must name one mNnNkN shape")
    at = shapes[0]
    m, n, k = (int(value) for value in _SHAPE.fullmatch(words[at]).groups())
    rest = words[at + 1 :]
    layouts = tuple(word for word in rest if word in _LAYOUTS)
    types = [word for word in rest if word not in _LAYOUTS + _QUALIFIERS]
    wanted = 4 if family == "mma_sync" else 3
    if len(types) != wanted or not all(_TYPE.fullmatch(word) for word in types):
        raise TritonTcError(
            f"MMA opcode {opcode} must name {wanted} operand types after its shape"
        )
    return MmaForm(
        opcode=opcode,
        family=family,
        m=m,
        n=n,
        k=k,
        d_type=types[0],
        a_type=types[1],
        b_type=types[2],
        c_type=types[3] if family == "mma_sync" else None,
        layouts=layouts,
        count=count,
    )


def read_instructions(text: str) -> list[tuple[int, str]]:
    """Return ``(line, opcode)`` for every instruction statement of a PTX text."""

    if type(text) is not str:
        raise TritonTcError("a PTX text must be a string")
    found = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _BLOCK_COMMENT.sub(" ", raw.split("//", 1)[0])
        for part in line.split(";"):
            statement = part.strip().lstrip("{}").strip()
            while _LABEL.match(statement):
                statement = statement[_LABEL.match(statement).end() :].strip()
            # The ")" closes the parameter list of the entry.
            if not statement or statement.startswith((".", "}", "{", ")")):
                continue
            guard = _GUARD.match(statement)
            if guard:
                statement = statement[guard.end() :]
            found.append((number, statement.split()[0]))
    return found


def _header(text: str, pattern: re.Pattern, name: str) -> str:
    lines = (line.split("//", 1)[0].strip() for line in text.splitlines())
    values = [match.group(1).strip() for match in map(pattern.match, lines) if match]
    if len(values) != 1:
        raise TritonTcError(f"a PTX text must declare .{name} once")
    return values[0]


@dataclass(frozen=True)
class PtxAnalysis:
    """The header, the class counts and the MMA forms of one PTX text."""

    version: str
    target: str
    instructions: int
    classes: Mapping[str, int]
    mma_forms: tuple[MmaForm, ...]
    ldmatrix_forms: Mapping[str, int]

    @property
    def tensor_core_classes(self) -> tuple[str, ...]:
        """Return the MMA classes that the text holds, in ``CLASSES`` order."""

        return tuple(name for name in TENSOR_CORE_CLASSES if self.classes[name])

    def to_mapping(self) -> dict[str, object]:
        return {
            "version": self.version,
            "target": self.target,
            "instructions": self.instructions,
            "classes": dict(self.classes),
            "mma_forms": [form.to_mapping() for form in self.mma_forms],
            "ldmatrix_forms": dict(self.ldmatrix_forms),
            "tensor_core_classes": list(self.tensor_core_classes),
        }


def analyze_ptx(text: str) -> PtxAnalysis:
    """Apply ``RULES`` to every instruction of one PTX text."""

    if type(text) is not str:
        raise TritonTcError("a PTX text must be a string")
    version = _header(text, _VERSION, "version")
    target = _header(text, _TARGET, "target")
    opcodes = Counter(opcode for _, opcode in read_instructions(text))
    classes = dict.fromkeys(CLASSES, 0)
    forms = []
    loads: dict[str, int] = {}
    for opcode, count in sorted(opcodes.items()):
        name = classify(opcode)
        if name is None:
            continue
        classes[name] += count
        if name in TENSOR_CORE_CLASSES:
            forms.append(parse_mma(opcode, count))
        elif name == "ldmatrix":
            loads[opcode] = count
    return PtxAnalysis(
        version=version,
        target=target,
        instructions=sum(opcodes.values()),
        classes=classes,
        mma_forms=tuple(forms),
        ldmatrix_forms=loads,
    )


def static_tile_products(
    analysis: PtxAnalysis, config: TcConfig = TC_CONFIG
) -> dict[str, Fraction]:
    """Return, per math class, the tile products that the static text holds.

    One ``fma.rn.f32`` is 1 multiply-add of one thread, one ``mma.sync`` is
    ``m*n*k`` of one warp, and one ``wgmma.mma_async`` is ``m*n*k`` of one
    warpgroup. A program has ``num_warps`` warps.
    """

    if type(analysis) is not PtxAnalysis:
        raise TypeError("analysis must be PtxAnalysis")
    if type(config) is not TcConfig:
        raise TypeError("config must be TcConfig")
    tile = config.block_m * config.block_n * config.block_k
    groups = {
        "mma_sync": Fraction(config.num_warps),
        "wgmma_mma_async": Fraction(config.num_warps, WARPS_PER_WARPGROUP),
    }
    products = {
        "fma_rn_f32": Fraction(
            analysis.classes["fma_rn_f32"] * config.num_warps * THREADS_PER_WARP, tile
        )
    }
    for name in TENSOR_CORE_CLASSES:
        issued = sum(
            form.count * form.multiply_adds
            for form in analysis.mma_forms
            if form.family == name
        )
        products[name] = issued * groups[name] / tile
    return products
