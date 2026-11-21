"""Lines of code of the Day 79 Triton text and of the Day 76 and Day 77 kernel texts.

The exercise asks for the lines of code of the Triton port and of the CUDA
version. ``COUNTING_RULE`` states the rule, and each text gets one
``LineCount``:

- A line is code when it holds code after its comments are removed. A blank
  line and a line that holds only a comment are not code. Comments are ``//``
  and ``/* */`` in CUDA C++ and in Metal, and ``#`` comments and docstrings
  in Python.
- Each code line belongs to one top-level definition. Kernel lines are the
  lines of a GPU function: a ``__global__`` function in CUDA, a
  ``kernel void`` function in Metal and a ``@triton.jit`` function in Triton,
  with its decorator. A helper that a kernel calls also counts as kernel.
- Host lines are the lines of the host launch code: the launcher and each
  helper that only host code calls, such as ``cdiv`` in the CUDA texts. A
  Metal text holds no host code: the Day 76 and Day 77 runtimes launch its
  kernels from Python, and those modules also time and check them. Thus a
  Metal text has no host count.
- Other lines are includes, imports and other top-level statements.

The counts are physical lines, so they depend on how each text is formatted.
Each Day 77 Metal text comes from a generator that writes one statement for
each output of a thread, so its count measures generated text. The
``cublasSgemm`` call text is not a kernel, so the rule does not count it.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass

from .gpu_checklist_cuda import PINNED_KERNELS, cuda_source
from .gpu_tiled_cuda import TILES, tiled_cuda_source
from .kernel_launch_cuda import CUDA_LANGUAGE, _code
from .matmul_checklist_source import block_kernel_source
from .matmul_checklist_variants import DESIGNS
from .matmul_tiled_source import TILE_SIZES, tiled_kernel_name, tiled_kernel_source
from .triton_matmul_kernel import (
    KERNEL_NAME,
    LANGUAGE,
    TRITON_SOURCE,
    TritonMatmulError,
    source_sha256,
)

METAL_LANGUAGE = "Metal Shading Language"
LANGUAGES = (LANGUAGE, CUDA_LANGUAGE, METAL_LANGUAGE)
LABS = ("day-79", "day-76", "day-77")
CATEGORIES = ("kernel", "host", "other", "comment", "blank")
COUNTING_RULE = (
    "A line is code when it holds code after its comments are removed; blank "
    "lines and comment-only lines are not code. Kernel lines are the code "
    "lines of a __global__, kernel void or @triton.jit function with its "
    "decorator, and of each helper that a kernel calls. Host lines are the "
    "code lines of the launcher and of each helper that only host code calls; "
    "a Metal text holds no host code. Other lines are includes, imports and "
    "other top-level statements. Physical lines, as each text is formatted."
)
_NOT_CODE = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
}
_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass(frozen=True)
class LineCount:
    """The lines of one text in each category of the counting rule."""

    kernel: int
    host: int | None
    other: int
    comment: int
    blank: int

    @property
    def total(self) -> int:
        return self.kernel + (self.host or 0) + self.other + self.comment + self.blank

    def to_mapping(self) -> dict[str, int | None]:
        return {
            "kernel": self.kernel,
            "host": self.host,
            "other": self.other,
            "comment": self.comment,
            "blank": self.blank,
            "total": self.total,
        }


def _tally(kinds: list[str], host: bool) -> LineCount:
    counts = {name: kinds.count(name) for name in CATEGORIES}
    return LineCount(
        kernel=counts["kernel"],
        host=counts["host"] if host else None,
        other=counts["other"],
        comment=counts["comment"],
        blank=counts["blank"],
    )


def _lines(text: str) -> list[str]:
    if type(text) is not str or not text.isascii() or not text.endswith("\n"):
        raise TritonMatmulError("a counted text must be ASCII and end in a newline")
    return text.split("\n")[:-1]


def _definitions(code: list[str]) -> list[tuple[str, list[int]]]:
    """Split C-like code lines into top-level units: (header, line indices).

    A unit ends where its braces close at depth 0, or at a ``;`` at depth 0
    before any brace. A preprocessor line is a unit of its own.
    """

    units: list[tuple[str, list[int]]] = []
    current: list[int] = []
    depth = 0
    opened = False
    for index, line in enumerate(code):
        if not line.strip():
            continue
        if not current and line.lstrip().startswith("#"):
            units.append(("#", [index]))
            continue
        current.append(index)
        depth += line.count("{") - line.count("}")
        opened = opened or "{" in line
        if depth < 0:
            raise TritonMatmulError("a counted text closes a brace it never opened")
        if depth == 0 and (opened or line.rstrip().endswith(";")):
            header = " ".join(code[item] for item in current).split("{", 1)[0]
            units.append((header, current))
            current, opened = [], False
    if current or depth:
        raise TritonMatmulError("a counted text ends inside a definition")
    return units


def count_c_like(text: str, *, host_code: bool) -> LineCount:
    """Count a CUDA C++ or Metal text; ``host_code=False`` for a Metal text."""

    original = _lines(text)
    stripped = _code(text)
    if stripped is None or stripped.count("\n") != text.count("\n"):
        raise TritonMatmulError(
            "the rule counts no text with a line splice, a trigraph or a block "
            "comment across lines"
        )
    code = stripped.split("\n")[:-1]
    kinds = [
        "blank" if not line.strip() else "comment" if not part.strip() else ""
        for line, part in zip(original, code)
    ]
    units = _definitions(code)
    kernels = [
        lines
        for header, lines in units
        if "__global__" in header or header.lstrip().startswith("kernel ")
    ]
    kernel_code = " ".join(code[index] for lines in kernels for index in lines)
    for header, lines in units:
        names = _NAME.findall(header)
        if header == "#" or not names:
            kind = "other"
        elif lines in kernels or re.search(
            rf"(?<![A-Za-z0-9_]){names[0]}\s*\(", kernel_code
        ):
            kind = "kernel"
        else:
            kind = "host"
        if kind == "host" and not host_code:
            raise TritonMatmulError("a Metal text must hold kernel functions only")
        for index in lines:
            kinds[index] = kind
    return _tally(kinds, host_code)


def _is_kernel(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef) and any(
        ast.unparse(item) in ("triton.jit", "jit") for item in node.decorator_list
    )


def count_python(text: str) -> LineCount:
    """Count a Triton text with ``tokenize`` for code and ``ast`` for spans."""

    original = _lines(text)
    tree = ast.parse(text)
    code_lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type not in _NOT_CODE:
            code_lines.update(range(token.start[0], token.end[0] + 1))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            code_lines -= set(range(body[0].lineno, body[0].end_lineno + 1))
    kinds = ["blank" if not line.strip() else "comment" for line in original]
    for node in tree.body:
        decorators = getattr(node, "decorator_list", [])
        start = min([node.lineno, *(item.lineno for item in decorators)])
        if _is_kernel(node):
            kind = "kernel"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "host"
        else:
            kind = "other"
        for number in range(start, node.end_lineno + 1):
            if number in code_lines:
                kinds[number - 1] = kind
    return _tally(kinds, True)


@dataclass(frozen=True)
class CountedText:
    """One kernel text: its lab, language, generator and its line count."""

    lab: str
    language: str
    name: str
    origin: str
    text: str

    def __post_init__(self) -> None:
        if self.lab not in LABS or self.language not in LANGUAGES:
            raise TritonMatmulError("a counted text needs a known lab and language")

    @property
    def sha256(self) -> str:
        return source_sha256(self.text)

    def count(self) -> LineCount:
        if self.language == LANGUAGE:
            return count_python(self.text)
        return count_c_like(self.text, host_code=self.language == CUDA_LANGUAGE)

    def to_mapping(self) -> dict[str, object]:
        return {
            "lab": self.lab,
            "language": self.language,
            "name": self.name,
            "origin": self.origin,
            "sha256": self.sha256,
            "lines": self.count().to_mapping(),
        }


def counted_texts() -> tuple[CountedText, ...]:
    """Return the Triton text, then the CUDA texts, then the Metal texts."""

    return (
        CountedText(
            "day-79",
            LANGUAGE,
            KERNEL_NAME,
            "ai_journey.triton_matmul_kernel.TRITON_SOURCE",
            TRITON_SOURCE,
        ),
        *(
            CountedText(
                "day-76",
                CUDA_LANGUAGE,
                f"tiled_matmul_t{tile}_kernel",
                f"ai_journey.gpu_tiled_cuda.tiled_cuda_source(tile={tile})",
                tiled_cuda_source(tile=tile),
            )
            for tile in TILES
        ),
        *(
            CountedText(
                "day-77",
                CUDA_LANGUAGE,
                kernel.kernel_name,
                f"ai_journey.gpu_checklist_cuda.cuda_source({kernel.name})",
                cuda_source(kernel),
            )
            for kernel in PINNED_KERNELS
        ),
        *(
            CountedText(
                "day-76",
                METAL_LANGUAGE,
                tiled_kernel_name(tile),
                f"ai_journey.matmul_tiled_source.tiled_kernel_source({tile})",
                tiled_kernel_source(tile),
            )
            for tile in TILE_SIZES
        ),
        *(
            CountedText(
                "day-77",
                METAL_LANGUAGE,
                design.name,
                f"ai_journey.matmul_checklist_source.block_kernel_source({design.name})",
                block_kernel_source(design),
            )
            for design in DESIGNS
        ),
    )


def loc_mapping() -> dict[str, object]:
    """Return the rule and the count of each text, as a report records them."""

    return {
        "rule": COUNTING_RULE,
        "texts": [text.to_mapping() for text in counted_texts()],
    }
