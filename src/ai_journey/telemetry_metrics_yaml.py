"""Emit and strictly read the YAML subset that the lab's rule files use.

PyYAML is not a dependency, so rule files use a subset small enough to read
back exactly. A document is a block mapping. Keys are plain identifiers.
Every scalar is a single-quoted string on one line, so no YAML 1.1 reader can
turn a value such as ``on`` or ``0s`` into a boolean, number or null. Every
sequence item is a block mapping whose first key shares the dash line, and
indentation is two spaces per level. ``#`` comment lines may appear only
before the document. Everything else is rejected: tabs, carriage returns,
blank lines, trailing spaces, flow collections, anchors, aliases, tags,
block scalars, plain and double-quoted scalars, document markers, inline
comments, duplicate keys and empty collections. As a result
``emit_yaml(read_yaml(text)) == text`` for every accepted document without
comments, and any YAML 1.2 reader gets the same nested strings back.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .telemetry_metrics_model import TelemetryMetricsError

MAX_YAML_BYTES = 256 * 1024
MAX_DEPTH = 16
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMMENT = re.compile(r"#(?: .*)?")


def _require_key(key: object) -> str:
    if type(key) is not str or _KEY.fullmatch(key) is None:
        raise TelemetryMetricsError(f"YAML keys must be plain identifiers: {key!r}")
    return key


def _quote(value: str) -> str:
    if not value.isprintable():
        raise TelemetryMetricsError("YAML scalars must be printable one-line text")
    return "'" + value.replace("'", "''") + "'"


def _emit_mapping(
    mapping: object, indent: int, lines: list[str], *, dash: bool = False
) -> None:
    if type(mapping) is not dict or not mapping:
        raise TelemetryMetricsError("YAML mappings must be non-empty dicts")
    if indent // 2 > MAX_DEPTH:
        raise TelemetryMetricsError(f"YAML nests deeper than {MAX_DEPTH} levels")
    for position, (key, value) in enumerate(mapping.items()):
        prefix = " " * (indent - 2) + "- " if dash and position == 0 else " " * indent
        head = prefix + _require_key(key) + ":"
        if type(value) is str:
            lines.append(f"{head} {_quote(value)}")
        elif type(value) is dict:
            lines.append(head)
            _emit_mapping(value, indent + 2, lines)
        elif type(value) is list:
            if not value:
                raise TelemetryMetricsError("YAML sequences must not be empty")
            lines.append(head)
            for item in value:
                _emit_mapping(item, indent + 4, lines, dash=True)
        else:
            raise TypeError("YAML values must be str, dict or list")


def emit_yaml(document: dict, *, comments: Sequence[str] = ()) -> str:
    """Render nested dicts, lists of dicts and strings as the subset.

    Optional comment lines come first; the document follows, every line
    ending with a line feed.
    """

    lines = []
    for comment in comments:
        if type(comment) is not str or not comment.isprintable():
            raise TelemetryMetricsError("YAML comments must be printable text")
        lines.append(f"# {comment}" if comment else "#")
    _emit_mapping(document, 0, lines)
    return "".join(line + "\n" for line in lines)


@dataclass(frozen=True)
class _Line:
    number: int
    indent: int
    dash: bool
    key: str
    value: str | None


def _unquote(text: str, where: str) -> str:
    inner = text[1:-1]
    if (
        len(text) < 2
        or text[0] != "'"
        or text[-1] != "'"
        or "'" in inner.replace("''", "")
    ):
        raise TelemetryMetricsError(f"{where}: values must be one single-quoted scalar")
    value = inner.replace("''", "'")
    if not value.isprintable():
        raise TelemetryMetricsError(f"{where}: values must be printable text")
    return value


def _tokenize(text: str) -> list[_Line]:
    if type(text) is not str:
        raise TypeError("YAML text must be str")
    if len(text.encode("utf-8", "surrogatepass")) > MAX_YAML_BYTES:
        raise TelemetryMetricsError(f"YAML exceeds {MAX_YAML_BYTES} bytes")
    if not text.endswith("\n"):
        raise TelemetryMetricsError("YAML must end with a line feed")
    lines: list[_Line] = []
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        where = f"line {number}"
        if any(character in line for character in ("\t", "\r", "\ufeff")):
            raise TelemetryMetricsError(f"{where}: tabs, CR and BOM are rejected")
        if line.startswith("#"):
            if lines or _COMMENT.fullmatch(line) is None:
                raise TelemetryMetricsError(
                    f"{where}: comments may only precede the document"
                )
            continue
        body = line.lstrip(" ")
        indent = len(line) - len(body)
        if not body or body.endswith(" ") or indent % 2:
            raise TelemetryMetricsError(
                f"{where}: blank lines, trailing spaces and odd indents are rejected"
            )
        dash = body.startswith("- ")
        if dash:
            body, indent = body[2:], indent + 2
        key, colon, rest = body.partition(":")
        if not colon or _KEY.fullmatch(key) is None:
            raise TelemetryMetricsError(f"{where}: expected a plain key and a colon")
        if rest and not rest.startswith(" "):
            raise TelemetryMetricsError(f"{where}: one space must follow the colon")
        value = _unquote(rest[1:], where) if rest else None
        lines.append(_Line(number, indent, dash, key, value))
    if not lines:
        raise TelemetryMetricsError("YAML holds no document")
    return lines


def _block(lines: list[_Line], index: int, parent: _Line, depth: int):
    """Read the nested block of a key that has no scalar on its own line."""

    if index >= len(lines) or depth > MAX_DEPTH:
        raise TelemetryMetricsError(
            f"line {parent.number}: {parent.key} needs a nested block"
        )
    child = lines[index]
    if child.dash and child.indent == parent.indent + 4:
        items = []
        while (
            index < len(lines)
            and lines[index].dash
            and lines[index].indent == child.indent
        ):
            item, index = _mapping(lines, index, child.indent, depth + 1)
            items.append(item)
        return items, index
    if not child.dash and child.indent == parent.indent + 2:
        return _mapping(lines, index, child.indent, depth + 1)
    raise TelemetryMetricsError(
        f"line {child.number}: nested blocks are indented by two spaces"
    )


def _mapping(lines: list[_Line], index: int, indent: int, depth: int):
    """Read keys at one indent; a dash may only open the first key."""

    result: dict[str, object] = {}
    first = True
    while index < len(lines):
        line = lines[index]
        if line.indent < indent or line.indent == indent and line.dash and not first:
            break
        if line.indent > indent or line.dash and not first:
            raise TelemetryMetricsError(f"line {line.number}: unexpected indentation")
        if line.key in result:
            raise TelemetryMetricsError(f"line {line.number}: duplicate key {line.key}")
        index += 1
        if line.value is None:
            result[line.key], index = _block(lines, index, line, depth)
        else:
            result[line.key] = line.value
        first = False
    return result, index


def read_yaml(text: str) -> dict:
    """Read the subset strictly into nested dicts, lists of dicts and strings."""

    lines = _tokenize(text)
    if lines[0].indent != 0 or lines[0].dash:
        raise TelemetryMetricsError("the YAML document must be a block mapping")
    document, index = _mapping(lines, 0, 0, 0)
    if index != len(lines):
        raise TelemetryMetricsError(
            f"line {lines[index].number}: unexpected content after the document"
        )
    return document
