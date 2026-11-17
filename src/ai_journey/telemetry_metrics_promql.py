"""The PromQL subset that the lab's SLO rules are written in.

Rules are built as expression trees, rendered in one canonical PromQL
spelling, parsed back strictly and evaluated at one instant against the
samples of an exposition. The subset covers what the declared objectives
need and nothing more:

* an instant-vector selector ``name{label="value",label!="value"}``;
* ``sum by (label, ...) (selector)``;
* a ratio of two such sums with the same grouping labels;
* a comparison of any of these with a number, which filters the vector as
  PromQL does instead of returning booleans.

Evaluation follows PromQL instant-query semantics on that subset: a matcher
on an absent label compares with the empty string, ``sum`` over no series is
no series rather than 0, division by zero gives ``+Inf``, ``-Inf`` or
``NaN`` as IEEE 754 does, and a comparison with ``NaN`` holds only for
``!=``. Results never carry a metric name. PromQL keeps the name on a
selector, alone or compared with a number, but an alert drops it and a
recording rule replaces it, so no rule outcome differs. Range vectors,
``rate`` and every other function are deliberately absent: the exposition is
one sealed snapshot of a finished run, so each counter already covers the
whole run.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .telemetry_metrics_exposition import Sample, format_value
from .telemetry_metrics_model import TelemetryMetricsError

_METRIC = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_NUMBER = re.compile(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
# Names PromQL reads as keywords, aggregations or functions in this position.
_RESERVED = frozenset(
    {
        "and",
        "avg",
        "bool",
        "by",
        "count",
        "group_left",
        "group_right",
        "ignoring",
        "inf",
        "max",
        "min",
        "nan",
        "offset",
        "on",
        "or",
        "sum",
        "unless",
        "without",
    }
)
MATCH_OPERATORS = ("=", "!=")
COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def _require_label(name: object) -> str:
    if type(name) is not str or _LABEL.fullmatch(name) is None or name[:2] == "__":
        raise TelemetryMetricsError(f"invalid PromQL label name: {name!r}")
    return name


@dataclass(frozen=True)
class Matcher:
    """One equality or inequality label matcher."""

    label: str
    op: str
    value: str

    def __post_init__(self) -> None:
        _require_label(self.label)
        if self.op not in MATCH_OPERATORS:
            raise TelemetryMetricsError("matchers use = or != only")
        if type(self.value) is not str or not self.value.isprintable():
            raise TelemetryMetricsError("matcher values must be printable text")

    def matches(self, labels: dict[str, str]) -> bool:
        equal = labels.get(self.label, "") == self.value
        return equal if self.op == "=" else not equal


@dataclass(frozen=True)
class Selector:
    """An instant-vector selector with matchers sorted by label name."""

    metric: str
    matchers: tuple[Matcher, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.metric) is not str
            or _METRIC.fullmatch(self.metric) is None
            or self.metric.lower() in _RESERVED
        ):
            raise TelemetryMetricsError(f"invalid PromQL metric name: {self.metric!r}")
        if type(self.matchers) is not tuple or any(
            type(matcher) is not Matcher for matcher in self.matchers
        ):
            raise TelemetryMetricsError("selector matchers must be Matcher values")
        labels = [matcher.label for matcher in self.matchers]
        if labels != sorted(set(labels)):
            raise TelemetryMetricsError("matchers must name distinct sorted labels")


@dataclass(frozen=True)
class SumBy:
    """``sum by (labels) (selector)`` with sorted, distinct grouping labels."""

    labels: tuple[str, ...]
    selector: Selector

    def __post_init__(self) -> None:
        if (
            type(self.labels) is not tuple
            or not self.labels
            or list(self.labels) != sorted(set(self.labels))
        ):
            raise TelemetryMetricsError("sum needs distinct sorted grouping labels")
        for label in self.labels:
            _require_label(label)
        if type(self.selector) is not Selector:
            raise TelemetryMetricsError("sum aggregates one Selector")


@dataclass(frozen=True)
class Ratio:
    """One sum divided by another sum over the same grouping labels."""

    numerator: SumBy
    denominator: SumBy

    def __post_init__(self) -> None:
        if type(self.numerator) is not SumBy or type(self.denominator) is not SumBy:
            raise TelemetryMetricsError("a ratio divides two SumBy values")
        if self.numerator.labels != self.denominator.labels:
            raise TelemetryMetricsError("a ratio's sums must group by the same labels")


@dataclass(frozen=True)
class Comparison:
    """A vector filtered by comparing each value with a finite threshold."""

    operand: Selector | SumBy | Ratio
    op: str
    threshold: float

    def __post_init__(self) -> None:
        if type(self.operand) not in (Selector, SumBy, Ratio):
            raise TelemetryMetricsError("a comparison filters a selector, sum or ratio")
        if self.op not in COMPARISONS:
            raise TelemetryMetricsError(f"unsupported comparison {self.op!r}")
        if type(self.threshold) is not float or not math.isfinite(self.threshold):
            raise TelemetryMetricsError("a comparison threshold is a finite float")


Expression = Selector | SumBy | Ratio | Comparison


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(expression: Expression) -> str:
    """Spell an expression the one way ``parse_expression`` accepts."""

    if type(expression) is Selector:
        matchers = ",".join(
            f"{matcher.label}{matcher.op}{_quote(matcher.value)}"
            for matcher in expression.matchers
        )
        return expression.metric + ("{" + matchers + "}" if matchers else "")
    if type(expression) is SumBy:
        labels = ", ".join(expression.labels)
        return f"sum by ({labels}) ({render(expression.selector)})"
    if type(expression) is Ratio:
        return f"{render(expression.numerator)} / {render(expression.denominator)}"
    if type(expression) is Comparison:
        threshold = format_value(expression.threshold)
        return f"{render(expression.operand)} {expression.op} {threshold}"
    raise TypeError("expression must be Selector, SumBy, Ratio or Comparison")


class _Reader:
    def __init__(self, text: str) -> None:
        self.text, self.index = text, 0

    def fail(self, message: str) -> TelemetryMetricsError:
        return TelemetryMetricsError(f"PromQL at offset {self.index}: {message}")

    def take(self, literal: str) -> bool:
        if self.text.startswith(literal, self.index):
            self.index += len(literal)
            return True
        return False

    def expect(self, literal: str) -> None:
        if not self.take(literal):
            raise self.fail(f"expected {literal!r}")

    def pattern(self, regex: re.Pattern[str], what: str) -> str:
        match = regex.match(self.text, self.index)
        if match is None:
            raise self.fail(f"expected {what}")
        self.index = match.end()
        return match.group()

    def string(self) -> str:
        self.expect('"')
        characters = []
        while not self.take('"'):
            if self.index >= len(self.text):
                raise self.fail("unterminated string")
            character = self.text[self.index]
            if character == "\\":
                following = self.text[self.index + 1 : self.index + 2]
                if following not in ('"', "\\"):
                    raise self.fail('strings escape only \\ and "')
                character, self.index = following, self.index + 1
            characters.append(character)
            self.index += 1
        return "".join(characters)

    def selector(self) -> Selector:
        metric = self.pattern(_METRIC, "a metric name")
        matchers = []
        if self.take("{"):
            while True:
                label = self.pattern(_LABEL, "a label name")
                op = "!=" if self.take("!=") else "="
                if op == "=":
                    self.expect("=")
                matchers.append(Matcher(label, op, self.string()))
                if self.take("}"):
                    break
                self.expect(",")
        return Selector(metric, tuple(matchers))

    def sum(self) -> SumBy:
        self.expect("sum by (")
        labels = [self.pattern(_LABEL, "a grouping label")]
        while self.take(", "):
            labels.append(self.pattern(_LABEL, "a grouping label"))
        self.expect(") (")
        selector = self.selector()
        self.expect(")")
        return SumBy(tuple(labels), selector)

    def operand(self) -> Selector | SumBy | Ratio:
        if not self.text.startswith("sum by (", self.index):
            return self.selector()
        numerator = self.sum()
        if self.take(" / "):
            return Ratio(numerator, self.sum())
        return numerator

    def expression(self) -> Expression:
        operand = self.operand()
        if self.index == len(self.text):
            return operand
        self.expect(" ")
        op = self.pattern(re.compile(r"[<>!=]=?"), "a comparison")
        if op not in COMPARISONS:
            raise self.fail(f"unsupported comparison {op!r}")
        self.expect(" ")
        threshold = float(self.pattern(_NUMBER, "a number"))
        if self.index != len(self.text):
            raise self.fail("unexpected trailing text")
        return Comparison(operand, op, threshold)


def parse_expression(text: str) -> Expression:
    """Parse one canonical subset expression; any other spelling is rejected."""

    if type(text) is not str:
        raise TypeError("PromQL text must be str")
    if not text.isprintable():
        raise TelemetryMetricsError("PromQL text must be printable one-line text")
    expression = _Reader(text).expression()
    if render(expression) != text:
        raise TelemetryMetricsError(f"PromQL is not in canonical form: {text!r}")
    return expression


def selectors(expression: Expression) -> tuple[Selector, ...]:
    """Every selector an expression reads, left to right."""

    if type(expression) is Selector:
        return (expression,)
    if type(expression) is SumBy:
        return (expression.selector,)
    if type(expression) is Ratio:
        return (expression.numerator.selector, expression.denominator.selector)
    if type(expression) is Comparison:
        return selectors(expression.operand)
    raise TypeError("expression must be Selector, SumBy, Ratio or Comparison")


@dataclass(frozen=True)
class Element:
    """One instant-vector element: name-free sorted labels and a value."""

    labels: tuple[tuple[str, str], ...]
    value: float


def _divide(numerator: float, denominator: float) -> float:
    if denominator != 0:
        return numerator / denominator
    if numerator == 0 or math.isnan(numerator):
        return math.nan
    return math.copysign(math.inf, numerator) * math.copysign(1.0, denominator)


def _vector(expression: Expression, samples: Sequence[Sample]) -> list[Element]:
    if type(expression) is Selector:
        return [
            Element(tuple(sorted(sample.labels)), sample.value)
            for sample in samples
            if sample.name == expression.metric
            and all(m.matches(sample.label_map()) for m in expression.matchers)
        ]
    if type(expression) is SumBy:
        groups: dict[tuple[tuple[str, str], ...], list[float]] = {}
        for element in _vector(expression.selector, samples):
            labels = dict(element.labels)
            key = tuple(
                (name, labels[name]) for name in expression.labels if labels.get(name)
            )
            groups.setdefault(key, []).append(element.value)
        return [Element(key, math.fsum(values)) for key, values in groups.items()]
    if type(expression) is Ratio:
        denominators = {
            element.labels: element.value
            for element in _vector(expression.denominator, samples)
        }
        return [
            Element(
                element.labels, _divide(element.value, denominators[element.labels])
            )
            for element in _vector(expression.numerator, samples)
            if element.labels in denominators
        ]
    if type(expression) is Comparison:
        compare = COMPARISONS[expression.op]
        return [
            element
            for element in _vector(expression.operand, samples)
            if compare(element.value, expression.threshold)
        ]
    raise TypeError("expression must be Selector, SumBy, Ratio or Comparison")


def evaluate(expression: Expression, samples: Sequence[Sample]) -> tuple[Element, ...]:
    """Evaluate at one instant over exposition samples; results sort by labels."""

    if any(type(sample) is not Sample for sample in samples):
        raise TypeError("samples must be exposition Sample values")
    return tuple(
        sorted(_vector(expression, samples), key=lambda element: element.labels)
    )
