"""PromQL syntax checks at the level of brackets, function names and selectors.

This is not a PromQL parser and does not check types, arity or precedence. It
tokenizes an expression and checks what a dashboard generator gets wrong most
often: parentheses, braces and brackets balance and nest; string literals end;
every name called like a function is a known PromQL function or aggregation
operator; ``by``, ``without``, ``on`` and ``ignoring`` take a parenthesized
label list; braces hold ``label op "value"`` matchers; range brackets follow a
selector and hold a duration or a Grafana interval variable, with an optional
subquery step; durations appear only in ranges or after ``offset``; and a
selector that names its metric does not match ``__name__`` as well, which
Prometheus rejects. When the caller passes the allowed series names, every
selector must name its metric before any braces and every referenced metric
must be one of them, so ``{__name__=~".+"}`` cannot select around the list.

The function list covers the Prometheus 2.x and 3.x query functions, including
both ``holt_winters`` and its 3.0 name ``double_exponential_smoothing``.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from .telemetry_traces_io import TelemetryVerificationError

FUNCTIONS = frozenset(
    {
        "abs",
        "absent",
        "absent_over_time",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "avg_over_time",
        "ceil",
        "changes",
        "clamp",
        "clamp_max",
        "clamp_min",
        "cos",
        "cosh",
        "count_over_time",
        "day_of_month",
        "day_of_week",
        "day_of_year",
        "days_in_month",
        "deg",
        "delta",
        "deriv",
        "double_exponential_smoothing",
        "exp",
        "floor",
        "histogram_avg",
        "histogram_count",
        "histogram_fraction",
        "histogram_quantile",
        "histogram_stddev",
        "histogram_stdvar",
        "histogram_sum",
        "holt_winters",
        "hour",
        "idelta",
        "increase",
        "irate",
        "label_join",
        "label_replace",
        "last_over_time",
        "ln",
        "log10",
        "log2",
        "mad_over_time",
        "max_over_time",
        "min_over_time",
        "minute",
        "month",
        "pi",
        "predict_linear",
        "present_over_time",
        "quantile_over_time",
        "rad",
        "rate",
        "resets",
        "round",
        "scalar",
        "sgn",
        "sin",
        "sinh",
        "sort",
        "sort_by_label",
        "sort_by_label_desc",
        "sort_desc",
        "sqrt",
        "stddev_over_time",
        "stdvar_over_time",
        "sum_over_time",
        "tan",
        "tanh",
        "time",
        "timestamp",
        "vector",
        "year",
    }
)
AGGREGATIONS = frozenset(
    {
        "avg",
        "bottomk",
        "count",
        "count_values",
        "group",
        "limit_ratio",
        "limitk",
        "max",
        "min",
        "quantile",
        "stddev",
        "stdvar",
        "sum",
        "topk",
    }
)
_LABEL_LISTS = frozenset({"by", "without", "on", "ignoring"})
_OPTIONAL_LABEL_LISTS = frozenset({"group_left", "group_right"})
_KEYWORDS = frozenset({"and", "or", "unless", "atan2", "bool", "offset"})
_NUMBER_WORDS = frozenset({"inf", "nan"})
_MATCHERS = frozenset({"=", "!=", "=~", "!~"})
_MAX_EXPRESSION = 4096
_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<string>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`[^`]*`)
  | (?P<duration>(?:[0-9]+(?:ms|[smhdwy]))+)
  | (?P<number>0[xX][0-9a-fA-F]+|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)
  | (?P<variable>\$__[A-Za-z_]+|\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*)
  | (?P<identifier>[A-Za-z_][A-Za-z0-9_:]*)
  | (?P<operator>==|!=|=~|!~|<=|>=|[-+*/%^<>=,@:])
  | (?P<bracket>[(){}\[\]])
    """,
    re.VERBOSE,
)
_PAIRS = {")": "(", "}": "{", "]": "["}


class PromqlSyntaxError(TelemetryVerificationError):
    """Raised when an expression fails the PromQL syntax checks."""


@dataclass(frozen=True)
class PromqlSummary:
    """Metric series and functions an accepted expression references."""

    metrics: tuple[str, ...]
    functions: tuple[str, ...]


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    offset: int


def _tokens(expression: str) -> list[_Token]:
    tokens = []
    offset = 0
    while offset < len(expression):
        match = _TOKEN.match(expression, offset)
        if match is None:
            character = expression[offset]
            if character in "\"'`":
                raise PromqlSyntaxError(f"unterminated string literal at {offset}")
            raise PromqlSyntaxError(f"unexpected character {character!r} at {offset}")
        if match.lastgroup != "space":
            tokens.append(_Token(match.lastgroup, match.group(), offset))
        offset = match.end()
    return tokens


class _Checker:
    """One pass over the tokens with a stack of bracket contexts."""

    def __init__(self, tokens: list[_Token], allowed: Collection[str] | None):
        self.tokens = tokens
        self.allowed = allowed
        self.stack: list[tuple[str, int, str]] = []
        self.metrics: set[str] = set()
        self.functions: set[str] = set()
        # Indices of "(" that open an aggregation argument after "by (...)".
        self.argument_openers: set[int] = set()
        self.matcher_state = "name"
        self.selector_named = False
        self.range_state = "start"

    def context(self) -> str:
        return self.stack[-1][2] if self.stack else "expression"

    def at(self, index: int) -> _Token | None:
        return self.tokens[index] if 0 <= index < len(self.tokens) else None

    def run(self) -> PromqlSummary:
        if not self.tokens:
            raise PromqlSyntaxError("expression is empty")
        for index, token in enumerate(self.tokens):
            getattr(self, f"_{token.kind}")(token, index)
        if self.stack:
            bracket, offset, _ = self.stack[-1]
            raise PromqlSyntaxError(f"unclosed {bracket!r} opened at {offset}")
        return PromqlSummary(tuple(sorted(self.metrics)), tuple(sorted(self.functions)))

    def _argument_after_modifier(self, index: int) -> int:
        """Return the "(" opening ``agg by (labels) (argument)``'s argument."""

        name, modifier = self.tokens[index].text, self.tokens[index + 1]
        opener = self.at(index + 2)
        if opener is None or opener.text != "(":
            raise PromqlSyntaxError(f"{modifier.text!r} needs a label list")
        close = next(
            (
                position
                for position in range(index + 3, len(self.tokens))
                if self.tokens[position].text == ")"
            ),
            None,
        )
        opener = None if close is None else self.at(close + 1)
        if opener is None or opener.text != "(":
            raise PromqlSyntaxError(f"aggregation {name!r} needs an argument")
        return close + 1

    def _bracket(self, token: _Token, index: int) -> None:
        text, previous = token.text, self.at(index - 1)
        if text in _PAIRS:
            if not self.stack or self.stack[-1][0] != _PAIRS[text]:
                raise PromqlSyntaxError(f"unbalanced {text!r} at {token.offset}")
            _, _, context = self.stack.pop()
            if context == "matchers" and self.matcher_state not in (
                "name",
                "separator",
            ):
                raise PromqlSyntaxError(f"incomplete label matcher at {token.offset}")
            if context == "range" and self.range_state == "start":
                raise PromqlSyntaxError(f"range needs a duration at {token.offset}")
            return
        if self.context() in ("labels", "matchers", "range"):
            raise PromqlSyntaxError(f"unexpected {text!r} at {token.offset}")
        name = previous.text if previous is not None else ""
        if text == "{":
            named = previous is not None and previous.kind == "identifier"
            named = named and name in self.metrics
            # Matchers alone, such as {__name__=~".+"}, select any metric.
            if not named and self.allowed is not None:
                raise PromqlSyntaxError(
                    f"selector at {token.offset} must name an allowed metric"
                )
            context, self.matcher_state = "matchers", "name"
            self.selector_named = named
        elif text == "[":
            if previous is None or not (
                previous.kind == "identifier" or name in ("}", ")")
            ):
                raise PromqlSyntaxError(
                    f"range must follow a selector at {token.offset}"
                )
            context, self.range_state = "range", "start"
        elif index in self.argument_openers:
            context = "call"
        elif name in _LABEL_LISTS or name in _OPTIONAL_LABEL_LISTS:
            context = "labels"
        elif (
            previous is not None
            and previous.kind == "identifier"
            and (name in FUNCTIONS or name in AGGREGATIONS or name in ("start", "end"))
        ):
            context = "call"
        else:
            context = "paren"
        self.stack.append((text, token.offset, context))

    def _identifier(self, token: _Token, index: int) -> None:
        context, name = self.context(), token.text
        following, previous = self.at(index + 1), self.at(index - 1)
        calls = following is not None and following.text == "("
        if context == "labels":
            return
        if context == "matchers":
            if self.matcher_state != "name":
                raise PromqlSyntaxError(f"unexpected label name at {token.offset}")
            if name == "__name__" and self.selector_named:
                raise PromqlSyntaxError(
                    f"selector sets its metric name twice at {token.offset}"
                )
            self.matcher_state = "operator"
            return
        if context == "range":
            raise PromqlSyntaxError(f"range needs a duration at {token.offset}")
        if name in _LABEL_LISTS and not calls:
            raise PromqlSyntaxError(f"{name!r} needs a label list at {token.offset}")
        if name in _LABEL_LISTS or name in _OPTIONAL_LABEL_LISTS:
            return
        if name == "offset":
            duration = (
                self.at(index + 2) if following and following.text == "-" else following
            )
            if duration is None or duration.kind != "duration":
                raise PromqlSyntaxError(f"offset needs a duration at {token.offset}")
            return
        if name.lower() in _NUMBER_WORDS or name in _KEYWORDS:
            return
        if name in ("start", "end") and previous is not None and previous.text == "@":
            if not calls:
                raise PromqlSyntaxError(f"{name!r} must be called at {token.offset}")
            return
        if name in AGGREGATIONS:
            self.functions.add(name)
            if following is not None and following.text in ("by", "without"):
                self.argument_openers.add(self._argument_after_modifier(index))
            elif not calls:
                raise PromqlSyntaxError(f"aggregation {name!r} needs an argument")
            return
        if calls:
            if name not in FUNCTIONS:
                raise PromqlSyntaxError(f"unknown PromQL function {name!r}")
            self.functions.add(name)
            return
        if self.allowed is not None and name not in self.allowed:
            raise PromqlSyntaxError(f"unknown metric {name!r}")
        self.metrics.add(name)

    def _string(self, token: _Token, index: int) -> None:
        context = self.context()
        if context == "matchers" and self.matcher_state == "value":
            self.matcher_state = "separator"
        elif context in ("matchers", "labels", "range"):
            raise PromqlSyntaxError(f"unexpected string at {token.offset}")

    def _duration(self, token: _Token, index: int) -> None:
        if self.context() == "range":
            self._range_value(token)
            return
        before = self.at(index - 1)
        if before is not None and before.text == "-":
            before = self.at(index - 2)
        if before is None or before.text != "offset":
            raise PromqlSyntaxError(f"duration outside a range at {token.offset}")

    def _variable(self, token: _Token, index: int) -> None:
        if self.context() != "range":
            raise PromqlSyntaxError(f"variable outside a range at {token.offset}")
        self._range_value(token)

    def _range_value(self, token: _Token) -> None:
        if self.range_state not in ("start", "colon"):
            raise PromqlSyntaxError(f"unexpected duration at {token.offset}")
        self.range_state = "duration" if self.range_state == "start" else "step"

    def _number(self, token: _Token, index: int) -> None:
        if self.context() in ("labels", "matchers", "range"):
            raise PromqlSyntaxError(f"unexpected number at {token.offset}")

    def _operator(self, token: _Token, index: int) -> None:
        context, text = self.context(), token.text
        if context == "matchers":
            if text == "," and self.matcher_state == "separator":
                self.matcher_state = "name"
            elif text in _MATCHERS and self.matcher_state == "operator":
                self.matcher_state = "value"
            else:
                raise PromqlSyntaxError(f"malformed label matcher at {token.offset}")
        elif context == "range":
            if text != ":" or self.range_state != "duration":
                raise PromqlSyntaxError(f"malformed range at {token.offset}")
            self.range_state = "colon"
        elif context == "labels":
            if text != ",":
                raise PromqlSyntaxError(f"malformed label list at {token.offset}")
        elif text == ":":
            raise PromqlSyntaxError(f"':' outside a range at {token.offset}")
        elif text == "," and context != "call":
            raise PromqlSyntaxError(
                f"',' outside a call or label list at {token.offset}"
            )


def check_promql(
    expression: str, *, allowed_metrics: Collection[str] | None = None
) -> PromqlSummary:
    """Check one expression and return the metrics and functions it uses."""

    if type(expression) is not str or not expression.strip():
        raise PromqlSyntaxError("expression must be a non-empty string")
    if len(expression) > _MAX_EXPRESSION:
        raise PromqlSyntaxError(f"expression exceeds {_MAX_EXPRESSION} characters")
    return _Checker(_tokens(expression), allowed_metrics).run()
