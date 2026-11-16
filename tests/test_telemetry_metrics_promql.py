import math

import pytest

from ai_journey.telemetry_metrics_exposition import Sample
from ai_journey.telemetry_metrics_model import TelemetryMetricsError
from ai_journey.telemetry_metrics_promql import (
    Comparison,
    Element,
    Matcher,
    Ratio,
    Selector,
    SumBy,
    evaluate,
    parse_expression,
    render,
    selectors,
)

SCOPE = ("scope", "local_cpu_lab")
ATTEMPTS = Selector("lab_attempts_total")
OK = Selector("lab_attempts_total", (Matcher("outcome", "=", "ok"),))
OK_RATIO = Ratio(SumBy(("scope",), OK), SumBy(("scope",), ATTEMPTS))
SAMPLES = (
    Sample("lab_attempts_total", (("outcome", "ok"), SCOPE), 48.0),
    Sample("lab_attempts_total", (("outcome", "refused"), SCOPE), 6.0),
    Sample("lab_attempts_total", (("outcome", "reset"), SCOPE), 2.0),
    Sample("lab_detection_seconds", (SCOPE,), 0.25),
    Sample("lab_zero_total", (SCOPE,), 0.0),
    Sample("lab_bare", (), 3.0),
    Sample("lab_latency_seconds_bucket", (SCOPE, ("le", "0.25")), 7.0),
    Sample("scope:lab_latency:ratio", (SCOPE,), 0.5),
)


@pytest.mark.parametrize(
    ("expression", "text"),
    [
        (ATTEMPTS, "lab_attempts_total"),
        (OK, 'lab_attempts_total{outcome="ok"}'),
        (
            Selector(
                "x",
                (Matcher("a", "!=", 'q"\\'), Matcher("le", "=", "0.25")),
            ),
            'x{a!="q\\"\\\\",le="0.25"}',
        ),
        (SumBy(("scope",), OK), 'sum by (scope) (lab_attempts_total{outcome="ok"})'),
        (SumBy(("a", "b"), ATTEMPTS), "sum by (a, b) (lab_attempts_total)"),
        (
            OK_RATIO,
            'sum by (scope) (lab_attempts_total{outcome="ok"}) / '
            "sum by (scope) (lab_attempts_total)",
        ),
        (
            Comparison(OK_RATIO, "<", 0.95),
            'sum by (scope) (lab_attempts_total{outcome="ok"}) / '
            "sum by (scope) (lab_attempts_total) < 0.95",
        ),
        (
            Comparison(Selector("lab_detection_seconds"), ">", 0.25),
            "lab_detection_seconds > 0.25",
        ),
        (Comparison(Selector("a:b:c"), ">=", 1e-05), "a:b:c >= 1e-05"),
        (Comparison(ATTEMPTS, "==", -1.0), "lab_attempts_total == -1"),
        (Comparison(ATTEMPTS, "!=", 2391208.0), "lab_attempts_total != 2.391208e+06"),
        (Comparison(ATTEMPTS, "<=", 0.0), "lab_attempts_total <= 0"),
    ],
)
def test_expressions_render_canonically_and_parse_back(expression, text):
    assert render(expression) == text
    assert parse_expression(text) == expression


@pytest.mark.parametrize(
    "text",
    [
        "",
        " lab_x",
        "lab_x ",
        "lab_x>1",
        "lab_x  > 1",
        "lab_x > 1.0",
        "lab_x > 0.250",
        "lab_x > +1",
        "lab_x > inf",
        "lab_x > NaN",
        "lab_x > 1 > 2",
        "lab_x <> 1",
        "lab_x = 1",
        "lab_x > bool 1",
        "lab_x{}",
        'lab_x{a="b",}',
        'lab_x{a = "b"}',
        'lab_x{b="1",a="2"}',
        'lab_x{a="1",a="2"}',
        'lab_x{a=~"b"}',
        'lab_x{a!~"b"}',
        "lab_x{a='b'}",
        'lab_x{a="b\\n"}',
        'lab_x{a="b}',
        'lab_x{__name__="x"}',
        "sum(lab_x)",
        "sum by(scope) (lab_x)",
        "sum by (scope)(lab_x)",
        "sum by (b, a) (lab_x)",
        "sum by (a,b) (lab_x)",
        "sum without (scope) (lab_x)",
        "sum by () (lab_x)",
        "sum by (scope) (sum by (scope) (lab_x))",
        "sum by (scope) (lab_x) / sum by (other) (lab_y)",
        "lab_x / lab_y",
        "sum by (scope) (lab_x)/sum by (scope) (lab_y)",
        "rate(lab_x[5m])",
        "lab_x[5m]",
        "lab_x offset 5m",
        "sum",
        "SUM",
        "Inf",
        "lab_x\n",
        "9lab",
    ],
)
def test_parser_rejects_everything_but_the_canonical_subset(text):
    with pytest.raises(TelemetryMetricsError):
        parse_expression(text)


def test_nodes_validate_their_parts():
    for build in (
        lambda: Matcher("le", "=~", "x"),
        lambda: Matcher("__name__", "=", "x"),
        lambda: Matcher("a", "=", "line\nfeed"),
        lambda: Selector("rule:name", (Matcher("b", "=", ""), Matcher("a", "=", ""))),
        lambda: Selector("by"),
        lambda: Selector("x", [Matcher("a", "=", "b")]),
        lambda: SumBy((), ATTEMPTS),
        lambda: SumBy(("b", "a"), ATTEMPTS),
        lambda: SumBy(("scope",), "lab_attempts_total"),
        lambda: Ratio(SumBy(("a",), ATTEMPTS), SumBy(("b",), ATTEMPTS)),
        lambda: Ratio(ATTEMPTS, ATTEMPTS),
        lambda: Comparison(ATTEMPTS, "=~", 1.0),
        lambda: Comparison(ATTEMPTS, ">", 1),
        lambda: Comparison(ATTEMPTS, ">", math.inf),
        lambda: Comparison(Comparison(ATTEMPTS, ">", 1.0), ">", 1.0),
    ):
        with pytest.raises(TelemetryMetricsError):
            build()
    with pytest.raises(TypeError):
        render("lab_x")
    with pytest.raises(TypeError):
        parse_expression(b"lab_x")


def test_selectors_lists_every_read_metric_in_order():
    comparison = Comparison(OK_RATIO, "<", 0.5)

    assert selectors(comparison) == (OK, ATTEMPTS)
    assert selectors(SumBy(("scope",), OK)) == (OK,)
    assert selectors(ATTEMPTS) == (ATTEMPTS,)


def test_selector_and_sum_follow_promql_instant_semantics():
    assert evaluate(OK, SAMPLES) == (Element((("outcome", "ok"), SCOPE), 48.0),)
    not_ok = Selector("lab_attempts_total", (Matcher("outcome", "!=", "ok"),))
    assert [e.value for e in evaluate(not_ok, SAMPLES)] == [6.0, 2.0]
    absent = Selector("lab_bare", (Matcher("scope", "=", ""),))
    assert evaluate(absent, SAMPLES) == (Element((), 3.0),)
    bucket = Selector("lab_latency_seconds_bucket", (Matcher("le", "=", "0.25"),))
    assert evaluate(bucket, SAMPLES) == (Element((("le", "0.25"), SCOPE), 7.0),)
    assert evaluate(SumBy(("scope",), ATTEMPTS), SAMPLES) == (Element((SCOPE,), 56.0),)
    assert evaluate(SumBy(("scope",), Selector("lab_bare")), SAMPLES) == (
        Element((), 3.0),
    )
    assert evaluate(SumBy(("scope",), Selector("lab_missing")), SAMPLES) == ()
    assert evaluate(Selector("scope:lab_latency:ratio"), SAMPLES)[0].value == 0.5


def test_ratios_divide_like_ieee_754_and_comparisons_filter():
    assert evaluate(OK_RATIO, SAMPLES) == (Element((SCOPE,), 48 / 56),)
    zero = SumBy(("scope",), Selector("lab_zero_total"))
    one = SumBy(("scope",), Selector("lab_detection_seconds"))
    nan = evaluate(Ratio(zero, zero), SAMPLES)[0].value
    assert math.isnan(nan)
    assert evaluate(Ratio(one, zero), SAMPLES)[0].value == math.inf
    missing = SumBy(("scope",), Selector("lab_missing"))
    assert evaluate(Ratio(one, missing), SAMPLES) == ()

    assert evaluate(Comparison(OK_RATIO, "<", 0.9), SAMPLES) == (
        Element((SCOPE,), 48 / 56),
    )
    assert evaluate(Comparison(OK_RATIO, ">=", 0.9), SAMPLES) == ()
    detection = Selector("lab_detection_seconds")
    assert evaluate(Comparison(detection, ">", 0.25), SAMPLES) == ()
    assert len(evaluate(Comparison(detection, ">=", 0.25), SAMPLES)) == 1
    assert evaluate(Comparison(Ratio(zero, zero), "<", 1.0), SAMPLES) == ()
    assert evaluate(Comparison(Ratio(zero, zero), ">", 1.0), SAMPLES) == ()
    assert len(evaluate(Comparison(Ratio(zero, zero), "!=", 1.0), SAMPLES)) == 1
    with pytest.raises(TypeError):
        evaluate(OK, [("lab_attempts_total", (), 1.0)])
