import pytest

from ai_journey.telemetry_dashboard_promql import (
    AGGREGATIONS,
    FUNCTIONS,
    PromqlSummary,
    PromqlSyntaxError,
    check_promql,
)
from ai_journey.telemetry_traces_io import TelemetryVerificationError

LATENCY = (
    "histogram_quantile(0.99, sum by (le) "
    "(rate(ai_journey_serving_request_latency_seconds_bucket[$__rate_interval])))"
)


@pytest.mark.parametrize(
    "expression",
    [
        "sum by (outcome) (ai_journey_serving_attempts_total)",
        LATENCY,
        'sum(rate(http_total{job="api", outcome!~"ok|timeout"}[5m])) by (job)',
        "max_over_time(up[1h:5m])",
        "max_over_time(up[1h:])",
        "up offset 5m",
        "up offset -1h30m",
        "a / on(job) group_left b",
        "a * ignoring (instance) group_right(job) b",
        'label_replace(up, "dst", "$1", "src", "(.*)")',
        "-up",
        "up > bool 0.5",
        "time() - 1e3 * up",
        "up @ end()",
        "up @ 1609746000",
        "up{}",
        '{__name__="up"}',
        "count without (instance) (up) or vector(0)",
        "holt_winters(up[10m], 0.5, 0.5)",
        "double_exponential_smoothing(up[10m], 0.5, 0.5)",
        "up > Inf",
        "  sum(\n  up\n)  ",
    ],
)
def test_well_formed_expressions_pass(expression):
    assert isinstance(check_promql(expression), PromqlSummary)


def test_summaries_name_the_referenced_series_and_functions():
    summary = check_promql(LATENCY)

    assert summary.metrics == ("ai_journey_serving_request_latency_seconds_bucket",)
    assert summary.functions == ("histogram_quantile", "rate", "sum")
    labels = check_promql('sum by (job, le) (a{outcome="ok"}) / ignoring(le) b')
    assert labels.metrics == ("a", "b")


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("sum(rate(up[5m])", r"unclosed '\(' opened at 3"),
        ("sum(up))", r"unbalanced '\)' at 7"),
        ("sum(up]", r"unbalanced '\]'"),
        ("up{job='a')", r"unbalanced '\)'"),
        ("rates(up[5m])", "unknown PromQL function 'rates'"),
        ("histogram_quantiles(0.9, up)", "unknown PromQL function"),
        ("sum", "aggregation 'sum' needs an argument"),
        ("sum by (le)", "aggregation 'sum' needs an argument"),
        ("sum by le (up)", "'by' needs a label list"),
        ("sum(up) by le", "'by' needs a label list"),
        ("sum by (le, 0.5) (up)", "unexpected number"),
        ("sum by (le=) (up)", "malformed label list"),
        ("sum by (le (x)) (up)", "needs an argument"),
        ("sum by ({x}) (up)", r"unexpected '\{'"),
        ('up{job="a}', "unterminated string"),
        ("up # comment", "unexpected character '#'"),
        ("up + 5m", "duration outside a range"),
        ("up[]", "range needs a duration"),
        ("up[job]", "range needs a duration"),
        ("up[5m:1m:1m]", "malformed range"),
        ("up[:5m]", "malformed range"),
        ("5[5m]", "range must follow a selector"),
        ("[5m]", "range must follow a selector"),
        ("up{job}", "incomplete label matcher"),
        ("up{job=}", "incomplete label matcher"),
        ('up{job>"a"}', "malformed label matcher"),
        ('up{job="a" env="b"}', "unexpected label name"),
        ('up{job="a",,}', "malformed label matcher"),
        ("up{job={}}", r"unexpected '\{'"),
        ("up, down", "',' outside a call"),
        ("(up, down)", "',' outside a call"),
        ("up offset", "duration outside|offset"),
        ("up + $__range", "variable outside a range"),
        ("job:up:rate5m + up : 1", "':' outside a range"),
        ("start()", "unknown PromQL function 'start'"),
        ("up @ end", "'end' must be called"),
        ("", "non-empty"),
        ("   ", "non-empty"),
        ("up" * 3000, "exceeds"),
    ],
)
def test_malformed_expressions_are_rejected_with_a_reason(expression, message):
    with pytest.raises(PromqlSyntaxError, match=message):
        check_promql(expression)


def test_allowed_series_restrict_metric_names_but_not_labels_or_functions():
    allowed = {"ai_journey_serving_attempts_total"}

    check_promql(
        'sum by (outcome) (ai_journey_serving_attempts_total{outcome="ok"})',
        allowed_metrics=allowed,
    )
    with pytest.raises(PromqlSyntaxError, match="unknown metric 'ai_journey_typo'"):
        check_promql("sum(ai_journey_typo)", allowed_metrics=allowed)


@pytest.mark.parametrize(
    "expression",
    [
        '{__name__="node_cpu_seconds_total"}',
        'sum({__name__=~".+"})',
        'sum by (outcome) ({outcome="ok"})',
        'ai_journey_serving_attempts_total and {job="other"}',
    ],
)
def test_selectors_without_a_metric_name_cannot_escape_the_allowed_series(
    expression,
):
    check_promql(expression)
    with pytest.raises(PromqlSyntaxError, match="must name an allowed metric"):
        check_promql(expression, allowed_metrics={"ai_journey_serving_attempts_total"})


def test_a_selector_cannot_set_its_metric_name_twice():
    # Prometheus rejects this as "metric name must not be set twice".
    for allowed in (None, {"up"}):
        with pytest.raises(PromqlSyntaxError, match="metric name twice"):
            check_promql('sum(up{__name__="node_load1"})', allowed_metrics=allowed)


def test_known_names_cover_query_functions_and_aggregations():
    assert {"rate", "increase", "histogram_quantile", "label_replace"} <= FUNCTIONS
    assert {"sum", "max", "topk", "count_values", "quantile"} <= AGGREGATIONS
    assert not FUNCTIONS & AGGREGATIONS
    assert issubclass(PromqlSyntaxError, TelemetryVerificationError)
    with pytest.raises(PromqlSyntaxError, match="non-empty string"):
        check_promql(None)
