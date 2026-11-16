import pytest
from test_telemetry_metrics_config import valid_config
from test_telemetry_metrics_serving import make_serving_run
from test_telemetry_metrics_worker import make_worker_run

from ai_journey.telemetry_metrics_model import (
    HistogramValue,
    MetricFamily,
    MetricSet,
    Series,
    TelemetryMetricsError,
    metric_set,
)
from ai_journey.telemetry_metrics_promql import Comparison, Matcher, Selector, SumBy
from ai_journey.telemetry_metrics_rules import (
    HEADER,
    POST_FAULT_LATENCY_RATIO,
    AlertingRule,
    RecordingRule,
    RuleFile,
    RuleGroup,
    build_rule_file,
    declared_objectives,
    evaluate_rule_file,
    parse_rule_file,
    render_rule_file,
    validate_rule_file,
)
from ai_journey.telemetry_metrics_serving import (
    load_verified_serving_evidence,
    serving_recovery_families,
)
from ai_journey.telemetry_metrics_worker import (
    load_verified_worker_evidence,
    worker_recovery_families,
)

RELAXED = {
    "serving_objectives": {
        "detection_latency_ms": 300_000,
        "restart_latency_ms": 300_000,
        "unavailability_ms": 300_000,
        "time_to_slo_recovery_ms": 300_000,
        "post_fault_latency_us": 1_000_000,
        "post_fault_within_latency_ratio": 0.5,
    },
    "worker_objectives": {"detection_ms": 300_000, "recovery_ms": 300_000},
}
LABELS = (("scope", "local_cpu_lab"),)
RECOVERED = SumBy(("scope",), Selector("ai_journey_serving_recovery_recovered"))


@pytest.fixture(scope="module")
def lab_metrics(tmp_path_factory):
    base = tmp_path_factory.mktemp("telemetry-rules")
    serving = load_verified_serving_evidence(make_serving_run(base))
    worker = load_verified_worker_evidence(make_worker_run(base))
    buckets = valid_config().latency_buckets_us
    return metric_set(
        serving_recovery_families(serving, latency_buckets_us=buckets)
        + worker_recovery_families(worker)
    )


def replace_values(metrics, name, change):
    families = []
    for family in metrics.families:
        if family.name == name:
            series = tuple(Series(item.labels, change(item)) for item in family.series)
            family = MetricFamily(
                family.name, family.kind, family.help, series, family.buckets
            )
        families.append(family)
    return MetricSet(tuple(families))


def test_rules_follow_the_declared_objectives():
    rule_file = build_rule_file(valid_config())
    serving, worker = rule_file.groups
    alerts = {
        rule.alert: rule for rule in rule_file.rules if type(rule) is AlertingRule
    }

    assert (serving.name, worker.name) == (
        "ai_journey_serving_recovery_slo",
        "ai_journey_worker_recovery_slo",
    )
    assert [type(rule) for rule in serving.rules[:3]] == [RecordingRule] * 3
    assert len(alerts) == 11
    assert [
        item.alert for _, group in declared_objectives(valid_config()) for item in group
    ] == list(alerts)
    thresholds = {name: rule.expr.threshold for name, rule in alerts.items()}
    assert thresholds["ServingRecoveryDetectionSlow"] == 0.25
    assert thresholds["ServingRecoveryRestartSlow"] == 2.0
    assert thresholds["ServingRecoveryUnavailabilityLong"] == 3.0
    assert thresholds["ServingRecoverySloRecoverySlow"] == 5.0
    assert thresholds["ServingRecoveryPostFaultLatencyMissed"] == 0.95
    assert thresholds["WorkerRecoveryDetectionSlow"] == 1.0
    assert thresholds["WorkerRecoverySlow"] == 5.0
    assert alerts["ServingRecoveryPostFaultLatencyMissed"].expr.operand == Selector(
        POST_FAULT_LATENCY_RATIO
    )
    latency = serving.rules[2]
    assert Matcher("le", "=", "0.25") in latency.expr.numerator.selector.matchers
    for rule in alerts.values():
        labels = dict(rule.labels)
        assert labels["scope"] == "local_cpu_lab" and labels["slo"]
        assert "not a production SLO" in dict(rule.annotations)["description"]
    tight = build_rule_file(valid_config(serving_objectives={"restart_latency_ms": 1}))
    assert tight.rules[4].expr.threshold == 0.001


def test_rule_files_round_trip_and_only_canonical_bytes_parse():
    rule_file = build_rule_file(valid_config())
    text = render_rule_file(rule_file)

    assert text.startswith("".join(f"# {line}\n" for line in HEADER))
    assert parse_rule_file(text) == rule_file
    assert "  - name: 'ai_journey_worker_recovery_slo'\n" in text
    assert (
        "        expr: 'ai_journey_serving_recovery_detection_latency_seconds > 0.25'\n"
        in text
    )
    for changed in (
        text.replace("# Generated", "# Edited", 1),
        text.replace(HEADER[1], "Production rules."),
        "\n".join(text.split("\n")[2:]),
        text.replace("> 0.25'", "> 0.250'"),
    ):
        with pytest.raises(TelemetryMetricsError):
            parse_rule_file(changed)


def swap_lines(text, first, second):
    return text.replace(first + second, second + first, 1)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda t: t.replace(
                "'ai_journey_worker_recovery_slo'", "'ai_journey_serving_recovery_slo'"
            ),
            "group names repeat",
        ),
        (
            lambda t: t.replace(
                "'WorkerRecoverySlow'", "'WorkerRecoveryDetectionSlow'"
            ),
            "rule names repeat",
        ),
        (
            lambda t: t.replace(
                "'scope:ai_journey_serving_recovery_attempts:ok_ratio'",
                "'serving_attempts_ok_ratio'",
            ),
            "level:metric:operations",
        ),
        (
            lambda t: t.replace("'WorkerRecoverySlow'", "'workerRecoverySlow'"),
            "CamelCase",
        ),
        (
            lambda t: t.replace("          slo:", "          __slo:", 1),
            "invalid label name",
        ),
        (lambda t: t.replace("          slo:", "          alertname:", 1), "alertname"),
        (lambda t: t.replace("          slo:", "          le:", 1), "reserved"),
        (
            lambda t: t.replace(
                "'The final result omitted workload items'", "'{{ $value }} omitted'"
            ),
            "templates",
        ),
        (lambda t: t.replace(" > 0.25'", ">0.25'", 1), "PromQL"),
        (
            lambda t: t.replace(
                "(ai_journey_serving_recovery_attempts_total)'",
                "(ai_journey_serving_recovery_attempts_total) < 1'",
                1,
            ),
            "record a sum or a ratio",
        ),
        (
            lambda t: t.replace(
                "'ai_journey_worker_recovery_recovery_seconds > 5'",
                "'ai_journey_worker_recovery_recovery_seconds'",
            ),
            "compare with a threshold",
        ),
        (
            lambda t: t.replace(
                "        labels:\n          scope: 'local_cpu_lab'\n          slo: 'worker-detection'",
                "        for: '5m'\n        labels:\n          scope: 'local_cpu_lab'\n"
                "          slo: 'worker-detection'",
            ),
            "fields must be",
        ),
        (
            lambda t: t.replace(
                "      - alert: 'WorkerRecoverySlow'\n",
                "      - record: 'a:b:c'\n        alert: 'WorkerRecoverySlow'\n",
            ),
            "fields must be",
        ),
        (
            lambda t: swap_lines(
                t,
                "          scope: 'local_cpu_lab'\n",
                "          slo: 'worker-detection'\n",
            ),
            "sorted by unique name",
        ),
        (
            lambda t: t.replace("groups:\n", "groups:\n  name: 'x'\n").split(
                "  - name: 'ai_journey_serving"
            )[0],
            "nested block|sequence",
        ),
        (lambda t: t.replace("  - name:", "  - title:", 1), "fields must be"),
        (
            lambda t: t.replace(
                "        expr: 'ai_journey_worker_recovery_omitted_items_total > 0'\n",
                "        expr:\n          nested: 'x'\n",
            ),
            "one PromQL string",
        ),
    ],
)
def test_rule_file_structure_is_validated_like_a_strict_rule_loader(mutate, message):
    text = render_rule_file(build_rule_file(valid_config()))
    changed = mutate(text)
    assert changed != text
    with pytest.raises(TelemetryMetricsError, match=message):
        parse_rule_file(changed)


def test_rules_bind_to_the_lab_exposition(lab_metrics):
    validate_rule_file(build_rule_file(valid_config()), lab_metrics)

    alien = build_rule_file(
        valid_config(
            latency_buckets_us=[250, 750, 250_000],
            serving_objectives={"post_fault_latency_us": 750},
        )
    )
    with pytest.raises(TelemetryMetricsError, match="le='0.00075'"):
        validate_rule_file(alien, lab_metrics)
    with pytest.raises(TypeError):
        validate_rule_file(alien, lab_metrics.families)


def group(*rules):
    return RuleFile((RuleGroup("lab", tuple(rules)),))


def alert(expr, name="LabAlert"):
    return AlertingRule(name, expr, LABELS, (("summary", "Lab."),))


@pytest.mark.parametrize(
    ("rule_file", "message"),
    [
        (
            group(alert(Comparison(Selector("ai_journey_missing_seconds"), ">", 1.0))),
            "neither exported nor recorded",
        ),
        (
            group(
                alert(
                    Comparison(
                        Selector(
                            "ai_journey_serving_recovery_attempts_total",
                            (Matcher("phase", "=", "ok"),),
                        ),
                        ">",
                        1.0,
                    )
                )
            ),
            "does not carry",
        ),
        (
            group(
                alert(
                    Comparison(
                        Selector(
                            "ai_journey_serving_recovery_attempts_total",
                            (Matcher("outcome", "!=", "okay"),),
                        ),
                        ">",
                        1.0,
                    )
                )
            ),
            "no ai_journey_serving_recovery_attempts_total series",
        ),
        (
            group(
                alert(
                    Comparison(
                        Selector(
                            "ai_journey_serving_recovery_success_latency_seconds_count",
                            (Matcher("le", "=", "+Inf"),),
                        ),
                        ">",
                        1.0,
                    )
                )
            ),
            "does not carry",
        ),
        (
            group(
                alert(Comparison(Selector("a:b:c"), ">", 1.0), "First"),
                RecordingRule("a:b:c", RECOVERED, LABELS),
            ),
            "neither exported nor recorded earlier",
        ),
        (
            group(
                RecordingRule("a:b:c", RECOVERED, LABELS),
                alert(
                    Comparison(
                        Selector("a:b:c", (Matcher("phase", "=", "x"),)), ">", 1.0
                    )
                ),
            ),
            "does not record",
        ),
        (
            RuleFile(
                (
                    RuleGroup(
                        "one",
                        (RecordingRule("a:b:c", RECOVERED, LABELS),),
                    ),
                    RuleGroup("two", (alert(Comparison(Selector("a:b:c"), ">", 1.0)),)),
                )
            ),
            "neither exported nor recorded earlier in its group",
        ),
    ],
)
def test_validation_rejects_rules_the_exposition_cannot_answer(
    lab_metrics, rule_file, message
):
    with pytest.raises(TelemetryMetricsError, match=message):
        validate_rule_file(rule_file, lab_metrics)


def test_rule_objects_validate_their_parts():
    for build in (
        lambda: RuleGroup("lab", ()),
        lambda: RuleGroup("lab-group", (alert(Comparison(Selector("x"), ">", 1.0)),)),
        lambda: RuleFile(()),
        lambda: alert(Comparison(Selector("x"), ">", 1.0), "lab"),
        lambda: AlertingRule("Lab", Comparison(Selector("x"), ">", 1.0), (), ()),
        lambda: AlertingRule(
            "Lab", Comparison(Selector("x"), ">", 1.0), LABELS, (("summary", ""),)
        ),
        lambda: RecordingRule("a:b", RECOVERED, LABELS),
        lambda: RecordingRule("a:b:c", Selector("x"), LABELS),
        lambda: RecordingRule("a:b:c", RECOVERED, (("scope", "a\nb"),)),
    ):
        with pytest.raises(TelemetryMetricsError):
            build()


def test_no_alert_fires_when_the_evidence_meets_relaxed_objectives(lab_metrics):
    rule_file = build_rule_file(valid_config(**RELAXED))

    assert evaluate_rule_file(rule_file, lab_metrics) == ()


def test_alerts_fire_for_each_breached_objective(lab_metrics):
    tight = RELAXED["serving_objectives"] | {"restart_latency_ms": 1}
    rule_file = build_rule_file(
        valid_config(**(RELAXED | {"serving_objectives": tight}))
    )
    (restart,) = evaluate_rule_file(rule_file, lab_metrics)
    assert restart.alert == "ServingRecoveryRestartSlow"
    assert dict(restart.labels) == {
        "alertname": "ServingRecoveryRestartSlow",
        "scope": "local_cpu_lab",
        "slo": "serving-restart-latency",
    }
    restart_seconds = (
        lab_metrics.family("ai_journey_serving_recovery_restart_latency_seconds")
        .series[0]
        .value
    )
    assert float(restart.value) == restart_seconds > 0.001

    relaxed = build_rule_file(valid_config(**RELAXED))
    broken = lab_metrics
    for name in (
        "ai_journey_worker_recovery_omitted_items_total",
        "ai_journey_worker_recovery_duplicated_items_total",
    ):
        broken = replace_values(broken, name, lambda item: 1.0)
    broken = replace_values(
        broken, "ai_journey_serving_recovery_recovered", lambda item: 0.0
    )
    broken = replace_values(
        broken,
        "ai_journey_serving_recovery_requests_total",
        lambda item: 1.0
        if item.label_map()["outcome"] == "failed_after_budget"
        else item.value,
    )

    def slow_post_fault(item):
        histogram = item.value
        if item.label_map()["phase"] != "post_fault":
            return histogram
        count = histogram.count
        return HistogramValue(
            (0.0,) * (len(histogram.cumulative_counts) - 1) + (count,), histogram.sum
        )

    broken = replace_values(
        broken, "ai_journey_serving_recovery_success_latency_seconds", slow_post_fault
    )
    fired = {item.alert: item.value for item in evaluate_rule_file(relaxed, broken)}
    assert fired == {
        "ServingRecoveryNotRecovered": "0",
        "ServingRecoveryPostFaultLatencyMissed": "0",
        "ServingRecoveryRequestsFailed": "1",
        "WorkerRecoveryItemsDuplicated": "1",
        "WorkerRecoveryItemsOmitted": "1",
    }
