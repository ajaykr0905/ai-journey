# Telemetry metrics from verified recovery evidence

This lab is the metrics half of the observability workstream. It turns the
sealed evidence of one serving-recovery run and one worker-recovery run
(the Day 70 labs, see [serving-failure-recovery.md](serving-failure-recovery.md))
into Prometheus metrics in text exposition format 0.0.4, plus recording and
alerting rules generated from declared objectives. Nothing is derived until
the existing verifiers accept the evidence. Every value is local CPU lab
telemetry, and every exported series carries the label
`scope="local_cpu_lab"`. This is an offline export from one finished run.
It is not a live exporter: no Prometheus server scrapes it, and no Grafana,
OpenTelemetry collector or Alertmanager is involved.

The lab uses only the standard library. It needs no PyYAML or Prometheus
client package, and it does not import torch or NumPy.

## Produce evidence, export and verify

Each command needs a new directory. The recovery labs and the export refuse
to reuse a directory or overwrite a file.

```bash
PYTHONPATH=src python -m ai_journey.serving_recovery_cli run \
  --source data/day-19-demo-names.txt \
  --config config/day-70-serving-recovery.json \
  --run-dir /tmp/day-73/serving

PYTHONPATH=src python -m ai_journey.recovery_cli run \
  --source data/day-19-demo-names.txt \
  --config config/day-70-worker-recovery.json \
  --run-dir /tmp/day-73/worker

PYTHONPATH=src python -m ai_journey.telemetry_metrics_cli export \
  --config config/day-73-telemetry-metrics.json \
  --serving-run-dir /tmp/day-73/serving \
  --worker-run-dir /tmp/day-73/worker \
  --output-dir /tmp/day-73/telemetry

PYTHONPATH=src python -m ai_journey.telemetry_metrics_cli verify \
  --output-dir /tmp/day-73/telemetry \
  --serving-run-dir /tmp/day-73/serving \
  --worker-run-dir /tmp/day-73/worker
```

Both telemetry commands print one JSON receipt. It holds the report seal,
the scope, the family, series and sample counts, the rule counts, the
breached objectives and the alerts that would fire. `export` verifies what
it wrote with the same code that `verify` runs.

| Exit code | Meaning |
| --- | --- |
| 0 | The export verified and every declared objective was met. |
| 1 | `verify` rejected the export or its evidence. |
| 2 | Usage error, invalid input, refused output, or evidence that `export` rejected. |
| 3 | The export verified, but a declared objective was breached, so its alert would fire. |

## What `export` does

1. It loads strict controls from `config/day-73-telemetry-metrics.json`.
   Unknown, missing and duplicate keys fail at every level. So do booleans
   in integer fields and non-finite numbers.
2. It runs `verify_serving_recovery_run` on the serving directory. That
   verifier recomputes the run from its raw evidence. The adapter then
   loads the sealed report and the attempt log. The report is bound to the
   verification by its seal. The parsed attempts are bound by re-encoding
   them to the attempt-log digest that the verified report records. A file
   replaced after verification is refused.
3. It loads the worker `report.json` as bounded, canonical, strict JSON and
   passes it to `verify_recovery_report` together with `trace.json`. The
   worker verifier compares numbers by value, so the adapter also requires
   counts, timestamps and exit codes to be exact integers. That verifier
   binds only the timestamps and the trace head to the trace. The adapter
   therefore re-reads the trace, requires the five-event kill-and-resume
   lifecycle, and requires the run and attempt identities, the failed exit
   code and the item counts to equal the events that record them.
4. It derives the metric families, described below, and encodes the
   exposition.
5. It generates the rules from the objectives, binds every selector to the
   exposition and evaluates the rules at one instant.
6. It judges every objective a second time, straight from the evidence, in
   integer nanoseconds and counts. The export is refused unless this
   judgement and the evaluated alerts agree on which objectives were
   breached.
7. It seals the report and publishes `config.json`, `metrics.prom`,
   `rules.yml` and then `report.json` into the new directory, using the
   serving lab's no-overwrite atomic writer. The output directory may not
   lie inside either evidence directory.

## Output directory

| File | Contents |
| --- | --- |
| `config.json` | Canonical copy of the controls |
| `metrics.prom` | Text exposition 0.0.4, served as `text/plain; version=0.0.4; charset=utf-8` |
| `rules.yml` | Prometheus recording and alerting rule groups in a constrained YAML subset |
| `report.json` | Sealed report: the config, both evidence seals, the file digests, the objective judgements, the firing alerts and the evidence boundary |

The report seal is the SHA-256 of the canonical report without the seal
field. It proves content identity, not authorship. That is why `verify`
recomputes everything instead of trusting the report.

## Metric families

Every series also carries `scope="local_cpu_lab"`. Counters cover the whole
run. Durations are seconds gauges, converted once from the verified integer
nanoseconds.

| Family | Type | Labels | Derived from |
| --- | --- | --- | --- |
| `ai_journey_serving_recovery_attempts_total` | counter | `outcome` (7 attempt classes) | reconciliation attempt outcomes |
| `ai_journey_serving_recovery_requests_total` | counter | `outcome` (3 request outcomes) | reconciliation totals |
| `ai_journey_serving_recovery_executions_total` | counter | `lifetime` | journal executions by server lifetime |
| `ai_journey_serving_recovery_duplicate_executions_total` | counter | | at-least-once duplicates |
| `ai_journey_serving_recovery_lost_response_executions_total` | counter | | executions whose response was lost |
| `ai_journey_serving_recovery_retried_attempts_total` | counter | | attempts after each request's first |
| `ai_journey_serving_recovery_late_acknowledgements_total` | counter | | successes from the killed lifetime that arrived after the kill |
| `ai_journey_serving_recovery_restarts_total` | counter | | replacement lifetimes started |
| `ai_journey_serving_recovery_detection_latency_seconds` | gauge | | kill to observed exit |
| `ai_journey_serving_recovery_restart_latency_seconds` | gauge | | observed exit to replacement ready |
| `ai_journey_serving_recovery_unavailability_seconds` | gauge | | first failure after the kill to first post-fault success; no sample when null |
| `ai_journey_serving_recovery_time_to_slo_recovery_seconds` | gauge | | kill to the end of the SLO run; no sample when null |
| `ai_journey_serving_recovery_recovered` | gauge | | 1 when the run recovered, else 0 |
| `ai_journey_serving_recovery_run_info` | gauge | `run_id`, `report_sha256` | always 1 |
| `ai_journey_serving_recovery_success_latency_seconds` | histogram | `phase` | acknowledged attempt latencies |
| `ai_journey_serving_recovery_failed_attempt_duration_seconds` | histogram | `outcome` (6 failure classes) | failed attempt durations |
| `ai_journey_worker_recovery_completed_items_total` | counter | | items in the final result |
| `ai_journey_worker_recovery_lost_items_total` | counter | | uncheckpointed items lost to the kill |
| `ai_journey_worker_recovery_replayed_items_total` | counter | | items recomputed from the checkpoint |
| `ai_journey_worker_recovery_omitted_items_total` | counter | | workload items missing from the result |
| `ai_journey_worker_recovery_duplicated_items_total` | counter | | items committed more than once |
| `ai_journey_worker_recovery_items_at_fault` | gauge | | items done when the kill landed |
| `ai_journey_worker_recovery_durable_items_at_fault` | gauge | | items covered by the last checkpoint |
| `ai_journey_worker_recovery_detection_seconds` | gauge | | kill to detected failure |
| `ai_journey_worker_recovery_restart_delay_seconds` | gauge | | detection to resumed start |
| `ai_journey_worker_recovery_recovery_seconds` | gauge | | kill to completed workload |
| `ai_journey_worker_recovery_attempt_exit_code` | gauge | `attempt` (`failed`, `resumed`) | worker process exit codes |
| `ai_journey_worker_recovery_run_info` | gauge | `run_id`, `report_sha256`, `trace_sha256` | always 1 |

The success latency phases split the acknowledged attempts exactly as the
serving lab's timing does. `pre_fault` attempts finished by the kill.
`late_acknowledgement` attempts were sent by the killed lifetime but arrived
after the kill. `post_fault` attempts were served by a later lifetime. The
adapter refuses evidence whose phase counts differ from the report's
latency sample counts.

Bucket bounds are integer microseconds in the config. Latencies stay
integer nanoseconds, so a bucket is decided by an exact integer comparison,
never by float rounding. Labels never carry request identities. Each label
takes its values from a small bounded set, apart from the identity labels on
the two `run_info` series.

## Text exposition format 0.0.4

The encoder writes one `# HELP` and one `# TYPE` line per family. Families
are sorted by name and series by label values. Labels are sorted by name,
with `le` last on bucket lines. Each histogram series lists its cumulative
buckets in bound order, ending with `le="+Inf"`, then `_sum` and `_count`.
Help text escapes the backslash and the line feed, and label values also
escape the double quote. No timestamps are written. Values use Go's shortest
`strconv.FormatFloat(v, 'g', -1, 64)` spelling, as the Go client library
writes them, so 2391208 appears as `2.391208e+06`. A gauge with no
measurement, such as a null unavailability, keeps its `# HELP` and `# TYPE`
lines with no sample.

The model and the parser enforce the Prometheus data model, plus a few house
rules:

- Metric names match `[a-zA-Z_][a-zA-Z0-9_]*`. Colons are reserved for
  recording rules.
- Label names never start with `__`. `le`, `quantile`, `job` and `instance`
  are reserved.
- Label values are never empty, because Prometheus reads an empty value as
  an absent label.
- Counters end in `_total`.
- Values are finite. A histogram's `+Inf` bucket equals its `_count`.

`parse_exposition` is strict. It rejects comments and blank lines,
timestamps, untyped and summary families, and repeated or late HELP and TYPE
lines. It also rejects a family split into two groups, unknown escapes,
empty label sets, repeated labels or series, and histograms whose buckets do
not ascend to `+Inf` before `_sum` and `_count`. Parsing the encoder's
output returns the same metric set. `is_canonical_exposition` also requires
the encoder's exact bytes, and the verifier uses that check.

## Recording and alerting rules

`rules.yml` has two groups. `ai_journey_serving_recovery_slo` first records
three ratios, each written as `sum by (scope)` over a counter divided by
`sum by (scope)` over another:

- `scope:ai_journey_serving_recovery_attempts:ok_ratio`
- `scope:ai_journey_serving_recovery_requests:success_ratio`
- `scope:ai_journey_serving_recovery_success_latency:post_fault_within_objective_ratio`:
  the post-fault bucket at the latency objective divided by the post-fault
  count.

Then one alert guards each declared objective. Each alert names its
objective in an `slo` label.

| Alert | `slo` | Fires when (pinned config) |
| --- | --- | --- |
| `ServingRecoveryDetectionSlow` | `serving-detection-latency` | `ai_journey_serving_recovery_detection_latency_seconds > 0.25` |
| `ServingRecoveryRestartSlow` | `serving-restart-latency` | `ai_journey_serving_recovery_restart_latency_seconds > 2` |
| `ServingRecoveryUnavailabilityLong` | `serving-unavailability` | `ai_journey_serving_recovery_unavailability_seconds > 3` |
| `ServingRecoverySloRecoverySlow` | `serving-time-to-slo-recovery` | `ai_journey_serving_recovery_time_to_slo_recovery_seconds > 5` |
| `ServingRecoveryNotRecovered` | `serving-recovered` | `ai_journey_serving_recovery_recovered < 1` |
| `ServingRecoveryRequestsFailed` | `serving-request-success` | `ai_journey_serving_recovery_requests_total{outcome="failed_after_budget"} > 0` |
| `ServingRecoveryPostFaultLatencyMissed` | `serving-post-fault-latency` | the post-fault latency ratio `< 0.95` |
| `WorkerRecoveryDetectionSlow` | `worker-detection` | `ai_journey_worker_recovery_detection_seconds > 1` |
| `WorkerRecoverySlow` | `worker-recovery-time` | `ai_journey_worker_recovery_recovery_seconds > 5` |
| `WorkerRecoveryItemsOmitted` | `worker-no-omitted-items` | `ai_journey_worker_recovery_omitted_items_total > 0` |
| `WorkerRecoveryItemsDuplicated` | `worker-no-duplicated-items` | `ai_journey_worker_recovery_duplicated_items_total > 0` |

The rules contain no `for` and no `rate()`. The exposition is one snapshot
of a finished run, so each counter already covers the whole run, and no
later scrape will come to wait for. A real deployment that scraped a live
service would choose rate windows and `for` durations itself.

The rules are written in a small PromQL subset: selectors with `=` and `!=`
matchers, `sum by`, a ratio of two sums, and a comparison with a number.
Each expression has one canonical spelling, and the parser accepts nothing
else. Evaluation follows PromQL instant-query semantics on that subset:

- A matcher on an absent label compares with the empty string.
- `sum` over no series yields no series, not 0.
- Results carry no metric name.
- Division by zero gives `+Inf`, `-Inf` or `NaN`.
- A comparison with `NaN` holds only for `!=`.

So an absent gauge fires no alert. A run without unavailability raises no
unavailability alert, and a run that never recovered raises
`ServingRecoveryNotRecovered`.

The YAML is a subset that the lab writes and reads itself, without PyYAML:

- block mappings, plain identifier keys and two-space indentation;
- sequences of mappings whose first key shares the dash line;
- single-quoted one-line scalars, so no YAML 1.1 reader turns `0s` or `on`
  into a number or a boolean;
- `#` comments only before the document.

The reader rejects tabs, blank lines, trailing spaces, flow collections,
anchors, aliases, tags, block scalars, plain scalars, inline comments,
duplicate keys and empty collections. The structural validator then checks
what Prometheus' rule loader checks, and more:

- group names and rule names are unique;
- each rule has exactly one of `record` and `alert`;
- recording names follow `level:metric:operations`;
- alert names are CamelCase;
- label names are valid and unreserved, and `alertname` is not set;
- annotations are literal, printable text without templates;
- every expression is in the canonical subset;
- the file is exactly the canonical bytes.

Every selector must also bind to the exposition. It must read an exported
sample name, or a recording rule defined earlier in the same group. Its
matchers must name labels that the series carry and values that occur in
the family. An `le` matcher must spell one of the family's bucket bounds.

## Declared objectives in the pinned config

| Control | Pinned value | Why |
| --- | --- | --- |
| `latency_buckets_us` | 250 us to 1 s in 1, 2.5, 5 steps | Loopback successes take about 1 ms here and refused connects take about 0.3 ms, so both land in distinct buckets |
| `serving_objectives.detection_latency_ms` | 250 | Below the Day 70 `detection_timeout_ms` of 1000; exit status is polled every 10 ms |
| `serving_objectives.restart_latency_ms` | 2000 | Covers the 50 ms restart backoff and an interpreter cold start; below the 3000 ms readiness timeout |
| `serving_objectives.unavailability_ms` | 3000 | Client-observed outage budget for one restart |
| `serving_objectives.time_to_slo_recovery_ms` | 5000 | Kill to 8 consecutive in-SLO successes |
| `serving_objectives.post_fault_latency_us` | 250000 | The Day 70 serving run's own `slo_latency_ms` of 250 |
| `serving_objectives.post_fault_within_latency_ratio` | 0.95 | Share of post-fault successes that must meet that latency |
| `worker_objectives.detection_ms` | 1000 | Above the Day 70 worker `heartbeat_timeout_ms` of 500 |
| `worker_objectives.recovery_ms` | 5000 | The Day 70 `worker_timeout_ms` |

These objectives are declared lab controls, chosen for this lab. They are
not service-level agreements and are not taken from any external service.
The objectives with no threshold always exist: the run recovers, no request
fails after its budget, and no worker item is omitted or duplicated.

## What `verify` recomputes

`verify` trusts nothing that the export states. It checks the report
schema, kind, scope and boundary, and it requires the copied config to be
canonical and equal to the report's. It then re-verifies both evidence
directories with their own verifiers, rederives the metric set and
regenerates the rules. `metrics.prom` and `rules.yml` must be exactly those
bytes, and each must also parse strictly on its own. Key samples of the
stored exposition are cross-checked against the verified evidence reports
directly, not through the adapters, so an adapter cannot mislabel a sample
and still pass its own check. Finally, every report section is recomputed
and compared as canonical JSON, including the objective judgements and the
firing alerts. Evidence from another run, a damaged trace, a reformatted
file or a resealed report all fail with exit code 1.

## One development-host run

The following values come from one run of the commands above on a macOS
arm64 development host on 2026-11-16. They show what the export looks like.
They are not a benchmark, and timings depend on the host and its load.

| Measurement | Value |
| --- | --- |
| Serving requests, attempts | 48 requests in 56 attempts: 48 `ok`, 6 `connection_refused`, 2 `connection_reset` |
| Serving detection, restart | 9.967 ms, 106.662 ms |
| Unavailability, time to SLO recovery | 187.877 ms, 191.645 ms |
| Success latency phases | 16 pre-fault, 1 late acknowledgement, 31 post-fault; all 31 post-fault successes took at most 2.5 ms |
| Worker detection, recovery | 7.376 ms, 66.488 ms; 3 items lost and replayed, none omitted or duplicated |
| Export | 28 families, 45 series, 171 samples in a 26,653-byte exposition; 3 recording and 11 alerting rules |
| Objectives | none breached, no alert firing, exit code 0 for `export` and `verify` |

## Limitations

- This is an offline export of one finished run. It is not a scrape target
  and holds no history, and a `rate()` over its counters would be
  meaningless.
- The rule evaluator implements only the PromQL subset above. It is not a
  PromQL engine, and the YAML reader reads only the subset its own emitter
  writes.
- The repository does not run Prometheus, `promtool`, Grafana, an
  OpenTelemetry collector or Alertmanager. During development, the
  exposition was re-encoded byte for byte by the Go client library's
  `expfmt` text parser and writer, and the rule file was read into the same
  structure by Go's `yaml.v3`. Those checks ran once by hand and are not
  part of the test suite.
- Durations come from `time.monotonic_ns` on the experiment host and depend
  on its load. The objectives are declared lab controls, not production
  SLOs.
- This is not GPU, CUDA, distributed, multi-host or production monitoring
  evidence.
