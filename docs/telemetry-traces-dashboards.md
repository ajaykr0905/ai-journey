# Telemetry traces and dashboards for the recovery labs

This lab is the tracing and dashboard half of the observability workstream
(OpenTelemetry, Prometheus and Grafana). It turns the verified, hash-chained
lifecycle evidence of the serving-recovery and worker-recovery labs into
OpenTelemetry spans encoded as OTLP/JSON, and it generates a Grafana dashboard
as code for the metric names the metrics workstream exposes. Everything is
derived from sealed local CPU evidence of one host and labeled as local CPU
lab telemetry. The modules import only the standard library.

## Run and verify

Produce one run of each lab, then export, generate and verify. Every output
directory must be new; nothing is ever overwritten.

```bash
PYTHONPATH=src python -m ai_journey.serving_recovery_cli run \
  --source data/day-19-demo-names.txt \
  --config config/day-70-serving-recovery.json \
  --run-dir /tmp/day-73/serving

PYTHONPATH=src python -m ai_journey.recovery_cli run \
  --source data/day-19-demo-names.txt \
  --config config/day-70-worker-recovery.json \
  --run-dir /tmp/day-73/worker

PYTHONPATH=src python -m ai_journey.telemetry_traces_cli export \
  --config config/day-73-telemetry-traces.json \
  --serving-run-dir /tmp/day-73/serving \
  --worker-run-dir /tmp/day-73/worker \
  --output-dir /tmp/day-73/traces

PYTHONPATH=src python -m ai_journey.telemetry_traces_cli dashboard \
  --config config/day-73-telemetry-traces.json \
  --output-dir /tmp/day-73/dashboard

PYTHONPATH=src python -m ai_journey.telemetry_traces_cli verify \
  --report /tmp/day-73/traces/report.json \
  --serving-run-dir /tmp/day-73/serving \
  --worker-run-dir /tmp/day-73/worker \
  --config config/day-73-telemetry-traces.json

PYTHONPATH=src python -m ai_journey.telemetry_traces_cli verify \
  --report /tmp/day-73/dashboard/report.json \
  --config config/day-73-telemetry-traces.json
```

`export` accepts either run directory alone or both. It writes
`traces.otlp.json` and a sealed `report.json`; `dashboard` writes
`dashboard.json` and a sealed `report.json`. Both then verify what they
wrote with the same code `verify` uses. Each command prints one JSON receipt
with the report seal, the artifact digest and `telemetry_scope` set to
`local-cpu-lab`; an export receipt also lists each run's root `traceparent`.
There is no console script yet, so use the module form.

| Exit code | Meaning |
| --- | --- |
| 0 | The artifact was written or verified. |
| 1 | Source evidence or a stored artifact was rejected. |
| 2 | Usage errors, unreadable or invalid inputs, or a refused output. |

## Conversion happens only after the lab verifiers accept the evidence

For a serving run, `verify_serving_recovery_run` must accept the directory
first. The report is then reloaded and must carry the seal the verifier
accepted, and `trace.json` and `attempts.json` are read once each and must
hash to the digests that report binds, so a file swapped after verification is
refused. For a worker run, the report is read once as bounded, canonical,
duplicate-free JSON and passed with its trace to `verify_recovery_report`;
the trace is then re-read, re-verified as a hash chain and bound to the
report by its head digest. `verify_recovery_report` never reads the report's
evidence boundary, so the export also requires the exact boundary text
`build_recovery_report` seals; a resealed report that claims other evidence
is rejected. The worker experiment writes exactly five records
(started, fault_injected, failure_detected, resumed, completed), and any other
shape is refused. `verify_recovery_report` compares numbers by value and never
reads record payloads, so the export also requires exact integer counts and
exit codes, each record's payload keys and types, and a report whose run ID,
workload digest, parent checkpoint, recovery time and item counts equal its
trace's. `verify_recovery_report` indexes the trace by position and
raises `IndexError` on a valid chain shorter than five records; the export
reports that as rejected evidence instead of crashing.

## Span model

Each run is one trace and one OTLP resource. Span names are fixed per role,
with at most a small lifetime or attempt number; identities live in
attributes.

Serving-recovery trace:

| Span | Kind | Interval | Notes |
| --- | --- | --- | --- |
| `serving-recovery run` | INTERNAL | first to last lifecycle record | Root; ERROR only if the run did not recover. |
| `server lifetime N` | INTERNAL | spawn or restart to observed exit or stop | One event per lifecycle record; the SIGKILLed lifetime is ERROR with `error.type=SIGKILL`. |
| `fault detection` | INTERNAL | fault to observed exit | Lasts `timing.detection_latency_ns`. |
| `server restart` | INTERNAL | observed exit to replacement ready | Lasts `timing.restart_latency_ns`. |
| `client unavailability` | INTERNAL | first failure after the fault to first replacement success | Lasts `timing.unavailability_ns`; omitted when null. |
| `SLO recovery` | INTERNAL | fault to the end of the in-SLO run | Lasts `timing.time_to_slo_recovery_ns`; omitted when null. |
| `serving request` | INTERNAL | first attempt start to last attempt finish | One per logical request. |
| `POST /generate` | CLIENT | attempt start to finish | One per recorded attempt, child of its request. |

Worker-recovery trace:

| Span | Kind | Interval | Notes |
| --- | --- | --- | --- |
| `worker-recovery run` | INTERNAL | started to completed | Root, with lost, replayed, omitted and duplicated item counts. |
| `worker attempt 1` | INTERNAL | started to failure detected | ERROR with `error.type=process_kill`. |
| `worker attempt 2` | INTERNAL | resumed to completed | Carries the parent checkpoint digest. |
| `failure detection` | INTERNAL | fault to failure detected | Reproduces `timing.detection_time_ms`. |
| `worker restart` | INTERNAL | failure detected to resumed | Reproduces `timing.restart_delay_ms`. |
| `recovery` | INTERNAL | fault to completed | Reproduces `timing.recovery_time_ms`. |

Phase spans carry `ai_journey.span.derived=true` and name the sealed timing
field they reproduce in `ai_journey.phase.source`. Every lifecycle record
becomes exactly one span event named `ai_journey.lifecycle.<kind>` that
carries the record's `ai_journey.event.sequence` and hash-chain digest
`ai_journey.event.sha256`. The injected fault is the
`ai_journey.lifecycle.fault` or `ai_journey.lifecycle.fault_injected` event.

Attributes use OpenTelemetry semantic-convention names where one applies:
`service.name`, `service.namespace` and `deployment.environment.name` on the
resource; `process.pid`, `process.exit.code`, `server.address`,
`server.port` and `error.type` on lifetime and attempt spans where they
apply; and
`http.request.method`, `url.full`, `url.template`,
`network.protocol.version`, `http.request.resend_count` (set on retries) and
`http.response.status_code` on CLIENT spans. A failed attempt's `error.type`
is its HTTP status for `http_error` and its outcome class otherwise.
`process.exit.code` is the status the supervisor observed; Python reports
death by signal N as -N. Lab facts use the `ai_journey.` namespace: run,
lifetime, request, attempt and fault details, the evidence digests and both
evidence boundaries, and the clock mapping. No schema URL is claimed.

## Identifiers

A trace ID is the first 16 bytes of SHA-256 over the canonical JSON of the
lab, the run identity and the head digest of the verified lifecycle trace. A
span ID is the first 8 bytes of SHA-256 over the trace ID and a stable span
key such as `lifetime/<lifetime_id>` or `attempt/<idempotency_key>/<n>`. A
digest prefix that is zero or equal to the parent span ID is re-derived with an
added salt, up to four derivations in total. Exporting the same evidence twice
gives the same IDs and bytes. A rerun of one pinned config keeps its run ID
but gets a new trace head, so it gets a new trace instead of colliding with
the first. Contexts are validated and rendered by `trace_context.TraceContext`;
the sampled flag is set and the W3C Level 2 random flag is not, because the
IDs are digests.

## Time mapping

The labs record only `time.monotonic_ns` in the experiment process, whose
epoch is arbitrary, while OTLP needs Unix nanoseconds. The export applies one
stated mapping per run:

```text
unix_nano = anchor_unix_nano + (monotonic_ns - origin_monotonic_ns)
```

The origin is the run's first lifecycle record. The anchor is the declared
instant `2026-11-16T00:00:00Z` (`1794787200000000000`) pinned in
`config/day-73-telemetry-traces.json`. Absolute OTLP timestamps are therefore
synthetic, while durations and ordering are exact measured differences. The
clock, formula, origin and anchor are recorded on every resource and in the
report.

## OTLP/JSON encoding and validation

`traces.otlp.json` is one `ExportTraceServiceRequest` (opentelemetry-proto
`trace/v1`) in the OTLP JSON mapping: lowerCamelCase field names, trace and
span IDs as hexadecimal strings rather than base64, `SpanKind` and
`StatusCode` as integers, and 64-bit integers (times and `intValue`) as
decimal strings. Attribute values are strings, booleans or 64-bit integers.
Spans are ordered by start time, then depth, so each parent precedes its
children. The bytes are canonical JSON (sorted keys, compact, ASCII) plus one
newline.

The validator reads that JSON on its own terms and is stricter than OTLP asks
of receivers. It rejects unknown fields; requires 32 and 16 nonzero lowercase
hexadecimal digits for trace and span IDs; requires decimal-string unsigned
64-bit nanoseconds with the start no later than the end; accepts only known
span kinds and status codes, and a status message only on errors; requires
unique semantic-convention attribute keys; and checks, across the request,
that span IDs are unique per trace, each trace lives in one scope and has one
root, every `parentSpanId` names an earlier span of the same trace whose
interval contains the child, spans in a scope are ordered by start time, and
events lie in order inside their span. Every resource must declare
`service.name`, `deployment.environment.name=local-cpu-lab` and an
`ai_journey.evidence.boundary` statement.

## Dashboard as code

`dashboard.json` is Grafana dashboard JSON with `schemaVersion` 39, a null
`id` for import and the UID, title, time range, refresh and latency quantiles
pinned in the config. Every query panel and target uses the Prometheus
datasource chosen through the `${datasource}` template variable.

| Panel | Type | Query |
| --- | --- | --- |
| Local CPU lab telemetry | text | The evidence boundary. |
| Serving client attempts by outcome | bargauge | `sum by (outcome) (ai_journey_serving_attempts_total)` |
| Serving executions and duplicates | stat | `sum(...)` of the executions and duplicate-executions counters |
| Serving recovery durations | stat | `max(...)` of the detection, restart, unavailability and SLO recovery gauges, in seconds |
| Serving request latency quantiles | timeseries | `histogram_quantile(q, sum by (le) (ai_journey_serving_request_latency_seconds_bucket))` for q in 0.5, 0.95, 0.99 |
| Worker recovery time | stat | `max(ai_journey_worker_recovery_seconds)` |
| Worker items lost | stat | `sum(ai_journey_worker_items_lost_total)` |

The metric catalog in `telemetry_dashboard_metrics.py` is the contract with the
metrics workstream:

| Metric | Type | Labels |
| --- | --- | --- |
| `ai_journey_serving_attempts_total` | counter | `outcome` |
| `ai_journey_serving_executions_total` | counter | |
| `ai_journey_serving_duplicate_executions_total` | counter | |
| `ai_journey_serving_detection_seconds` | gauge | |
| `ai_journey_serving_restart_seconds` | gauge | |
| `ai_journey_serving_unavailability_seconds` | gauge | |
| `ai_journey_serving_slo_recovery_seconds` | gauge | |
| `ai_journey_serving_request_latency_seconds` | histogram | |
| `ai_journey_worker_recovery_seconds` | gauge | |
| `ai_journey_worker_items_lost_total` | counter | |

The counters and the latency histogram are as agreed. The duration metrics are
assumed to be gauges holding the last verified run's value; if they are
exposed with another type, change the catalog and regenerate the dashboard.

The dashboard validator checks the top-level, panel and target fields the
generator emits, `schemaVersion` 39, unique positive panel IDs and unique
titles, `gridPos` rectangles inside the 24-column grid that do not overlap,
datasource references that resolve to a declared Prometheus datasource
variable, unique uppercase `refId`s, the local CPU lab tag, title and header
panel, and every PromQL expression. The PromQL check is a tokenizer, not a
parser: brackets must balance and nest, strings must end, every called name
must be a known PromQL function or aggregation, `by`, `without`, `on` and
`ignoring` must take a label list, braces must hold `label op "value"`
matchers, ranges must follow a selector and hold a duration or a Grafana
interval variable, and every selected series must belong to the catalog.
A selector must name its series before any braces and may not also match
`__name__`, so matchers alone, such as `{__name__=~".+"}`, cannot select
series outside the catalog. Types, arity and operator precedence are not
checked.

## Sealed reports and verification

Each report is canonical JSON whose `report_sha256` is the SHA-256 of the
report without that field. The trace-export report binds the export's size
and digest to the embedded config, the encoding rules, each run's report seal,
trace head, evidence boundary, trace ID, root `traceparent` and time mapping,
and the validator's counts. It records no filesystem paths. The dashboard
report binds the dashboard's digest to the config, the metric catalog, the
Grafana schema version, the query summary and the statement that the
dashboard was not imported into a running Grafana.

`verify` trusts no stored summary. For an export it needs the run directory
of exactly the exported labs, re-runs each lab verifier, rebuilds the export
under the embedded config and requires the same bytes and source statements.
Both kinds of report must then rebuild byte for byte, so a resealed report
that spells a sealed number as `1.0` or `true` is rejected.
Separately from the encoder it maps the OTLP JSON back to the raw evidence:
one event per lifecycle record with that record's sequence, digest and
mapped time; resource identities and clock mappings equal to the run's; phase
durations equal to the sealed timings; and one CLIENT span per recorded
attempt, with matching durations and error count. For a dashboard it
re-validates the file, requires every catalog metric to be charted and
rebuilds the same bytes from the embedded config. `--config` binds either
report to a pinned config file.

## Evidence boundary

Local CPU lab telemetry: spans derived from sealed, independently verified
recovery-lab evidence recorded on one CPU host. This is not production, GPU,
CUDA, distributed or multi-host monitoring. Durations are measured
monotonic-clock differences; absolute timestamps come from a declared anchor
and are synthetic.

The dashboard states its own boundary: Local CPU lab telemetry: panels read
metrics derived from sealed recovery-lab evidence recorded on one CPU host.
This is not production, GPU, CUDA, distributed or multi-host monitoring, and
it makes no SLO, capacity or alerting claim.

## Limitations

- The export has not been sent to an OTLP collector, and the dashboard has not
  been imported into Grafana, as part of this lab. Both are files checked by
  this lab's own validators.
- The OTLP validator accepts only the subset this encoder emits, for example
  no links, `traceState`, `doubleValue` or schema URLs.
- Latency quantiles are `histogram_quantile` estimates interpolated within
  buckets. The labs' sealed reports keep the exact nearest-rank percentiles.
- The worker lab records no process IDs, and its float `recovery_time_ms`
  payload is represented by the recovery phase span's exact duration.
- A seal proves content identity, not authorship. Evidence rewritten
  consistently enough to pass the lab verifiers exports and verifies like
  genuine evidence. Keep the run directories and compare them with
  independent runs.
