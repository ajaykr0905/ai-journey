# Serving failure injection and measured recovery

This lab kills a real loopback inference server process while clients are
sending requests, then measures what those clients experience during failure
detection, restart and recovery. Everything runs on one CPU host over
`127.0.0.1`. The served model is a tiny deterministic character-bigram model
built from the checked-in public fixture `data/day-19-demo-names.txt`.

## Run and verify the pinned experiment

Use a new run directory. The runner refuses to reuse one or overwrite evidence.

```bash
ai-journey-serving-recovery run \
  --source data/day-19-demo-names.txt \
  --config config/day-70-serving-recovery.json \
  --run-dir /tmp/day-70-serving-recovery

ai-journey-serving-recovery verify \
  --run-dir /tmp/day-70-serving-recovery
```

From a source checkout, the module form takes the same arguments:
`PYTHONPATH=src python -m ai_journey.serving_recovery_cli run` and
`PYTHONPATH=src python -m ai_journey.serving_recovery_cli verify`. It also works
from an installed package without `PYTHONPATH`. Both commands print one JSON
receipt with the run identity, report seal, request count, duplicate executions
and millisecond summary. `run` writes the report and then verifies it with the
same code `verify` uses.

| Exit code | Meaning |
| --- | --- |
| 0 | Evidence verified; every request succeeded and the SLO run was observed. |
| 1 | `verify` rejected the stored evidence. |
| 2 | Invalid inputs, or a run that could not complete (for example the restart budget ran out). |
| 3 | Evidence verified, but the run did not recover within its declared bounds. |

## What happens during a run

1. Strict controls are loaded from JSON. Missing, unknown and duplicate keys
   and non-finite numbers are rejected. Cross-field rules keep the controls
   coherent; for example, each request's retry backoff must outlast the
   supervisor's detection timeout, restart backoff and readiness timeout.
2. The run identity is derived from the config and model digests. Each server
   lifetime and each logical request gets an identity derived from it. All
   attempts of one request carry the same `Idempotency-Key`.
3. The supervisor launches the server without a shell, in a new session. The
   server binds `127.0.0.1` port 0 and publishes its pid, port, lifetime and
   model digest in a one-line stdout handshake. The supervisor accepts it only
   from the pid it spawned and then requires an exact `/health` identity.
4. Up to `concurrency` client threads send requests. Every attempt has a
   whole-request deadline and is classified as `ok`, `connection_refused`,
   `connection_reset`, `timeout`, `http_error` or `invalid_response`. Failed
   attempts retry with capped exponential backoff against the currently
   published endpoint. That endpoint stays stale until a replacement is ready.
5. When the shared attempt ledger records the K-th acknowledged response, it
   asks the supervisor to SIGKILL its current, ready, live child. This happens
   while the ledger holds its lock, so no later acknowledgement is recorded
   before the kill.
6. The supervisor observes the exit status, waits the declared restart
   backoff, launches a replacement lifetime and waits for its readiness. At most
   `restart_limit` restarts are allowed. If they run out, or the supervisor
   itself fails unexpectedly, it records that failure, clients stop at their
   next attempt, and the run ends at once with that error and exit code 2.
7. After every request finishes, the replacement is stopped with SIGTERM. The
   runner reconciles attempts with both lifetimes' journals, measures recovery,
   chains the lifecycle trace, writes the sealed report and verifies it.

## Run directory

| File | Contents |
| --- | --- |
| `source.txt`, `config.json` | Byte copy of the source and the canonical controls |
| `attempts.json` | Every classified attempt in ledger order, with monotonic start and finish times |
| `journal-lifetime-N.jsonl` | One committed line per request executed by lifetime N |
| `trace.json` | Hash-chained lifecycle: spawn, ready, fault, exit observed, restart, ready, recovered, finished |
| `report.json` | Sealed summary with the evidence digests, measurements and evidence boundary |
| `lifetime-N.stderr` | Server diagnostics; normally empty and not part of the verified evidence |

The server writes each journal line with an unbuffered append *before* it
sends the response. A trailing newline commits a line. The reader tolerates
exactly one torn final line in a killed lifetime's journal and rejects any
other damage.

## What is measured

All timestamps come from `time.monotonic_ns()` in the experiment process: the
fault injector, the supervisor's observations and every client attempt. Only
differences between these readings are used. Server-process clocks are never
mixed in. Durations are integer nanoseconds. The report adds a rounded
millisecond summary.

- **Detection latency:** from the SIGKILL until the supervisor observes the
  child's exit status. Exit status is polled every `probe_interval_ms`.
- **Restart latency:** from that observed exit until the replacement is
  ready. It includes the declared `restart_backoff_ms`.
- **Client-observed unavailability:** from the first failed attempt that
  finishes after the kill to the first success served by a post-fault
  lifetime. It is null if no client saw a failure before that success.
- **Time to SLO recovery:** from the kill to the end of the first run of
  `slo_consecutive_successes` in-SLO successes, in completion order. An
  in-SLO success started after the replacement was ready, was served by a
  post-fault lifetime and took at most `slo_latency_ms`. Any other attempt
  that finishes in between resets the run.
- **Latency percentiles:** p50, p95 and p99 of successful attempt latencies,
  using the nearest-rank method: sorted samples[ceil(p * n / 100) - 1].
  *Pre-fault* covers successes that finished by the kill. *Post-fault* covers
  successes served by later lifetimes. Successes that the killed server had
  already sent but that arrived after the kill are counted separately as late
  acknowledgements.

The SLO threshold, consecutive-success count, timeouts and backoffs in the
pinned config are declared experiment controls chosen for this lab. They are
not taken from any external latency source or product SLO.

## Reconciliation and at-least-once duplicates

Each logical request ends as `first_try_success`, `success_after_retry` or
`failed_after_budget`. Each committed journal record is matched to the client
attempt (idempotency key and attempt number) that caused it. The following
identities are enforced:

- requests = succeeded + failed
- attempts = the sum of per-request attempts
- executions = acknowledged requests + executions whose response was lost

The server does not deduplicate keys. If a kill lands after a request was
journaled but before its response reached the client, the retry runs it again
on the replacement. That second run is counted as an at-least-once duplicate.
Every acknowledged and executed response for one key must be byte-identical.
The output is a pure function of the prompt, seed, output budget and model
digest, so the verifier also recomputes each expected response body.

In the pinned run the kill fires right after an acknowledgement. The other
in-flight request has usually not been executed yet, so duplicates are often
zero. `tests/test_serving_recovery_duplicates.py` covers the duplicate path
deterministically. It holds a committed response with the server's bounded
`--hold-response-ms` window, SIGKILLs the process inside that window, and
checks that exactly one duplicate execution is reconciled.

## Measurements from the development host

Five runs of the pinned config on one development CPU host, all verified,
measured:

- detection: 4.5 to 8.8 ms
- restart: 124.6 to 134.3 ms, including the 50 ms declared backoff
- client-observed unavailability: 173.0 to 188.5 ms
- time to SLO recovery: 179.2 to 193.0 ms

Each run completed 48 of 48 requests, 2 of them after retries. Each saw
6 `connection_refused` and 2 `connection_reset` attempts and 0 duplicate
executions. Successful-attempt p99 latency stayed below 2 ms before and after
the fault. These numbers describe that host under its load at the time. They
are not a portable benchmark.

## Why a dedicated stdlib server

The existing `ai-journey-ddp serve` path needs a torch distributed checkpoint.
On the development host, importing torch alone took 1.3 to 2.0 s per process.
That would dominate every measured restart. The lab server imports only the
standard library and the lab's stdlib-only modules. On the same host it reached
its readiness handshake in 72 to 102 ms in four of five measured starts; the
first start, with cold file caches, took 343 ms. The package
`__init__` imports torch eagerly. The supervisor therefore starts the server
with `python -I` behind a package stub that loads only the needed submodules.

## Evidence boundary

This evidence comes from one CPU host over 127.0.0.1 loopback. A supervisor
SIGKILLed one local server process running a tiny deterministic
character-bigram model built from a public names fixture, then restarted it.

This is not GPU, CUDA, distributed, multi-host or production serving evidence.
It makes no model-quality, capacity or cost claim, and timings depend on the
host and its load.

## Limitations

- Each run has one fault: one process, one host, loopback HTTP/1.0 with one
  connection per attempt. Network partitions, slow disks, GPU faults and
  multi-replica failover are not exercised.
- Health-probe hang detection is implemented. It is tested by SIGSTOPping a
  child, not exercised by the pinned run. Detection resolution is bounded by
  the probe interval.
- Journal lines survive a process kill, but no per-record fsync is issued.
  Power loss or a kernel crash is outside the durability claim.
- The report seal and the evidence digests prove content identity, not
  authorship or honest collection. Someone who rewrites the raw evidence
  consistently and reseals it is caught only when a derived value stops
  matching. Retain the run directory and compare it with independent runs.
- The idempotency key identifies and reconciles retries. It does not make the
  server exactly-once.
