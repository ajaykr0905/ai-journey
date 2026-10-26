# Measure the real local HTTP serving path

Start the existing CPU lab inference server with an independently cleared model,
then point the installed command at its loopback port. The measurement client
does not start or replace a service, train a model, or access an external host.

```bash
ai-journey-http-load profile --output artifacts/http-profile.json
# Review the token prompts, generation controls and request budgets first.
ai-journey-http-load measure --endpoint http://127.0.0.1:8080 \
  --profile artifacts/http-profile.json --output artifacts/http-load.json
ai-journey-http-load verify --report artifacts/http-load.json
```

The bounded profile permits at most eight outstanding measured requests, 128
measured attempts, 16 sequential warmup attempts, 32 output tokens per request,
8192 total requested input/output tokens, and a declared 120-second I/O budget.
Each run gets a fresh UUID namespace; warmup and measured IDs are distinct to
avoid response-cache hits. Request prompts and sampling seeds follow the saved
deterministic schedule. Timings themselves are measured and are not deterministic.

Every attempt remains in the evidence, including admission/connection rejection,
HTTP errors, invalid streams, transport failure, deadline expiry and deliberate
client cancellation. Warmup is retained separately. Successful-output throughput
uses completed output tokens divided by the actual measured work window,
including dispatch, connection setup, queueing, failures and thread cleanup.
It never divides by a sum of per-request latencies or omits failed attempts from
the attempted-request denominator.

Time to first token includes connection setup and queueing. Inter-token gaps
use client receipt timestamps of complete SSE events; events delivered in one
socket read share a timestamp and may produce zero gaps. These are not server
compute timings. JSON response mode has no token-arrival timing. Cancellation
means the client stopped receiving; it does not prove that server compute stopped.
For matched streaming/JSON output checks with distinct fresh IDs, the Python API
provides `measure_http_output_parity`.

Reports retain the model identity, protocol/profile controls, raw monotonic
timestamps, token ids and classifications. Verification is offline: it checks
schema, schedule, identities, chronology, concurrency, work coverage and digest,
then independently recomputes counts, token totals, throughput and nearest-rank
p50/p95/p99 distributions. Empty successful distributions remain null. A digest
detects content changes; it cannot authenticate that a timing observation was
actually collected. Preserve source evidence and inspect the local environment.
This is a CPU loopback experiment, with no GPU or production-scale claim.

Writers publish complete new files atomically and refuse to overwrite reports.
The commands add no dependencies, perform no retries, and make no network calls
when verifying an existing report.
