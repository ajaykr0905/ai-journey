# Bounded CPU serving operations

Install the package and create a distributed checkpoint with the pinned public
fixture. Keep the source and specification used to create the checkpoint:

```bash
ai-journey-ddp train --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json --checkpoint /tmp/serving-lab/state.pt \
  --output /tmp/serving-lab/training.json --workdir /tmp/serving-lab/ranks --steps 2
ai-journey-ddp serve --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json --checkpoint /tmp/serving-lab/state.pt \
  --port 8080 --max-seconds 60 --max-connections 8 --drain-seconds 10
```

The loader verifies schema, topology, progress, source/specification binding,
tensor layouts and digests, plus optimizer controls and moments before exposing
a frozen model snapshot. Baseline and opt-in RMSNorm/SwiGLU/RoPE specifications
use the same pipeline. This loader accepts distributed lab checkpoints, not the
different local-training checkpoint format.

The unauthenticated server binds only to `127.0.0.1`. It is not an Internet-facing
deployment. Accepted connections are bounded before worker creation; overload
returns 503 with `Retry-After`. The generation token bucket separately returns
429. Invalid requests return 400, including JSON integers too large for finite
floating-point sampling controls; they cannot consume generation tokens or an
inference slot. The measurement client rejects the same controls before opening
a connection. Internal generation failures return a generic
500, and a failure after streaming begins emits one terminal `error` SSE event.
Requests have a ten-second generation deadline and a five-second socket timeout.
An absolute five-second input budget also covers request headers and body;
trickling bytes cannot renew it. Each raw socket receive uses the remaining
absolute budget, so expiry interrupts input without shutting down its read side;
incomplete input cannot enter generation. The server flushes a 408 response,
half-closes its write side and drains at most 16 KiB for at most 100 ms before
releasing the connection slot. This bounded grace protects the response from a
TCP reset caused by bytes already in flight; it is not an unlimited drain for
uncooperative senders. `make_server` accepts an explicit
`receive_timeout_seconds` in `(0, 30]` for bounded local operating checks. This
input budget ends after receipt and does not shorten an admitted decode deadline.
The service separately bounds running-plus-waiting generation calls to eight by
default (`GenerationService(max_inflight_requests=...)`, maximum 64); a full
inference queue returns 503 with `Retry-After` without charging the token bucket.
Queued cancellation and deadline expiry release their admitted slot without
waiting for the active decode lock or charging generation work. This does not
preempt a running model kernel.

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/ready
curl --fail http://127.0.0.1:8080/metrics
curl --no-buffer -H 'Content-Type: application/json' \
  -d '{"request_id":"example-1","prompt":[97],"sampling":{"max_new_tokens":4,"temperature":0}}' \
  http://127.0.0.1:8080/stream
```

`/health` remains a liveness response during drain. `/ready` switches to 503 as
soon as drain starts. New generation requests then receive 503 while admitted
requests may finish within the declared drain timeout. Long model kernels are
not preempted; deadlines are checked between tokens. Do not interpret the drain
timeout as a guarantee that an arbitrary hung kernel has terminated.

The response cache binds request IDs to the complete input and the frozen model.
Reusing an ID with different input is invalid. Cached token replay still honors
cancellation/deadlines; interrupted replay never replaces the complete cached
response. A fresh ID is required when measuring new generation work.

Prometheus metrics record service-call outcomes, durations, cache hits and newly
generated tokens. Cached replay does not count as new model computation. HTTP
parse failures and connection-limit rejections occur before a service call and
are not included in those counters. No prompt or request-ID labels are retained.
Valid W3C `traceparent` values receive a new child span ID in response headers;
invalid or ambiguous parents start fresh context. Baggage and opaque extensions
are not propagated. This is context propagation, not an OpenTelemetry exporter.

The CI gate runs real local Gloo workers, modern cached/full-forward parity,
HTTP error/overload/shutdown checks and concurrent metrics tests. Results are
CPU fixture evidence only: no production, GPU-throughput or distributed-scale
serving claim follows from them.
