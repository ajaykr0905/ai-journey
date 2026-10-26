# Distributed GPT-2 CPU lab

This lab trains the existing 7,536-parameter GPT-2 decoder in real, separate
CPU/Gloo processes. It supports deterministic global batches, checkpoint recovery,
document-isolated evaluation, cached inference, and a bounded loopback server.
The checked protocol is `config/ddp-lab.json`; it uses the public CC0 fixture
`data/day-19-demo-names.txt`, bound to its SHA-256 and an immutable repository URL.

The same worker, optimizer, checkpoint and evidence paths also accept schema-two
training specs with a tagged `decoder_variant` model contract. RMSNorm, SwiGLU,
learned positions and RoPE use the shared architecture contracts described in
`docs/gpt2-training-controls.md`. Modern decoder dimensions and the resolved
SwiGLU width count toward the same CPU budgets. Checkpoints bind the entire
architecture, including the RoPE base; equal tensor shapes do not permit resuming
with a different positional contract. Real two-rank regressions check exact
model, optimizer, RNG and progress recovery for modern component combinations.

## Train and resume

Install the package with `python -m pip install -e .`. PyTorch must provide Gloo.
On Linux, a CPU wheel can be installed first with
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

```bash
ai-journey-ddp train --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json --checkpoint /tmp/ddp-demo/state.pt \
  --output /tmp/ddp-demo/training.json --workdir /tmp/ddp-demo/train --steps 2

ai-journey-ddp train --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json --checkpoint /tmp/ddp-demo/state.pt \
  --output /tmp/ddp-demo/resumed.json --workdir /tmp/ddp-demo/resumed \
  --steps 2 --resume

ai-journey-ddp verify --report /tmp/ddp-demo/resumed.json
```

Keep training reports outside the rank workdir and distinct from the source,
specification and checkpoint. Recovery/comparison workdirs must be new or empty.
Artifact admission runs before workers: symlinks, nonregular worker outputs and
hard links to protected inputs or another rank's artifact are rejected without
opening/truncating them.

Each rank receives disjoint shuffled samples and the same number of complete
micro-batches. The effective global batch is
`world_size × micro_batch_size × accumulation_steps`. The protocol uses
`2 × 1 × 2 = 4` sequences, four target bytes per sequence, and four optimizer
updates. Incomplete global batches are discarded explicitly. Epoch and cursor
positions are reconstructed from the durable global optimizer step.

The worker uses DDP `no_sync()` for intermediate micro-batches. The final backward
pass synchronizes gradients, then all ranks agree on gradient validity before
clipping and updating. A rank-local optimizer exception or nonfinite parameter
update restores model, optimizer, RNG, gradients, and module modes. Rank-zero
checkpoint publication happens only after replicated model fingerprints agree.
Publication uses a flushed temporary file, atomic replacement, and directory sync.

Use `train --checkpoint-interval 2` to publish every second completed global
optimizer update and the final update. The interval counts durable optimizer
steps across resume, never micro-batches. Model and optimizer replicas must agree
before an update succeeds or a checkpoint is published; rejected updates restore
the caller's parameters, optimizer settings/moments, RNG, gradients and modes.

The checkpoint contains AdamW moments, the global step, source/config identity,
and every rank's RNG state. Restore requires the same world size and configuration.
Archive expansion is bounded before weights-only deserialization. Recovery covers
completed optimizer boundaries; a failed process can lose work since the last
published checkpoint.

The `launch_training` Python API accepts `failure_rank=1`, `failure_mode="exit"`
and `failure_after_step=3` to test a rank exit after the third optimizer update
but before its checkpoint. With `checkpoint_interval=2`, the checkpoint remains
at step two. Relaunch with `resume=True` and the remaining schedule to recover in
fresh processes. The regression compares final model, AdamW moments, per-rank RNG
and step against an uninterrupted run exactly.

For an actual abrupt process loss, `failure_mode="crash"` makes only the selected
owned worker exit via `os._exit(86)`, without process-group or Python cleanup.
This is distinct from legacy `exit`, which raises a normal exception. An atomic
`rank-N.crash.json` witness records the completed boundary, PID and local monotonic
timestamp before that child exits. `WorkerFailure.supervision` retains observed
exit statuses/timestamps and parent cleanup completion after all owned children
are reaped; `WorkerTimeout` provides the same observations for a deadline. These
are CPU-local fault controls, not machine power-loss or production failover proof.

### Measure abrupt-crash recovery

```bash
ai-journey-ddp recover-crash --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json --output /tmp/crash-proof/report.json \
  --workdir /tmp/crash-proof/owned
ai-journey-ddp verify-crash-recovery --report /tmp/crash-proof/report.json \
  --workdir /tmp/crash-proof/owned
```

This separate, versioned operation leaves clean `recover` evidence unchanged.
It runs an uninterrupted control, abruptly exits rank one after completed step
three, reads the actual last durable checkpoint (step two), and resumes the
remaining two updates in fresh processes. The report computes one lost global
optimizer update from observed progress; it does not assume zero loss. Controls
are explicit: `--steps`, `--checkpoint-interval`, `--failure-after-step`,
`--failure-rank` and `--world-size`, within the existing bounded CPU protocol.

Each resumed rank records a versioned restoration witness after checkpoint
validation and collective schedule agreement, before further updates. The maximum
ready timestamp measures when all ranks have observed restoration. Actual crash
time lies between the child's pre-exit witness and the parent's observed exit;
the reported `recovery_ready_lower_ns`/`upper_ns` retain that RTO interval rather
than pretending the exact exit instant was measured. `restoration_wall_ns`
includes process startup and restore; `resume_completion_wall_ns` additionally
includes replay and remaining training. These local timings are observations,
not deterministic or production availability guarantees.

The report validates actual exit status 86, parent reaping, fresh worker PIDs,
progress, source/specification digests, monotonic chronology and exact final
model/AdamW/RNG/step parity against the control. Offline verification recomputes
loss and timing bounds and rejects resealed type drift. With `--workdir`, it also
recomputes state digests from `continuous.pt`/`recovered.pt`, checks preserved
`last-durable.pt`, and compares crash/restoration witnesses. New crash experiments
retain the exact bounded input bytes as `training-spec.json`, and every worker
loads that witness. Artifact verification hashes those bytes and binds all three
checkpoints' model/training configuration, source digest and world size to the
retained specification/report. An older directory without this witness cannot
prove specification binding and fails artifact verification; the v1 report
fields themselves are unchanged. Without `--workdir`,
verification proves internal report consistency only. A digest cannot authenticate
that timings were collected honestly. Keep the owned artifact directory; no GPU,
elastic-world-size, machine power-loss or production failover claim follows.

## Reproduce the acceptance experiments

The local launcher takes nonblocking advisory leases on both the output directory
and checkpoint path before opening rank logs. A concurrent cooperating launcher
sharing either resource fails without truncating the active run's diagnostics.
The empty `.<checkpoint-name>.ai-journey-ddp.lock` sidecar is deliberately retained;
do not unlink it while a run might own it. Kernel descriptor cleanup releases
ownership after an abrupt supervisor exit, so no PID guessing, foreign-process
signals or artifact deletion is needed. Symlink, hard-link, special-file and
nonempty sidecar collisions fail before workers start. These are local filesystem
advisory locks, not distributed fencing against noncooperating file writers.

```bash
ai-journey-ddp recover --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json --output /tmp/ddp-demo/recovery.json \
  --workdir /tmp/ddp-demo/recovery --steps 2

ai-journey-ddp matrix --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json --output /tmp/ddp-demo/comparison.json \
  --workdir /tmp/ddp-demo/comparison --steps 4

ai-journey-ddp verify-comparison \
  --report artifacts/ddp-lab-2026-10-22/comparison.json

ai-journey-ddp pipeline --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json --output /tmp/ddp-demo/pipeline.json \
  --workdir /tmp/ddp-demo/pipeline --steps 4
```

`recover` compares four uninterrupted updates against two updates followed by two
updates in fresh worker processes. The local acceptance run reproduced model,
optimizer, per-rank RNG, and progress exactly.

`matrix` predeclares three decompositions of the same global batch. All use the
same public bytes, model, seed 48, AdamW settings, schedule, FP64 dtype, and 64 total
target tokens:

| Configuration | Processes | Micro-batch | Accumulation | Max parameter error against serial |
| --- | ---: | ---: | ---: | ---: |
| Serial accumulation | 1 | 1 | 4 | 0 |
| DDP accumulation | 2 | 1 | 2 | 1.6823348270023075e-13 |
| DDP micro-batch | 2 | 2 | 1 | 3.6849689966089727e-13 |

The acceptance threshold is absolute parameter error `1e-10`. The published report
records separate worker PIDs, configurations, per-update metrics, runtime identity,
model fingerprints, and a canonical integrity digest. This digest detects accidental
changes; it is not a third-party signature. Independent accounting checks also
reject resealed invalid reports. CI checks the stored evidence and generates new
reports in its own Linux runtime.

`pipeline` holds out four complete documents before training. Every within-document
validation target is scored once, including short tails. It reports the initial and
trained NLL, loads the verified checkpoint into an immutable serving snapshot, and
benchmarks repeated output parity. The local run's 20 held-out targets changed from
NLL `12.8548239372311` to `11.911954112734469`. Retain regressions in other runs.
This one tiny fixture and seed do not establish generalized quality or scaling.

## Serve the verified checkpoint

```bash
ai-journey-ddp serve --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json --checkpoint /tmp/ddp-demo/state.pt \
  --port 8080 --max-seconds 60
```

The server prints its loopback URL and model fingerprint, then closes at its
declared lifetime. `GET /health` returns readiness and request/token counters.
`POST /generate` returns JSON; `POST /stream` emits ordered `token` events followed
by one `done` event. Both accept this body:

```json
{"request_id":"demo-1","prompt":[97],"sampling":{"max_new_tokens":4,"temperature":0,"seed":48}}
```

The unauthenticated lab server binds only to `127.0.0.1`. Bodies are limited to
16 KiB, prompts to 512 token IDs, output to 128 tokens, and decode work to a
10-second monotonic deadline. Admission uses a refillable token budget. A retry
with the same request ID and input reuses a bounded cached response; a conflicting
input for a retained ID is rejected. Token callbacks stop generation when a stream
disconnects. The library also supports explicit cancellation callbacks.

Cached generation uses model-bound keys/values and absolute position offsets. At
context capacity it rebuilds the sliding window, preserving uncached semantics.
Tests compare all decoder logits and seeded generated tokens across rollover.

```bash
ai-journey-ddp benchmark --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json --checkpoint /tmp/ddp-demo/state.pt \
  --output /tmp/ddp-demo/benchmark.json --repeats 5
```

Warmup requests are excluded, request IDs are fresh to avoid response-cache hits,
and generated output must match across repeats. Throughput uses the full measured
window. Latency percentiles use empirical nearest rank; with five observations,
p95 and p99 are the maximum observation. Reports retain runtime and sample counts.
These short CPU measurements are functional checks, not production capacity claims.

## Failure diagnosis and limits

Rank logs and JSON reports remain in the supplied work directory. A failing or
stalled worker stops the supervisor and all processes it owns. Collective timeouts
are bounded to 30 seconds; the launcher deadline is 60 seconds by default. Tests
inject rank exit, stall, gradient poisoning, schedule mismatch, malformed optimizer
state, and forged evidence. To rerun the complete lab suite:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_ddp*.py
```

Preflight limits are four workers, one million model parameters, context 128,
1 MiB public source, 8,192 global target tokens per update, and 128 updates per
invocation. The dedicated CI job reproduces real process recovery, comparisons,
evaluation, and serving acceptance. NVIDIA multi-GPU training, NCCL, GPU utilization,
elastic world-size recovery, and production failure recovery remain unverified.
Automated output does not certify lesson viewing, manual exercises, or reflection.

The synchronization and accumulation behavior follows the
[PyTorch DDP contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).

## Opt-in immutable checkpoint journal

Programmatic `save_checkpoint(..., journal_directory=Path(...))` publishes a
rank-zero immutable version before replacing the conventional latest checkpoint.
Omitting this argument keeps the single-file behavior and checkpoint schema.
The latest path must be outside the journal. A journal belongs to the exact
source, architecture, tensor dtypes, training schedule and world size; it cannot
be adopted for another run.

`load_checkpoint_journal(directory, model=..., optimizer=..., config=...,
source=...)` returns `JournalRecovery(step, recorded_step, checkpoint_sha256)`.
Every rank hashes and deserializes the **same bounded byte snapshot**, validates
the declared step and complete existing restore contract, and rolls back a
rejected candidate before trying an older one. Transport failure is not treated
as checkpoint corruption. `recorded_step` is only the manifest's high-water mark,
not evidence that the newest version is usable.

The runtime uses the journal's default limits: 16 committed versions, 64 MiB per
blob, and 256 MiB total. It never prunes or deletes history. A full journal rejects
publication. Uncommitted orphan files after an interrupted publication require
explicit inspection; storage integrity alone does not establish restorability.
Manifest publication can become visible before a later directory-fsync error:
an exception does not imply that nothing committed. Inspect the manifest before
retrying. The journal is authoritative when enabled; its commit and replacing
the convenience latest path are not one cross-file atomic transaction.
