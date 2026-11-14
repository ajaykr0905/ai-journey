# AI Journey: Reproducible Transformer Training and Serving Lab

This repository is a public-safe, executable record of an added setup day plus the
selected verified code through Day 67 of a 90-day AI engineering learning plan. It combines
short study notes with small, deterministic Python exercises that can be reviewed
and rerun.

The schedule is adapted from the user-provided **AI Engineer 90-Day Tracker**
workbook. Workbook entries are planning inputs, not proof that lectures or manual
work were completed. See [`PROGRESS.md`](PROGRESS.md) for the evidence status.

**Learning goal:** build and explain reliable AI systems from first principles,
then grow these exercises into production-quality training and inference projects.
**Target role:** AI infrastructure / ML systems engineer. **Start date:**
2026-09-05.

## What is included

- A reproducible 500-example SFT dataset builder with an immutable public source,
  explicit CC BY-SA attribution, category-balanced selection, assistant-only ChatML
  spans, leakage-free partitions, sealed artifacts, exact source replay, and an
  installed CI gate in the [SFT dataset guide](docs/sft-dataset-construction.md)
- A replayable Project 1 MinHash/LSH deduplicator with stable word shingles,
  seeded signatures, exact Jaccard acceptance, transitive clusters, deterministic
  representatives, removal metrics, sealed artifacts, source-byte replay, and an
  installed CI gate in the [MinHash deduplication guide](docs/minhash-deduplication.md)
- A bounded 100-page HTML-to-text path with public URL validation, explicit
  charset evidence, boilerplate suppression, per-page quality outcomes, atomic
  publication, and independent verification in the
  [HTML extraction guide](docs/html-extraction.md)
- A bounded scaling experiment protocol with immutable run identities, complete
  replicate grids, held-out-model evaluation, honest extrapolation gates,
  tamper-evident reports, and an explicitly non-measured reference fixture in the
  [Day 62 scaling experiment guide](docs/scaling-experiment-protocol.md)
- A content-private serving experiment protocol with immutable workload contracts,
  deterministic traces, exact request and recovery metrics, regression gates, and
  tamper-evident reports in the
  [serving reliability guide](docs/serving-reliability.md)
- A preregistered 100M-parameter scaling plan with exact token and 6ND compute
  targets, optimizer-step alignment, corpus-reuse warnings, IsoFLOP allocation,
  sealed replay, installed CI verification, and explicit non-execution boundaries
  in the [scaling-law planning guide](docs/scaling-law-planning.md)
- An exact 7B/32k/batch-16 KV-cache and inference-capacity plan, including GQA,
  paged allocation, prefill/decode work, continuous batching, speculative
  decoding, and sealed verification, documented in the
  [inference capacity guide](docs/inference-capacity-planning.md)
- A checksum-pinned public-data build with deterministic cleaning, explainable
  quality rejection, deduplication, leakage-safe splits, balanced shards,
  train-only BPE compression evidence, atomic publication, independent replay,
  and installed CI acceptance in the
  [Day 59 data pipeline guide](docs/day-59-data-pipeline.md)
- Explicit LoRA adapters, gradient audits, dense inference export, and bounded
  JSON checkpoints in the [adapter operating guide](docs/lora-adapters.md)
- Deterministic GPT-2-style gradient accumulation, AdamW grouping, warmup and
  cosine scheduling, clipping, rollback, exact checkpoint resume, and fresh-run
  evidence replay in the [training controls guide](docs/gpt2-training-controls.md)
- Real CPU/Gloo process training, exact checkpoint recovery, controlled global-batch
  comparisons, held-out evaluation, cached inference, and loopback serving in the
  [distributed lab runbook](docs/distributed-cpu-lab.md)
- Exact ZeRO 0–3 state ownership, stage-specific collective schedules, sealed
  analytical comparisons, and executed CPU/Gloo collective equivalence in the
  [ZeRO stage runbook](docs/zero-stage-comparison.md)
- Checksum-pinned HellaSwag ingestion, completion-only likelihood scoring,
  calibration, uncertainty, exact sharding, and sealed paired comparisons in the
  [bounded evaluation runbook](docs/hellaswag-evaluation.md)
- A checksum-pinned LAMBADA last-word evaluator with contamination checks,
  content-private score records, deterministic uncertainty, paired comparisons,
  and a model-identity-gated three-axis suite in the
  [evaluation harness guide](docs/evaluation-harness.md)
- Grouped preference cross-validation with response-only DPO-style scoring,
  matched full/LoRA arms, negative findings, and exact replay in the
  [preference optimization guide](docs/preference-optimization.md)
- Day-by-day notes in [`days/`](days/)
- Reusable Python 3.11 code in [`src/ai_journey/`](src/ai_journey/)
- One command to run Days 0–11 in [`scripts/run_days_00_11.py`](scripts/run_days_00_11.py)
- A validated GPT tensor-shape diagram generator in
  [`scripts/run_day_12.py`](scripts/run_day_12.py)
- A deterministic NumPy attention experiment in
  [`scripts/run_day_13.py`](scripts/run_day_13.py)
- A deterministic MLP memory and superposition experiment in
  [`scripts/run_day_14.py`](scripts/run_day_14.py)
- A scalar reverse-mode autodiff engine and independent gradient check in
  [`scripts/run_day_15.py`](scripts/run_day_15.py)
- An edge-by-edge manual backprop trace with three-way gradient verification in
  [`scripts/run_day_16.py`](scripts/run_day_16.py)
- A deterministic scalar MLP trainer with parameter gradient probes in
  [`scripts/run_day_17.py`](scripts/run_day_17.py)
- A deterministic character bigram count model, corpus manifest, and model smoke
  test in [`scripts/run_day_19.py`](scripts/run_day_19.py)
- A deterministic neural bigram model with exact smoothed-count equivalence,
  negative log-likelihood, and gradient checks in
  [`scripts/run_day_20.py`](scripts/run_day_20.py)
- A CI-ready corpus drift gate with explicit divergence and perplexity limits in
  [`scripts/check_corpus_shift.py`](scripts/check_corpus_shift.py)
- A boundary-safe context-window dataset and deterministic embedding MLP in
  [`src/ai_journey/context_mlp.py`](src/ai_journey/context_mlp.py)
- Leakage-safe train/development/test model selection, learning-rate sweeps, and
  overfitting diagnostics in
  [`src/ai_journey/model_selection.py`](src/ai_journey/model_selection.py)
- A deterministic PyTorch decoder-only transformer, exact restart checkpoints,
  held-out evaluation, and an exact-size overfit capacity gate in
  [`src/ai_journey/transformer_lab.py`](src/ai_journey/transformer_lab.py)
- Controlled transformer initialization comparisons with activation and gradient
  histograms, distribution-health checks, and update-to-parameter ratios in
  [`src/ai_journey/activation_experiment.py`](src/ai_journey/activation_experiment.py)
- A controlled fixed-normal versus fan-in Kaiming experiment with initialization
  audits, matched loss curves, held-out evaluation, and evidence gates in
  [`src/ai_journey/initialization_comparison.py`](src/ai_journey/initialization_comparison.py)
- A scratch BatchNorm implementation with persistent running statistics,
  checkpoint-safe transformer integration, and a controlled train/eval mode-trap
  experiment in [`src/ai_journey/batch_normalization.py`](src/ai_journey/batch_normalization.py)
- A stable manual categorical loss, finite-difference and PyTorch gradient checks,
  and a parameter-wide transformer audit in
  [`scripts/run_day_28.py`](scripts/run_day_28.py)
- A strict one-variable ablation protocol with paired deterministic trials,
  negative-result-preserving interpretation, and self-verifying evidence in
  [`scripts/run_ablation.py`](scripts/run_ablation.py)
- A hierarchical character model built from registered PyTorch module containers,
  with runtime shape traces, exact restart checkpoints, seeded sampling, and
  self-verifying evidence in [`scripts/run_day_32.py`](scripts/run_day_32.py)
- An independent rebuild of the hierarchical character model from raw parameters
  and tensor operations, with reference equivalence, numerical gradient audits,
  exact restart, and a bounded capacity gate in
  [`scripts/run_day_33.py`](scripts/run_day_33.py)
- A bounded reliability certification for the rebuild covering parameter,
  activation, gradient, update, reproducibility, sampling, footprint, and
  tamper-evident report contracts in
  [`scripts/certify_day_33.py`](scripts/certify_day_33.py)
- A checksum-pinned Tiny Shakespeare character bigram baseline with exhaustive
  held-out evaluation, exact restart, deterministic sampling, and tamper-evident
  evidence in [`scripts/run_day_34.py`](scripts/run_day_34.py)
- Four equivalent causal-prefix averaging methods with gradient, no-future-leak,
  padding, streaming, restart, and tamper-evident evidence in
  [`scripts/run_day_35.py`](scripts/run_day_35.py)
- A checksum-pinned, scale-limited Tiny Shakespeare transformer run with bounded
  token partitions, exact restart checkpoints, deterministic sampling, runtime
  provenance, and tamper-evident evidence in
  [`src/ai_journey/day38_training.py`](src/ai_journey/day38_training.py)
- Unit tests in [`tests/`](tests/)
- A lossless multilingual BPE tokenizer, verified public-data training pipeline,
  durable restart, strict UTF-8 cached serving and parity-gated benchmarks;
  reproduce them using [`docs/bpe-training-serving.md`](docs/bpe-training-serving.md)
- A matched-seed 18-decoder vocabulary experiment with document-isolated
  bits-per-byte scoring, analytical no-learning baselines, paired uncertainty,
  and exact source-backed replay in [`docs/bpe-model-comparison.md`](docs/bpe-model-comparison.md)
- A deployment-readiness checker in
  [`tools/deployment_readiness_check.py`](tools/deployment_readiness_check.py)
- GitHub Actions CI in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/run_days_00_11.py
python scripts/run_day_12.py --output artifacts/day-12-gpt-shapes.md
python scripts/run_day_13.py --output artifacts/day-13-attention.md
python scripts/run_day_14.py --output artifacts/day-14-mlp-memory.md
python scripts/run_day_15.py --output artifacts/day-15-scalar-autodiff.md
python scripts/run_day_16.py --output artifacts/day-16-manual-backprop.md
python scripts/run_day_17.py --output artifacts/day-17-scalar-mlp.md
python scripts/run_day_19.py \
  --output artifacts/day-19-bigram.md \
  --json-output artifacts/day-19-bigram.json \
  --manifest-output artifacts/day-19-corpus-manifest.json
python scripts/run_day_20.py \
  --output artifacts/day-20-neural-bigram.md \
  --json-output artifacts/day-20-neural-bigram.json
python scripts/check_corpus_shift.py \
  --baseline data/day-19-demo-names.txt \
  --candidate data/day-21-indian-cities.txt \
  --max-js-divergence 0.05 \
  --max-perplexity-ratio 2.5 \
  --output artifacts/corpus-shift.json
python scripts/train_context_mlp.py \
  --corpus data/day-19-demo-names.txt \
  --steps 100 \
  --seed 22 \
  --output artifacts/context-mlp.json \
  --checkpoint artifacts/context-mlp-checkpoint.json
python scripts/run_day_23.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-23-model-selection.json \
  --checkpoint artifacts/day-23-context-mlp.json \
  --plot artifacts/day-23-learning-rate-sweep.svg
python scripts/run_day_24.py \
  --corpus data/day-19-demo-names.txt \
  --steps 100 \
  --output artifacts/day-24-transformer.json \
  --checkpoint artifacts/day-24-transformer.pt
python scripts/check_day_24_overfit.py \
  --corpus data/day-19-demo-names.txt \
  --examples 100 \
  --output artifacts/day-24-overfit.json
python scripts/run_day_25.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-25-diagnostics.json \
  --plot artifacts/day-25-activations.svg \
  --gradient-plot artifacts/day-25-gradients.svg
python scripts/run_day_26.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-26-kaiming.json \
  --loss-plot artifacts/day-26-loss-curves.svg \
  --audit-plot artifacts/day-26-initialization.svg
python scripts/run_day_27.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-27-batchnorm.json \
  --plot artifacts/day-27-batchnorm.svg
python scripts/run_ablation.py \
  --protocol config/day-31-learning-rate-ablation.json \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-31-ablation.json
python scripts/run_day_32.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-32-wavenet.json
python scripts/run_day_33.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-33-wavenet-rebuild.json
python scripts/certify_day_33.py \
  --corpus data/day-19-demo-names.txt \
  --output artifacts/day-33-certification.json
ai-journey-day-34 \
  --corpus artifacts/tinyshakespeare.txt \
  --download \
  --output artifacts/day-34-bigram.json \
  --checkpoint artifacts/day-34-bigram.pt
ai-journey-day-35 \
  --output artifacts/day-35-causal-average.json
ai-journey-day-38 \
  --corpus artifacts/tinyshakespeare.txt \
  --download \
  --output artifacts/day-38-transformer.json \
  --checkpoint artifacts/day-38-transformer.pt
ai-journey-gpt2-train verify \
  --report artifacts/gpt2-training-2026-10-21/training-evidence.json
ai-journey-gpt2-train replay \
  --source data/day-19-demo-names.txt \
  --report artifacts/gpt2-training-2026-10-21/training-evidence.json
python -m unittest discover -s tests -v
```

The GPT-2 training report records the exact Python, PyTorch build, operating
system, machine architecture, CPU capability, and thread count. Replay requires
that environment because floating-point kernels can differ across platforms;
`verify` remains portable and checks the published report without re-execution.

Day 64 wraps this exact training path in a hash-chained lifecycle journal and a
content-addressed final bundle. The [Project 1 run operator](docs/project1-run-operator.md)
documents the checked-in CPU fixture, installed commands, artifact layout, and
the strict boundary between automated run evidence and learner or GPU evidence.

## Compare opt-in decoder architectures on CPU

The GPT-2 weight-interoperable model remains unchanged. A separate decoder can
select RMSNorm, SwiGLU, and RoPE independently or together, then compare all
five named arms on identical byte windows and a separate held-out file:

```bash
mkdir -p artifacts
ai-journey-decoder-compare \
  --train data/day-19-demo-names.txt \
  --heldout data/day-21-indian-cities.txt \
  --output artifacts/decoder-comparison.json
ai-journey-decoder-verify \
  --report artifacts/decoder-comparison.json \
  --train data/day-19-demo-names.txt \
  --heldout data/day-21-indian-cities.txt
```

Supply only independently cleared public data. The example is a bounded
pipeline smoke test: its tiny, different-domain inputs and one-step default
cannot support model-quality or generalization claims. The JSON records input
hashes, matched-window digest, architecture fingerprints, parameter counts,
and training/held-out loss; verification reruns the CPU experiment with the same
PyTorch version and compares every recorded value. Neither command runs on a
GPU or demonstrates distributed or production-scale behavior. Increase steps
only within the CLI's explicit resource limits, and keep the training and
held-out byte files disjoint.

## Train the Day 38 full transformer

The installed Day 38 command verifies the pinned Tiny Shakespeare bytes, keeps
the ordered train and validation partitions disjoint, and trains the complete
decoder stack with attention, feed-forward layers, residual paths, LayerNorm,
and dropout. Default work is explicitly bounded to 65,536 training tokens,
8,192 held-out tokens, 80 updates, and a 30,656-parameter CPU model. The final
sample is replayed from the published restart checkpoint, while the report
records exact configs, losses, token counts, runtime, model/source fingerprints,
and scope limitations.

Two independent local runs on 2026-10-12 produced the same model fingerprint,
losses, update trace, and 201-token sample. The measured run reduced training
NLL from 4.184218 to 2.575125 and held-out NLL from 4.181267 to 2.710894 in
about 10 seconds per run on an Apple-silicon CPU with five PyTorch intra-op
threads. These are bounded diagnostic measurements, not a claim of language
quality, GPU performance, distributed scale, production readiness, lecture
completion, or Ajay's personal exercise completion. See
[`days/day-38.md`](days/day-38.md) for the operating contract and remaining
learner evidence.

## Verify causal prefix averaging

The Day 35 command compares the explicit prefix loop, normalized
lower-triangular matrix multiplication, masked softmax over zero logits, and a
linear-memory cumulative implementation. It also checks input-gradient parity,
weight invariants, future-token isolation, variable-length padding, chunked
streaming, exact stream restart, and the self-verifying report contract.

The experiment is a deterministic CPU float64 numerical audit. It demonstrates
the masking mechanism that precedes learned attention, not learned query-key
scores, language-model quality, GPU performance, or distributed execution. See
[`days/day-35.md`](days/day-35.md) for the operating limits and learner evidence
that still requires Ajay's own confirmation.

## Certify transformer attention and inference restart

```bash
ai-journey-day-37 --output artifacts/day-37-certification.json \
  --protocol config/day-37-head-count-ablation.json \
  --corpus data/day-19-demo-names.txt \
  --ablation-output artifacts/day-37-head-count.json
```

Independent per-head forward/backward checks cover all attention projection
parameters. The command also verifies padding isolation and exact cached decode
through learned-position rollover. Three predeclared head counts use matched
seeds and batches; four updates per trial provide a reproducibility check with
supporting, contradicting or inconclusive findings, not a model-quality claim.
See [the operating guide](docs/transformer-attention-serving.md) for pre/post-norm,
SDPA, activation checkpointing, cache restart, and the CPU-only scope.

## Measure cached generation on a bounded CPU control

```bash
mkdir -p artifacts
ai-journey-serving-benchmark --output artifacts/serving-benchmark.json
```

The installed command compares full-context and cached greedy generation using
a seeded, untrained float32 CPU model and synthetic token IDs. Every warmup and
measured output must match exactly. The report records per-call samples, median,
nearest-rank p95, generated-token throughput, controls and model/prompt/output
hashes. Warmups are excluded; complete-call timings include validation, cache
hashing and context rollover. Calls always run full then cached, and a three-run
p95 is just the largest sample, not a stable tail-latency estimate.

The output directory must exist. CLI limits are 1–32 new tokens, 1–20 repeats,
0–5 warmups, batch size 1–4 and 1–128 prompt tokens. Timing varies by machine and
load; cached generation may be slower. This is not trained-model quality,
production throughput, GPU or distributed evidence. Existing output is replaced
only after the complete JSON report is flushed successfully.

## Train the Tiny Shakespeare baseline

The Day 34 command fetches the exact checksum-pinned public corpus, trains a
seeded character bigram model, evaluates every consecutive pair in the ordered
held-out suffix, writes a complete restart checkpoint, and publishes a
self-verifying JSON report. Re-running with the same source and controls produces
the same update trace, sample, and model fingerprint. Interrupted training can
restore the model, AdamW state, and sampling RNG without changing later updates.

The corpus is downloaded rather than vendored. This remains a small CPU baseline,
not a transformer-quality, GPU, distributed, or production claim. See
[`days/day-34.md`](days/day-34.md) for the evidence contract and learner work that
still requires Ajay's own confirmation.

## Resume transformer checkpoints safely

The Day 24 training command accepts `--resume` with the same corpus and
configuration. Saves flush a unique temporary archive before replacing the
checkpoint; serialization, flush and replacement errors preserve the previous
file. Concurrent saves publish a complete archive from one writer, with no
ordering guarantee. These checks cover filesystem errors, not power-loss recovery.

Save and restore reject nonfinite model and optimizer state. AdamW moments must
match their parameters, variance estimates must be non-negative, and step
counters must be non-negative integers. A rejected restore preserves the caller's
model, optimizer, batch position and CPU RNG. No running training thread may
mutate these objects during save or restore.

## Verify the BatchNorm reverse pass

```bash
python scripts/check_batchnorm_backward.py --output artifacts/batchnorm-backward.json
```

The handwritten reverse pass exposes ten intermediate/leaf gradients, compared
with CPU float64 autograd. Inputs, scale and bias also pass centered finite
differences. Tests compare against native PyTorch BatchNorm and cover constant
features, negative/zero scales and arbitrary upstream gradients. This checks
training-mode biased variance; it does not model running-statistic updates or
claim that the learner completed the manual derivation. Exit codes are 0 for a
passing audit, 1 for failed tolerances and 2 for invalid arguments. Invalid inputs
leave an existing report untouched.

## Verify the loss derivative

```bash
python scripts/run_day_28.py --output artifacts/day-28-gradient-audit.json
```

The JSON report compares handwritten mean cross-entropy gradients with PyTorch
autograd and centered finite differences, including tied maxima and underflowing
target probabilities. It also compares every parameter gradient in the existing
small transformer using manual versus native loss derivatives. Model derivatives
still use PyTorch. The fixed synthetic token batch is CPU float64 diagnostic
evidence, not public-corpus training, GPU performance, or production evidence.

Exit code 0 means all declared tolerances passed; 1 means a numerical gate failed;
2 means an argument is invalid. Failed numerical gates retain their diagnostic
report. Invalid arguments leave an existing report untouched. Reports are
published with an atomic file replacement. Set `--seed`, `--epsilon`,
`--tolerance`, and `--transformer-tolerance` explicitly when changing the experiment.

PyTorch is required for the Day 24–28 transformer experiments. The default verification
commands run on CPU and do not claim GPU or distributed execution.

## Run a controlled ablation

The checked-in Day 31 protocol declares its hypothesis, primary metric, minimum
effect, fixed model/training controls, three learning-rate arms, and paired seeds
before execution. The loader rejects unknown or duplicate fields and any arm that
changes more than the named independent variable. Reports retain supporting,
contradicting, and inconclusive outcomes rather than turning a negative result into
a failed run. Canonical fingerprints bind the protocol, corpus, measurements, and
interpretation; they detect accidental or manual report changes but are not digital
signatures.

The default protocol is a small CPU reproducibility gate over the public demo corpus.
It is not evidence of GPU performance, model quality, statistical significance, or
the learner's blank-file backward-pass completion. Increase steps and seed count in
a new predeclared protocol before drawing research conclusions.

## Configuration safety

Repository experiments require no credentials. Never commit `.env`, credentials, browser data,
cookies, or access tokens. See [`SECURITY.md`](SECURITY.md).

## Plan distributed training capacity

Build the pinned 16-rank analytical plan and then independently recompute every
derived field and its SHA-256 integrity digest:

```bash
ai-journey-distributed-capacity build \
  --config config/day-55-distributed-capacity.json \
  --output artifacts/day-55-distributed-capacity.json
ai-journey-distributed-capacity verify \
  --report artifacts/day-55-distributed-capacity.json
```

The report inventories exact parameter shards, persistent parameter/gradient/
optimizer state, forward all-gather peaks, gradient bucket communication, and
full or sharded checkpoint capacity. Checkpoint write, retention, and recovery
calculations use the rates declared in the config. They are planning estimates:
the command does not launch ranks, allocate a GPU, execute a collective, access
remote storage, or claim measured throughput. Change bandwidth, latency,
topology, dtype, and storage values only when recording a new explicit scenario;
do not reinterpret the pinned assumptions as observations of local hardware.

## Inject serving failures and measure recovery

Kill a real loopback inference server under bounded client load, restart it,
and verify what the clients observed. Use a run directory that does not exist
yet; the runner refuses to reuse one or overwrite evidence:

```bash
ai-journey-serving-recovery run \
  --source data/day-19-demo-names.txt \
  --config config/day-70-serving-recovery.json \
  --run-dir /tmp/day-70-serving-recovery
ai-journey-serving-recovery verify \
  --run-dir /tmp/day-70-serving-recovery
```

Both commands print the same JSON receipt. `verify` recomputes every identity,
digest and derived metric from the raw attempts, per-lifetime journals and
hash-chained trace. The wall-clock run duration has no independent raw record,
so `verify` only checks that it covers the traced lifecycle and stays within
`max_run_ms`. Exit code 0 means verified and recovered, 1 rejected evidence,
2 invalid input or an incomplete run, and 3 verified but not recovered within
the declared bounds.

Detection, restart, client-observed unavailability, time to SLO recovery and
nearest-rank p50/p95/p99 latency are real measurements from one monotonic clock
on the host that runs `run`, so they change with that host and its load. The
SLO, timeout, retry and backoff values are declared lab controls, not external
targets. Evidence boundary: one CPU host, 127.0.0.1 loopback, and a tiny
deterministic character-bigram model. This is not GPU, CUDA, distributed,
multi-host or production serving evidence and makes no model-quality, capacity
or cost claim. See the
[serving failure recovery guide](docs/serving-failure-recovery.md) for the run
directory, metrics, reconciliation and limitations.

## Model SIMT execution limits analytically

Build the pinned SIMT execution-model report from declared NVIDIA inputs, then
recompute every field independently:

```bash
ai-journey-simt build \
  --config config/day-71-simt-execution-plan.json \
  --output /tmp/day-71-simt-report.json
ai-journey-simt verify \
  --report /tmp/day-71-simt-report.json
```

Both commands print the same SHA-256 digest with the counts `devices=3`,
`sweep_cells=40`, `memory_rows=18` and `arithmetic_rows=12` and the flags
`measured=false gpu_executed=false`. They return 0 on success and 1 for an
invalid plan, report or path; `build` refuses to write over its plan. The report
covers CUDA launch limits, 1D and 2D index coverage with masked lanes,
theoretical occupancy under the `cuda_occupancy.h` allocation rules,
Little's-law latency hiding, wave quantization, and FP32/DRAM roofline balance
for the V100 SXM2, T4 and A100 SXM4 80GB.

Everything in the report is analytical. Architecture limits, device specs,
latencies, register budgets and kernel byte and FLOP counts are declared inputs
cited from the CUDA C++ Programming Guide v12.6, CUDA 12.6 `cuda_occupancy.h`,
and NVIDIA datasheets and whitepapers; each device must reproduce its datasheet
FP32 peak. No GPU, CUDA kernel, or latency was executed or measured, so no
number in it is achieved occupancy, bandwidth or run time. See the
[SIMT execution model guide](docs/simt-execution-model.md).

## Model GPU memory access and time CPU matmul baselines

Seal the analytical report, measure the CPU baseline into a separate report,
and verify each against the pinned plan:

```bash
ai-journey-memory-access analyze \
  --config config/day-74-memory-access-plan.json \
  --output /tmp/day-74-memory-access-analysis.json
ai-journey-memory-access verify \
  --report /tmp/day-74-memory-access-analysis.json \
  --config config/day-74-memory-access-plan.json
ai-journey-memory-access benchmark \
  --config config/day-74-memory-access-plan.json \
  --output /tmp/day-74-memory-access-measurement.json
ai-journey-memory-access verify \
  --report /tmp/day-74-memory-access-measurement.json \
  --config config/day-74-memory-access-plan.json
```

`analyze` is analytical and prints `measured=false gpu_executed=false`: per-warp
32-byte sectors, 128-byte lines and coalescing efficiency, shared-memory bank
conflicts for 1-, 2- and 4-byte lanes, a 32 x 32 transpose tile with and
without padding, and naive versus shared-tiled matmul traffic on a roofline
whose A100 80GB SXM FP32 ceilings are declared from NVIDIA's specifications
table. `benchmark` is the only measurement: wall-clock float32 NumPy and torch
matmuls on the CPU of the host that runs it, each checked against a float64
rounding bound and sealed in a separate report with scope
`cpu_wall_clock_this_host`. `verify` recomputes every analytical number
independently and checks the measurement accounting, but cannot re-observe a
timing. Exit code 0 means success, 1 a report that failed verification, and 2
invalid input or a refused output.

No GPU, CUDA kernel, cache, or device timing was executed or measured for the
analytical report, and the CPU timings do not measure or predict GPU, CUDA,
tensor-core, distributed, or production performance. CI benchmarks
`config/day-74-memory-access-ci-smoke.json`, which equals the pinned plan
except for its name and a benchmark of one 32^3 case per backend with 1 warmup
and 3 repeats instead of 2 and 7. See the
[memory-access model guide](docs/memory-access-model.md).

## Measure the CPU memory hierarchy

Measure STREAM bandwidth across working-set sizes, access-pattern costs and a
NumPy roofline on the CPU of the host that runs the command, then verify the
sealed report against the pinned config:

```bash
ai-journey-memory-hierarchy run \
  --config config/day-71-memory-hierarchy.json \
  --output /tmp/day-71-memory-hierarchy.json
ai-journey-memory-hierarchy verify \
  --report /tmp/day-71-memory-hierarchy.json \
  --config config/day-71-memory-hierarchy.json
```

`run` times STREAM copy, scale, add and triad at 12 sweep points (triad's
working set grows from 24 KiB to 192 MiB), copy and sequential, strided and
random `take` gathers over one 64 MiB source, and a 1024^3 float64 NumPy
matmul as the compute ceiling, within a 20-second budget and a 256 MiB
allocation bound. It derives a measured roofline through `roofline.py` and a
Little's-law view from a stated, unmeasured 100 ns latency, prints lines
labeled `cpu_host_measurement`, and ends with the report digest,
`scope=cpu_wall_clock_this_host` and `gpu_executed=false`. `verify`
recomputes every statistic, rate, ceiling and Little's-law product from the
raw samples, but cannot re-observe a timing. Exit code 0 means success, 1 a
report that failed verification, and 2 usage errors, invalid inputs, refused
outputs or failed runs.

Bandwidths count the operand bytes each NumPy call reads and writes, not
hardware traffic, and include NumPy call overhead. The matmul may use several
BLAS threads; the other kernels use one. No cache size is detected or claimed,
and nothing here measures or predicts GPU, CUDA, distributed, or production
performance. CI runs the installed command on
`config/day-71-memory-hierarchy-ci-smoke.json`, the pinned config with fewer
sweep points, smaller arrays and fewer, shorter samples, so its timings are a
smoke check, not a baseline. See the
[memory-hierarchy lab guide](docs/memory-hierarchy-lab.md).

## Project 1 release status

The flagship is a decoder-only transformer training and serving lab with
deterministic checkpoints, public-data GPT-2 interoperability, assistant-only
SFT, LoRA/full-fine-tune comparisons, HellaSwag and LAMBADA evaluation paths,
serving admission controls, and verifiable-reward fixtures. Day 70 adds a real
local CPU worker process kill, measured controller-side detection and recovery,
restart from a sealed checkpoint, and exact lost/replayed work accounting.

Run `ai-journey-recovery run` with
`config/day-70-worker-recovery.json`, then use `ai-journey-recovery verify` on the
result directory. The [Project 1 release guide](docs/project1-release.md) records
the architecture, evaluation surface, operating steps, claim boundary, and
remaining failures.

Recorded external cloud and GPU spend for the checked-in CPU fixtures is **₹0**.
Host electricity, engineering time, and employer-provided compute were not
metered, so total INR cost is unavailable. What went wrong: this environment
still lacks the required real NVIDIA multi-GPU run, GPU utilization and memory
measurements, verified vLLM or SGLang deployment, and a distributed or GPU
serving failure trace. The local recovery measurement is useful process
evidence, not a GPU, distributed-scale, production-readiness, or model-quality
claim.

## Daily map

Run the sealed Day 68 public-data SFT comparison with
`ai-journey-sft-finetune build`, then bind or reproduce it with `verify` and
`replay`. See [the SFT fine-tuning operator guide](docs/sft-finetuning.md) for the
assistant-only masking, deterministic checkpoint, evidence, and interpretation
boundaries.

Run the Day 69 fixture-backed reward and recovery workflow with
`ai-journey-verifiable-rewards build`, then independently reconstruct it with
`verify` or reproduce it with `replay`. The
[verifiable reward guide](docs/verifiable-rewards.md) defines its verifier,
failure-injection, recovery, and evidence boundaries.

| Day | Date | Focus |
|---:|:---:|---|
| 0 | 2026-09-04 | Added repository setup and configuration safety day |
| 1 | 2026-09-05 | LLM lifecycle and learning goal |
| 2 | 2026-09-06 | Python, NumPy, PyTorch, and notebook diagnostics |
| 3 | 2026-09-07 | Gradient descent for `f(x) = x²` |
| 4 | 2026-09-08 | Single sigmoid-neuron derivation |
| 5 | 2026-09-09 | Two-layer backward-pass derivation |
| 6 | 2026-09-10 | Pure-NumPy two-layer forward pass |
| 7 | 2026-09-11 | Manual output-bias gradient |
| 8 | 2026-09-12 | Manual output-layer weight gradients |
| 9 | 2026-09-13 | Full backpropagation and gradient checks |
| 10 | 2026-09-14 | Deterministic XOR training |
| 11 | 2026-09-15 | Python fluency drills |
| 12 | 2026-09-16 | Decoder-only GPT dataflow and tensor shapes |
| 13 | 2026-09-17 | Single-head scaled dot-product attention in NumPy |
| 14 | 2026-09-18 | Transformer MLPs as key/value memories and superposition |
| 15 | 2026-09-19 | Scalar values, computation graphs, and reverse-mode autodiff |
| 16 | 2026-09-20 | Manual add/multiply backprop and chain-rule trace |
| 17 | 2026-09-21 | Scalar neurons, dense layers, and deterministic MLP training |
| 18 | 2026-09-22 | Blank-file micrograd rebuild; learner evidence remains pending |
| 19 | 2026-09-23 | Character bigram counts, broadcasting, and deterministic sampling |
| 20 | 2026-09-24 | Neural bigram NLL, smoothing equivalence, and gradient checks |
| 21 | 2026-09-25 | Dataset shift, cross-corpus evaluation, and record-boundary integrity |
| 22 | 2026-09-26 | Context windows, embeddings, deterministic training, and checkpoint restart |
| 23 | 2026-09-27 | Minibatches, learning-rate search, held-out evaluation, and overfitting diagnostics |
| 24 | 2026-09-28 | Deterministic transformer training, exact restart, and overfit capacity gate |
| 25 | 2026-09-29 | Controlled initialization, activation histograms, and gradient diagnostics |
| 26 | 2026-09-30 | Fan-in Kaiming initialization, scale audits, and matched loss curves |
| 27 | 2026-10-01 | Scratch BatchNorm, persistent running state, and train/eval mode diagnostics |
| 28 | 2026-10-02 | Manual cross-entropy and transformer parameter-gradient audit |
| 31 | 2026-10-05 | Predeclared paired transformer ablation protocol and evidence gate |
| 32 | 2026-10-06 | Hierarchical character model, module containers, and exact restart |
| 33 | 2026-10-07 | Primitive WaveNet rebuild, equivalence audits, and exact resume |
| 34 | 2026-10-08 | Tiny Shakespeare character bigram and training loop |
| 35 | 2026-10-09 | Causal prefix averaging, masked softmax, and streaming equivalence |
| 36 | 2026-10-10 | Single-head forward/backward and causal-isolation foundations verified with Day 37 |
| 37 | 2026-10-11 | Full residual blocks, multi-head audits, controlled configurations, and inference restart |
| 42 | 2026-10-16 | Low-rank adapter primitives, frozen-base audits, dense merge equivalence, paired comparison, and exact replay |
| 43 | 2026-10-17 | Validated chat/tool conversations, assistant-only SFT, matched full/LoRA trials, uncertainty, and exact replay |
| 45 | 2026-10-19 | Exact GPT-2 architecture, public checkpoint import, full-logit parity, and tamper-evident evidence |
| 48 | 2026-10-22 | Real DDP processes, global-batch equivalence, exact recovery, document-held-out evaluation, and bounded serving |
| 49 | 2026-10-23 | Pinned HellaSwag ingestion, completion-only scoring, calibration, uncertainty, sharding, and evidence replay |
| 50 | 2026-10-24 | Source-backed BPE merge replay, bounded audits, Unicode controls, and document-safe streaming |
| 51 | 2026-10-25 | Exact transformer parameter, FLOP, memory, sharding, runtime, and cost planning |
| 55 | 2026-10-29 | Reproducible roofline analysis and distributed capacity, sharding, and checkpoint planning |
| 57 | 2026-10-31 | Exact ZeRO stage planning and executed CPU/Gloo collective conformance |
| 60 | 2026-11-03 | Analytical Chinchilla target, IsoFLOP allocation, corpus sufficiency, and sealed replay |
| 62 | 2026-11-05 | Bounded scaling sweep protocol, held-out extrapolation gates, and sealed reference replay |
| 63 | 2026-11-06 | Held-out likelihood, HellaSwag, and contamination-audited LAMBADA evaluation suite |
| 68 | 2026-11-11 | Public-data assistant-only SFT, matched pre/post evaluation, checkpoints, and exact replay |
| 69 | 2026-11-12 | Verifiable reward groups, deterministic failure injection, exact recovery, and sealed replay |
| 70 | 2026-11-13 | Real local worker kill, measured detection and recovery, exact loss accounting, and Project 1 release boundaries |

## License

The code in this learning repository is available under the MIT License.
