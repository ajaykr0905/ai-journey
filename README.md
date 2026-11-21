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

## Export recovery telemetry as metrics, traces and a dashboard

Derive Prometheus metrics and alerting rules, OpenTelemetry traces and a
Grafana dashboard from one serving-recovery run and one worker-recovery run,
then verify each output against those runs. Every run and output directory
must be new; nothing is overwritten:

```bash
ai-journey-serving-recovery run \
  --source data/day-19-demo-names.txt \
  --config config/day-70-serving-recovery.json \
  --run-dir /tmp/day-73/serving
ai-journey-recovery run \
  --source data/day-19-demo-names.txt \
  --config config/day-70-worker-recovery.json \
  --run-dir /tmp/day-73/worker
ai-journey-telemetry-metrics export \
  --config config/day-73-telemetry-metrics.json \
  --serving-run-dir /tmp/day-73/serving \
  --worker-run-dir /tmp/day-73/worker \
  --output-dir /tmp/day-73/telemetry
ai-journey-telemetry-metrics verify \
  --output-dir /tmp/day-73/telemetry \
  --serving-run-dir /tmp/day-73/serving \
  --worker-run-dir /tmp/day-73/worker \
  --config config/day-73-telemetry-metrics.json
ai-journey-telemetry-traces export \
  --config config/day-73-telemetry-traces.json \
  --serving-run-dir /tmp/day-73/serving \
  --worker-run-dir /tmp/day-73/worker \
  --output-dir /tmp/day-73/traces
ai-journey-telemetry-traces dashboard \
  --config config/day-73-telemetry-traces.json \
  --output-dir /tmp/day-73/dashboard
ai-journey-telemetry-traces verify \
  --report /tmp/day-73/traces/report.json \
  --serving-run-dir /tmp/day-73/serving \
  --worker-run-dir /tmp/day-73/worker \
  --config config/day-73-telemetry-traces.json
ai-journey-telemetry-traces verify \
  --report /tmp/day-73/dashboard/report.json \
  --config config/day-73-telemetry-traces.json
```

Neither run is converted until its own lab verifier accepts it. The metrics
export writes `metrics.prom` in Prometheus text exposition format 0.0.4, 28
families with every series labeled `scope="local_cpu_lab"`, and `rules.yml`
with 3 recording and 11 alerting rules generated from the declared
objectives. The traces export writes `traces.otlp.json`, one OTLP/JSON
request with a trace per run, and `dashboard` writes Grafana `dashboard.json`
(schemaVersion 39, 7 panels, 12 queries) whose queries select only series
the metrics export writes. Each command prints one JSON receipt. `export`
and `dashboard` seal a report and verify what they wrote. `verify` rebuilds
every stored file byte for byte, after re-running the lab verifiers on an
export's runs, and `--config` binds it to the pinned config.

`ai-journey-telemetry-metrics` exits 0 when the export verifies and every
declared objective is met, 1 when `verify` rejects an export or its
evidence, 2 for usage errors, invalid inputs, refused outputs or evidence
that `export` rejects, and 3 when the export verifies but an objective is
breached, so its alert would fire. `ai-journey-telemetry-traces` exits 0 on
success, 1 for rejected evidence or a rejected stored artifact, and 2 for
usage errors, invalid inputs or a refused output.

Evidence boundary: these are offline files derived from sealed local CPU
evidence recorded on one host. No Prometheus server scraped the metrics, no
OpenTelemetry collector received the traces and no Grafana imported the
dashboard. Durations are measured monotonic-clock differences; absolute
timestamps come from a declared anchor and are synthetic. The objectives
are declared lab controls, not service-level agreements, and the measured
timings change with the host and its load. This is not GPU, CUDA,
distributed, multi-host or production monitoring evidence. CI job
`telemetry-cpu` runs these commands from an installed wheel without torch,
NumPy or matplotlib. See the
[telemetry metrics guide](docs/telemetry-metrics.md) and the
[telemetry traces and dashboards guide](docs/telemetry-traces-dashboards.md).

## Simulate CUDA kernel launches and time Metal kernels on an Apple GPU

Two labs write a vector-add kernel and an RGB-to-grayscale kernel. Both labs
use the same exact integer rule for grayscale. The kernel-launch lab
simulates CUDA launches of both kernels in pure Python on the CPU. The Metal
lab compiles both kernels for the Apple GPU of a Mac and times them against
PyTorch. Each `run` seals a report, and each `verify` checks a report against
the pinned config:

```bash
ai-journey-kernel-launch run \
  --config config/day-73-kernel-launch.json \
  --output /tmp/day-73-kernel-launch-report.json
ai-journey-kernel-launch verify \
  --report /tmp/day-73-kernel-launch-report.json \
  --config config/day-73-kernel-launch.json
ai-journey-kernel-metal run \
  --config config/day-73-kernel-metal.json \
  --output /tmp/day-73-kernel-metal.json
ai-journey-kernel-metal verify \
  --report /tmp/day-73-kernel-metal.json \
  --config config/day-73-kernel-metal.json
```

`ai-journey-kernel-launch` runs each kernel thread by thread, in the
canonical order and in three seeded orders. It records each out-of-bounds
access and each data race in the simulated buffers. On the pinned plan, all
8 runs of the guarded kernels match exact references. At ragged extents, the
unguarded kernels make 128 out-of-bounds writes and 232 write conflicts. The
lab also pins CUDA C++ source for both kernels, and 10 structural checks pass
on it without a compiler. Both commands print one line with the report digest
`95b6791cc71fddc96e0e8e45bba60f1f95951921a8b382fbcae383e447ed9d9a`.

`ai-journey-kernel-metal` compiles the kernels at run time through
`torch.mps.compile_shader`. The Metal kernels and their PyTorch equivalents on
the MPS device and on the CPU must each match a NumPy reference bit for bit.
Then the command times the Metal kernels against PyTorch on the same MPS
device. `verify` does not import torch. It rebuilds the correctness results
on the CPU, and it checks the timings for structure only.

In three runs of the pinned config on the development host, all 13
correctness cases were exact. The Metal grayscale kernel was 12.8 to 23.6
times faster than the eager PyTorch equivalent. For vector add, the ratio of
the PyTorch median to the Metal median stayed between 0.85 and 1.24. Neither
side was faster at every point. Later runs, while other jobs used the host,
gave grayscale speedups from 8.1 to 24.3 times and vector-add ratios up to
1.30. These values are host-specific Apple GPU (Metal) measurements from one
shared host, not portable figures.

`ai-journey-kernel-launch` and `ai-journey-kernel-metal` exit 0 on success, 1
when a report fails verification, and 2 for usage errors, unreadable or
invalid inputs, or a refused output. For `ai-journey-kernel-metal`, code 2
also marks a run that failed a check, had a device error or exceeded its time
budget. Code 3 means that the host cannot run Metal kernels through PyTorch.
Then `run` stops before it measures anything.

Evidence boundary: the kernel-launch lab runs no GPU, compiles no CUDA
source and measures no time. Its CUDA C++ text is source for a later run on
an NVIDIA GPU. The Metal timings do not measure or predict CUDA, NVIDIA,
distributed or production performance. No CUDA kernel ran, and the labs used
no NVIDIA GPU.

CI job `kernel-launch-cpu` runs and verifies the pinned plan from an
installed wheel without torch, NumPy or matplotlib. CI job
`kernel-metal-contracts` runs the Metal lab tests on Linux, where the GPU
tests skip. It also requires the installed `run` to exit 3 and write no
report. See the [kernel-launch lab guide](docs/kernel-launch-lab.md) and the
[Metal kernel lab guide](docs/kernel-metal-lab.md).

## Model a naive matmul and compare it with measured Apple GPU ceilings

The Day 74 labs write a naive matmul kernel, time it and compare its rate with
ceilings that the same host measured. Five commands do this work:

- `ai-journey-matmul-model` runs the naive kernel in a pure-Python launch
  simulator on the CPU. It counts the memory requests of each warp, the FLOPs
  and the bytes. It also checks the structure of pinned CUDA C source without
  a compiler.
- `ai-journey-matmul-metal` times two naive Metal kernels and `torch.matmul`
  on the Apple GPU of a Mac.
- `ai-journey-gpu-ceilings` measures an FMA ceiling and a bandwidth ceiling on
  the same GPU.
- `ai-journey-metal-launch-sweep` times the naive kernel at 7 threadgroup
  shapes and vector add at 9 group sizes.
- `ai-journey-matmul-roofline` compares a matmul-metal report with a ceiling
  report and seals the gap between them.

Each command seals a report, and each `verify` checks a report again. These
commands model the kernel and measure its gap on a Mac:

```bash
ai-journey-matmul-model run \
  --config config/day-74-matmul-model.json \
  --output /tmp/day-74-matmul-model-report.json
ai-journey-matmul-model verify \
  --report /tmp/day-74-matmul-model-report.json \
  --config config/day-74-matmul-model.json
ai-journey-matmul-metal run \
  --config config/day-74-matmul-metal.json \
  --output /tmp/day-74-matmul-metal.json
ai-journey-gpu-ceilings run \
  --config config/day-74-gpu-ceilings.json \
  --output /tmp/day-74-gpu-ceilings.json
ai-journey-matmul-roofline gap \
  --matmul-report /tmp/day-74-matmul-metal.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json \
  --output /tmp/day-74-matmul-roofline.json
ai-journey-matmul-roofline verify \
  --report /tmp/day-74-matmul-roofline.json \
  --matmul-report /tmp/day-74-matmul-metal.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json
```

On the pinned plan, all 16 model runs match the reference bit for bit, and all
16 CUDA checks pass. Both model commands print one line with the report digest
`85f0aa8ddefb7e99d0e9de81047663a1e2e3d43332347a51560b1a19b34b1136`.

In this section, "peak" means a ceiling that the gpu-ceilings lab measured on
this host. On the development host, the naive `matmul_x_columns` kernel
reached at most 406 GFLOP/s. That rate is 0.088 of the measured FMA ceiling of
4618 GFLOP/s. `torch.matmul` reached 3963 GFLOP/s, 0.858 of the same ceiling.
At all 9 sizes, both naive kernels ran 4.79 to 16.15 times above the no-reuse
bandwidth roof. Thus caches probably served part of their loads. The labs read
no hardware counters, so they do not measure this traffic.

The 1-minute load averages were 7.98 to 10.72 before and after the two runs.
These values are host-specific Apple GPU (Metal) measurements from one shared
host, not portable figures.

All five commands exit 0 on success and 1 when a report fails verification.
They exit 2 for usage errors, unreadable or invalid inputs, or a refused
output. The gap command also exits 2 for two reports that do not pair. For the
three Metal commands, code 2 also marks a run that failed a check, had a
device error or exceeded its time budget. Code 3 means that the host cannot
run Metal kernels through PyTorch. Then `run` stops before it measures
anything.

Evidence boundary: the model lab runs no GPU and compiles no CUDA source. Its
CUDA C text is source for a later run on an NVIDIA GPU. Each ceiling is a rate
that the ceiling lab's own kernels reached on this host, not a vendor peak.
The Metal timings do not measure or predict CUDA, NVIDIA, distributed or
production performance.

CI job `matmul-model-cpu` runs and verifies the pinned plan from an installed
wheel without torch, NumPy or matplotlib. CI job `metal-roofline-contracts`
runs the tests of the other four labs on Linux, where the GPU tests skip. It
also requires each installed Metal `run` to exit 3 and write no report. See
the [matmul model lab guide](docs/matmul-model-lab.md), the
[matmul-metal lab guide](docs/matmul-metal-lab.md), the
[gpu-ceilings lab guide](docs/gpu-ceilings-lab.md), the
[launch-sweep guide](docs/metal-launch-sweep.md) and the
[gap guide](docs/matmul-roofline-gap.md).

## Profile the naive matmul and name its top bottleneck

The Day 75 labs profile the Day 74 kernels and name a bottleneck from the
measurements. Four commands do this work:

- `ai-journey-sim-profile` profiles the pure-Python launch simulator with
  `cProfile` and times it on the CPU. It seals one snapshot for each
  optimization round.
- `ai-journey-profile-trace` captures `torch.profiler` traces of the naive
  Metal kernels and `torch.matmul`. It parses, analyzes and verifies the
  Chrome trace files.
- `ai-journey-gpu-timing` times each launch on three clocks: MPS events, the
  host wait and the host dispatch.
- `ai-journey-matmul-bottleneck` names a bottleneck for each naive kernel at
  each shape. It reads three Day 74 reports and a one-call timing input.

Each command seals a report, and each `verify` checks a report again. These
commands profile the kernels on a Mac and name the bottleneck:

```bash
ai-journey-sim-profile run \
  --config config/day-75-sim-profile.json \
  --stage round-4 --output /tmp/day-75-round-4.json
ai-journey-profile-trace capture \
  --config config/day-75-profile-trace.json \
  --output-dir /tmp/day-75-profile-trace
ai-journey-profile-trace verify \
  --report /tmp/day-75-profile-trace/report.json \
  --config config/day-75-profile-trace.json
ai-journey-gpu-timing run \
  --config config/day-75-gpu-timing.json \
  --output /tmp/day-75-gpu-timing.json
ai-journey-gpu-timing verify \
  --report /tmp/day-75-gpu-timing.json \
  --config config/day-75-gpu-timing.json
ai-journey-matmul-bottleneck classify \
  --matmul-report /tmp/day-75-matmul-metal-cubes.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json \
  --gap-report /tmp/day-75-matmul-roofline-cubes.json \
  --timing /tmp/day-75-gpu-timing.json \
  --config config/day-75-matmul-bottleneck.json \
  --output /tmp/day-75-matmul-bottleneck-joined.json
```

The `classify` command reads the sealed gpu-timing report. That report must
pass its own verifier first. The bottleneck guide gives the commands for the
cube matmul-metal report and its gap.

On the development host, the profiler trace at 2048 x 2048 x 2048 named
`matmul_x_rows/synchronize` as the top op. It had a share of 0.685 of the
iteration host time. The load averages were 11.00, 12.79 and 12.49 before
that capture.

The gpu-timing lab measured an empty-kernel event median of 64.9 us and 69.6
us in two runs. At 4096 cubed, `torch.matmul` reached 0.834 and 0.850 of the
4618 GFLOP/s FMA ceiling that Day 74 measured. The load averages were 9.87,
13.62 and 12.65 before run 1.

The joined bottleneck report named `latency or occupancy-bound` as the top
name of each naive kernel, at 4 of 6 shapes. At 128 cubed, both kernels were
`dispatch-bound`. The load averages were 8.90, 8.80 and 10.84 before the
joined run. All values in these three paragraphs are host-specific Apple GPU
(Metal) measurements from one shared host, not portable figures.

On the CPU, round 3 of the simulator profiling gave the only repeatable
gain, 1.29 to 1.40 in each run. The sealed sessions gave overall speedups of
2.079 and 1.508 at a 1-minute load of 14.7 to 18.4.

All four commands exit 0 on success and 1 when a report fails verification.
They exit 2 for usage errors, unreadable or invalid inputs, or a refused
output. The bottleneck command also exits 2 for inputs that do not pair.
Code 3 means that the host cannot run Metal kernels through PyTorch. Then
`capture` or `run` stops before it measures anything.

Evidence boundary: each bottleneck name is an inference from timings and
from the ceilings that Day 74 measured on the same host. The labs read no
hardware counter and used no Nsight and no Instruments. `torch.profiler`
records host spans only, so GPU work shows as host wait time. The timings
do not measure or predict CUDA, NVIDIA, distributed or production
performance.

CI job `profile-sim-cpu` runs the trace and sim-profile tests without torch.
It also runs and verifies the pinned sim-profile stage from an installed
wheel. CI job `gpu-timing-contracts` runs the timing and bottleneck tests on
Linux with CPU torch. It also requires each installed Apple GPU command to
exit 3 and write no report. See the
[simulator profiling guide](docs/simulator-profiling.md), the
[profile-trace lab guide](docs/profile-trace-lab.md), the
[gpu-timing lab guide](docs/gpu-timing-lab.md) and the
[bottleneck lab guide](docs/matmul-bottleneck-lab.md).

## Tile the matmul and compare the naive, tiled and vendor rates

The Day 76 labs tile the Day 74 matmul with threadgroup memory, the Metal
counterpart of CUDA shared memory. Three commands do this work:

- `ai-journey-matmul-tiled` checks three tiled Metal kernels, with tiles of 8,
  16 and 32. It times them with the naive Day 74 kernel and `torch.matmul` on
  the Apple GPU of a Mac.
- `python -m ai_journey.matmul_tiled_gap_cli` times one call of each side. It
  compares each side with the Day 74 ceilings and gives the Day 75 bottleneck
  name.
- `python -m ai_journey.gpu_tiled_cli` writes a tiled CUDA kernel and one
  cuBLAS SGEMM call as source text. It checks the texts without a compiler,
  and it runs the kernel schedule in a CPU simulator.

Each command seals a report, and each `verify` checks a report again. These
commands run the Metal lab on a Mac and check the CUDA texts on any host:

```bash
ai-journey-matmul-tiled run \
  --config config/day-76-matmul-tiled.json \
  --output /tmp/day-76-matmul-tiled.json
ai-journey-matmul-tiled verify \
  --report /tmp/day-76-matmul-tiled.json \
  --config config/day-76-matmul-tiled.json
python -m ai_journey.gpu_tiled_cli run \
  --output /tmp/day-76-gpu-tiled-cuda.json
python -m ai_journey.gpu_tiled_cli verify \
  --report /tmp/day-76-gpu-tiled-cuda.json
```

The gap commands read the tiled report, a gpu-ceilings report and a one-call
timing report. The tiled lab guide gives these commands in order.

On the development host, one run of the pinned config gave these numbers. Each
value is the median GFLOP/s of 11 samples, and the guide gives the IQR range of
each median. The tiled column gives the best tile, which was tile 32 at each
shape. The vendor column gives `torch.matmul` on the MPS device.

| M x N x K | Naive | Tiled | Vendor | cuBLAS |
| --- | --- | --- | --- | --- |
| 1000 x 1000 x 1000 | 369.1 | 324.2 | 2160.0 | not measured |
| 1024 x 1024 x 1024 | 380.2 | 338.5 | 2294.1 | not measured |
| 2048 x 2048 x 2048 | 387.3 | 339.6 | 3608.8 | not measured |
| 4096 x 4096 x 4096 | 369.4 | 353.5 | 3940.1 | not measured |
| 2000 x 3000 x 1000 | 393.8 | 342.3 | 3448.9 | not measured |

The tiled kernels were not faster than the naive kernel. Tile 32 was slower at
4 shapes, and the rule found no clear difference at 4096 x 4096 x 4096.
`torch.matmul` was 6.7 to 11.1 times as fast as the best tile. The report
recorded load averages of 9.93, 11.62 and 14.06 before the run and 9.70, 11.40
and 13.90 after it.

The gap run gave `latency or occupancy-bound` as the top name of the naive
kernel and of each tile. Each kernel used at most 0.082 of the measured FMA
ceiling of 4836.8 GFLOP/s. The `uptime` command printed 9.81, 10.52 and 13.03
before the ceiling run. All values in the table and in these two paragraphs
are host-specific Apple GPU (Metal) measurements from one shared host, not
portable figures.

On the pinned CUDA plan, all 297 CUDA checks and all 63 cuBLAS checks pass. The
CPU simulator gives the replay bits in all 27 runs. Both CUDA commands print
the report digest
`2419f0bfedcffd5eaa363e07640023e4e81a4bcfc752a415fd8a30158f06ffae`.

The Metal commands exit 0 on success and 1 when a report fails verification.
They exit 2 for usage errors, unreadable or invalid inputs, or a refused
output. The gap command also exits 2 for reports that do not pair. Code 2 also
marks a run that failed a check, had a device error or exceeded its time
budget. Code 3 means that the host cannot run Metal kernels through PyTorch.
The CUDA command needs no GPU, so it uses only the codes 0, 1 and 2.

Evidence boundary: the Metal numbers come from one Apple GPU through Metal on
one host. PyTorch does not name the kernel of `torch.matmul`, so the vendor
number names the call, not a kernel. No CUDA kernel ran, the labs used no
NVIDIA GPU, and they did not measure cuBLAS. Nothing compiled or ran the CUDA
and cuBLAS texts. Each bottleneck name is an inference from timings and
measured ceilings, not from hardware counters.

CI job `matmul-tiled-contracts` runs the tiled and gap tests on Linux with CPU
torch. It also requires each installed Apple GPU command to exit 3 and write no
report. Then it verifies the recorded reports and builds their gap again from
the installed wheel. CI job `nvidia-preflight-contracts` runs the CUDA contract
tests. See the [tiled matmul lab guide](docs/matmul-tiled-lab.md) and the
[tiled CUDA contract guide](docs/gpu-tiled-cuda-contract.md).

## Apply a GPU performance checklist and record the speedup of each step

The Day 77 labs apply a GPU performance checklist to the float32 matmul of the
Day 74 and Day 76 labs. The items are coalescing, occupancy, tiling,
arithmetic intensity, memory vs compute bound and control divergence. Three
commands do this work:

- `ai-journey-matmul-checklist` declares eight steps, S0 to S7, each with one
  change and one hypothesis. It checks each kernel and times it with
  `torch.matmul` on the Apple GPU of a Mac.
- `python -m ai_journey.matmul_checklist_bound_cli` times one call of each
  side. It compares each rate with the measured Day 74 ceilings and a traffic
  model, and it gives the Day 75 bottleneck name.
- `python -m ai_journey.gpu_checklist_cli` writes thread-coarsened and
  register-blocked CUDA matmul texts. It checks the texts without a compiler.

These commands run the Metal lab on a Mac and check the CUDA texts on any
host:

```bash
ai-journey-matmul-checklist run \
  --config config/day-77-matmul-checklist.json \
  --output /tmp/day-77-matmul-checklist.json
ai-journey-matmul-checklist verify \
  --report /tmp/day-77-matmul-checklist.json \
  --config config/day-77-matmul-checklist.json
python -m ai_journey.gpu_checklist_cli sources \
  --output-dir /tmp/day-77-cuda-sources
python -m ai_journey.gpu_checklist_cli check \
  --source-dir /tmp/day-77-cuda-sources
```

The bound commands read the checklist report, a gpu-ceilings report and a
one-call timing report. The printable checklist guide gives these commands in
order. Every hypothesis entered the repository before the first timed run of a
Day 77 kernel.

On the development host, one run of the pinned config gave these speedups.
Each cell gives the ratio of the median GFLOP/s of the step kernel to that of
its base kernel, at one sweep point. Each median comes from 11 samples. The
word after the ratio is the verdict of the Day 76 rule.

| Step | Item | Kernel | Base kernel | 1024 x 1024 x 1024 | 2048 x 2048 x 2048 | 4096 x 4096 x 4096 | 2000 x 3000 x 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | coalescing | `matmul_x_columns` | `matmul_x_rows` | 1.983, faster | 2.647, faster | 1.753, faster | 3.296, faster |
| S1 | tiling | `matmul_tiled_32` | `matmul_x_columns` | 0.914, no clear difference | 0.893, slower | 1.142, faster | 0.889, slower |
| S2 | arithmetic intensity | `matmul_s2_coarse_4x1` | `matmul_tiled_32` | 1.942, faster | 2.036, faster | 1.807, faster | 2.163, faster |
| S3 | arithmetic intensity | `matmul_s3_block_4x4` | `matmul_s2_coarse_4x1` | 1.463, no clear difference | 1.662, faster | 1.896, faster | 1.808, faster |
| S4 | memory vs compute bound | `matmul_s4_vec4` | `matmul_s3_block_4x4` | 1.324, faster | 1.319, faster | 1.350, faster | 1.153, faster |
| S5 | tiling | `matmul_s5_pad4` | `matmul_s4_vec4` | 1.060, no clear difference | 0.981, no clear difference | 0.986, no clear difference | 1.002, no clear difference |
| S6 | occupancy | `matmul_s6_group_8x8` | `matmul_s5_pad4` | 0.886, no clear difference | 1.038, no clear difference | 1.025, no clear difference | 1.024, no clear difference |
| S6 | occupancy | `matmul_s6_group_32x32` | `matmul_s5_pad4` | 0.712, no clear difference | 0.749, slower | 0.779, slower | 0.722, slower |
| S7 | control divergence | `matmul_s7_interior` | `matmul_s5_pad4` | 1.015, no clear difference | 1.164, faster | 1.182, faster | 1.080, no clear difference |

The rule calls a ratio faster or slower only when it is 1.1 or more in that
direction and the IQR ranges do not overlap. The steps form one chain, so each
ratio measures one change on top of the changes before it. The predictions
held at 24 of 36 points. They failed most for S4 and for the 8 x 8 groups, at
all four points each.

S7 was the fastest kernel at each point. It reached 1312.0 to 1825.8 GFLOP/s,
3.70 to 6.16 times the S0 rate. That is 0.518 to 0.584 of the `torch.matmul`
rate at the same point. The report recorded load averages of 19.21, 21.17 and
23.01 before the run and 16.63, 20.39 and 22.64 after it. The `uptime` command
printed 16.11, 20.65 and 22.85 before the lock.

The bound run joined that report with a gpu-ceilings run and a one-call timing
run of the same day. The FMA ceiling was 4835.1 GFLOP/s. The highest FMA share
of a kernel was 0.378, for S7 at 4096 cubed. `torch.matmul` reached 0.465 to
0.728. The rule named 16 of the 44 points `latency or occupancy-bound`, a
residual name, and the other 28 points `indeterminate`. The `uptime` command
printed 17.24, 15.91 and 16.75 before the ceiling run.

All values in the table and in the three paragraphs after it come from
host-specific Apple GPU (Metal) measurements on one shared host. They are not
portable figures.

The CUDA check runs 11 checks on each coarsened text and 12 checks on each
blocked text, at each of 18 Day 74 shapes. Each pinned text passes each check.
The CPU block simulator gives the replay bits in all 45 runs. These results
come from text checks and a CPU model, not from a GPU.

The Metal commands exit 0 on success and 1 when a report fails verification.
They exit 2 for usage errors, unreadable or invalid inputs, or a refused
output. The bound command also exits 2 for reports that do not pair. Code 2
also marks a run that failed a check, had a device error or exceeded its time
budget. Code 3 means that the host cannot run Metal kernels through PyTorch.
The CUDA command needs no GPU, so it uses only the codes 0, 1 and 2.

Evidence boundary: the Metal numbers come from one Apple GPU through Metal on
one host. PyTorch does not name the kernel of `torch.matmul`, so the vendor
number names the call, not a kernel. No CUDA kernel ran, the labs used no
NVIDIA GPU, and they did not measure cuBLAS. Nothing compiled or ran the CUDA
texts. The labs read no hardware counter, so they did not measure occupancy.
Each bottleneck name is an inference from timings and measured ceilings, not
from hardware counters.

CI job `matmul-checklist-contracts` runs the checklist and bound tests on
Linux with CPU torch. It also requires each installed Apple GPU command to
exit 3 and write no report. Then it verifies the recorded reports and builds
their bound again from the installed wheel. CI job
`nvidia-preflight-contracts` runs the CUDA contract tests. See the
[checklist lab guide](docs/matmul-checklist-lab.md), the
[printable GPU checklist](docs/gpu-performance-checklist.md) and the
[checklist CUDA contract guide](docs/gpu-checklist-cuda-contract.md).

## Speed up a sum reduction step by step and compare three scans

Day 78 of the curriculum names GPU MODE Lecture 9, "Reductions", and Lecture
20, "Scan Algorithm". Ajay must still watch both lectures and check the labs
against them. Three commands do this work:

- `ai-journey-gpu-reduce` declares six float32 sum reductions, R0 to R5. Each
  step makes one change to the step before it and states one hypothesis. The
  command checks each kernel against the exact sum and a derived error bound.
  It times each kernel with `torch.sum` on the Apple GPU of a Mac.
- `python -m ai_journey.gpu_reduce_cuda_cli` writes R1 and a warp-shuffle twin
  of R3 as CUDA C++ text. It checks the texts without a compiler, and a CPU
  simulator predicts the bits of each result.
- `ai-journey-gpu-scan` declares three float32 inclusive scans, S0 to S2, each
  with one hypothesis. It checks each scan against a float64 reference and a
  declared bound. It times the scans with `torch.cumsum` and the Day 74 copy
  kernel on the Apple GPU.

These commands run the Metal labs on a Mac and check the CUDA texts on any
host:

```bash
ai-journey-gpu-ceilings run \
  --config config/day-74-gpu-ceilings.json \
  --output /tmp/day-74-gpu-ceilings.json
ai-journey-gpu-reduce run \
  --config config/day-78-gpu-reduce.json \
  --ceilings /tmp/day-74-gpu-ceilings.json \
  --output /tmp/day-78-gpu-reduce.json
ai-journey-gpu-reduce verify \
  --report /tmp/day-78-gpu-reduce.json \
  --config config/day-78-gpu-reduce.json \
  --ceilings /tmp/day-74-gpu-ceilings.json
ai-journey-gpu-scan run \
  --config config/day-78-gpu-scan.json \
  --output /tmp/day-78-gpu-scan.json
ai-journey-gpu-scan verify \
  --report /tmp/day-78-gpu-scan.json \
  --config config/day-78-gpu-scan.json
python -m ai_journey.gpu_reduce_cuda_cli sources \
  --output-dir /tmp/day-78-cuda-sources
python -m ai_journey.gpu_reduce_cuda_cli check \
  --source-dir /tmp/day-78-cuda-sources
python -m ai_journey.gpu_reduce_cuda_cli simulate \
  --config config/day-78-gpu-reduce.json
```

The reduction run divides each rate by the bandwidth ceiling of a Day 74
report, so the first command makes a fresh one. Each hypothesis of both labs
entered the repository before the first timed run of its kernels. The scan
plan uses one fact from an untimed correctness check, and the scan guide
states it.

On the development host, one run of the pinned reduction config gave these
rates. Each value is the median GB/s of 14 samples, with 4 n bytes for each
call. The reduction guide gives the IQR range of each median.

| n | R0 | R1 | R2 | R3 | R4 | R5 | `torch.sum` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 65536 | 13.4 | 23.5 | 28.4 | 33.8 | 38.6 | 24.8 | 36.9 |
| 1048576 | 19.3 | 50.8 | 90.1 | 141.2 | 199.1 | 283.8 | 331.9 |
| 16777216 | 19.7 | 56.2 | 93.6 | 110.1 | 107.3 | 106.6 | 101.5 |
| 67108864 | 19.9 | 55.2 | 100.9 | 108.2 | 107.7 | 108.0 | 108.1 |
| 10000019 | 19.9 | 55.3 | 95.2 | 97.5 | 99.0 | 93.6 | 18.7 |

Each cell of the next table gives the ratio of a step kernel over its base
kernel, from their median GB/s at one size. The word after the ratio is the
verdict of the Day 76 rule. The rule calls a ratio faster or slower only
when it is 1.1 or more in that direction and the IQR ranges do not overlap.

| Step | Kernel | Base | 65536 | 1048576 | 16777216 | 67108864 | 10000019 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | `reduce_r0_interleaved` | `torch.sum` | 0.362, slower | 0.058, slower | 0.194, slower | 0.184, slower | 1.063, no clear difference |
| R1 | `reduce_r1_sequential` | R0 | 1.755, faster | 2.630, faster | 2.848, faster | 2.768, faster | 2.785, faster |
| R2 | `reduce_r2_first_add` | R1 | 1.213, faster | 1.774, faster | 1.667, faster | 1.829, faster | 1.721, faster |
| R3 | `reduce_r3_simd_shuffle` | R2 | 1.188, faster | 1.568, faster | 1.176, faster | 1.073, no clear difference | 1.025, no clear difference |
| R4 | `reduce_r4_simd_sum` | R3 | 1.144, faster | 1.410, faster | 0.974, no clear difference | 0.995, no clear difference | 1.015, no clear difference |
| R5 | `reduce_r5_grid_stride` | R4 | 0.642, slower | 1.425, faster | 0.994, no clear difference | 1.003, no clear difference | 0.946, no clear difference |

The steps form one chain, so each ratio measures one change on top of the
changes before it. The predictions held at 23 of 30 points. At 2^16, R0 was
slower than `torch.sum`, and R2, R3 and R4 were each faster than their bases.
The plan said that the host dispatch sets the time there. The fastest kernel
reached 2.891 to 14.707 times the R0 rate.

The ceiling of the fresh Day 74 run was 87.77 GB/s, from `stream_copy` at 256
MiB. From 2^24, R2 to R5 reached 1.067 to 1.254 of the ceiling. The ceiling is
the best median of the Day 74 kernels, not a hardware limit. At the ragged size
of 10000019, `torch.sum` gave 18.7 GB/s. PyTorch does not report which kernel
`torch.sum` ran, so the lab cannot tell why.

All 78 outputs stayed within their bounds: 13 cases and 6 kernels. In the
three exact cases, every kernel gave the exact sum. R3 kept the bits of R2 in
all 13 cases and at all 5 sizes, as the plan predicted. The reduction report
recorded load averages of 12.58, 11.47 and 12.76 before the run and 12.51,
11.51 and 12.74 after it. The `uptime` command printed 12.40, 11.38 and 12.76
before the ceiling run.

One run of the pinned scan config gave these rates. Each value is the median
GB/s of 11 samples, with 8 bytes for each element. The scan guide gives the IQR
range of each median.

| Length | S0 | S1 | S2 | `torch.cumsum` | `stream_copy` |
| --- | --- | --- | --- | --- | --- |
| 2^16 | 31.93 | 27.54 | 51.49 | 39.32 | 134.47 |
| 2^20 | 63.44 | 72.46 | 100.62 | 23.38 | 159.38 |
| 2^24 | 42.98 | 47.08 | 47.78 | 4.21 | 80.00 |
| 5000003 | 44.01 | 44.58 | 45.35 | 5.08 | 77.42 |

Each cell of the next table gives the ratio over the base kernel of the plan
and the verdict of the same rule.

| Step | Kernel | Base | 2^16 | 2^20 | 2^24 | 5000003 |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | `scan_s0_hillis_steele` | `torch.cumsum` | 0.812, slower | 2.714, no clear difference | 10.210, faster | 8.667, faster |
| S1 | `scan_s1_blelloch` | `scan_s0_hillis_steele` | 0.862, slower | 1.142, faster | 1.095, no clear difference | 1.013, no clear difference |
| S2 | `scan_s2_simd` | `scan_s0_hillis_steele` | 1.612, faster | 1.586, faster | 1.112, faster | 1.030, no clear difference |

S2 was the fastest kernel at each point. The predictions held at 5 of 12
points. At 2^24 and at 5000003, S0 was 10.2 and 8.7 times as fast as
`torch.cumsum`. At 2^20, the IQR range of `torch.cumsum` was 11.58 to 92.55
GB/s, a factor of 8.

Each run of each kernel kept its sentinel tail and stayed within its bound. S0
and S1 held the bits of their CPU replays in each of the 8 normal cases.
`torch.cumsum` gave other bits from run to run in the normal cases of 65536,
1048577 and 5000003 elements. The scan report recorded load averages of 10.46,
10.89 and 12.69 before the run and 11.51, 11.12 and 12.74 after it.

The four tables and the paragraphs between the first table and this one give
host-specific Apple GPU (Metal) measurements. They come from one shared host,
and they are not portable figures. One run gives each ratio, so another run can
give another verdict.

The CUDA check runs 13 checks on the R1 text and 14 checks on the shuffle
text, at each of 14 sizes. Each pinned text passes each check. The simulator
predicts 36 results, and each result is within its bound and has no race. At
all 8 inputs of the recorded Metal report, each simulated result has the bits
of its Metal twin. These results come from text checks and a CPU model, not
from a GPU.

The Metal commands exit 0 on success and 1 when a report fails verification.
They exit 2 for usage errors, unreadable or invalid inputs, or a refused
output. Code 2 also marks a run that failed a check, had a device error or
exceeded its time budget. Code 3 means that the host cannot run Metal kernels
through PyTorch. The CUDA command needs no GPU, so it uses only the codes 0, 1
and 2.

Evidence boundary: the Metal numbers come from one Apple GPU through Metal on
one host. SIMD-group functions are Metal's counterpart of CUDA warp shuffles,
not the same instruction. PyTorch does not name the kernels of `torch.sum` and
`torch.cumsum`, so each vendor number names the call, not a kernel. No CUDA
kernel ran, and the labs used no NVIDIA GPU. Nothing compiled or ran the CUDA
texts. The labs read no hardware counter, so each bound name is an inference
from timings and measured rates.

CI job `gpu-reduce-scan-contracts` runs the reduction, CUDA and scan tests on
Linux with CPU torch. It also requires each installed Apple GPU command to exit
3 and write no report. Then the installed `verify` command of each lab checks
its recorded report against its recorded config. The reduction report must also
match the recorded Day 74 ceiling. CI job `nvidia-preflight-contracts` runs the
same tests. See the [reduction lab guide](docs/gpu-reduction-lab.md), the
[reduction CUDA contract guide](docs/gpu-reduce-cuda-contract.md) and the
[scan lab guide](docs/gpu-scan-lab.md).

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
