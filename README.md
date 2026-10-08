# AI Journey: Days 0–34

This repository is a public-safe, executable record of an added setup day plus the
selected work through Day 34 of a 90-day AI engineering learning plan. It combines
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
- Unit tests in [`tests/`](tests/)
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
python -m unittest discover -s tests -v
```

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

## Daily map

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

## License

The code in this learning repository is available under the MIT License.
