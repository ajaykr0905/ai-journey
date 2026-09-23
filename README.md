# AI Journey: Days 0–19

This repository is a public-safe, executable record of an added setup day plus the
first nineteen workbook days of a 90-day AI engineering learning plan. It combines
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
python -m unittest discover -s tests -v
```

PyTorch is optional. When it is installed, Day 2 reports CUDA availability and
Day 9 compares the manual gradients with `torch.autograd`. Without PyTorch, the
finite-difference verification still runs and CI remains meaningful.

## Configuration safety

Days 0–19 require no credentials. Never commit `.env`, credentials, browser data,
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

## License

The code in this learning repository is available under the MIT License.
