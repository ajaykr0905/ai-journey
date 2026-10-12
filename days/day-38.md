# Day 38: bounded full-transformer training

## Curriculum contract

The plan assigns the final segment of Karpathy's GPT lecture, with attention to
dropout, scaling, and the connection to larger GPT training. The manual task is
to train the full model, generate Tiny Shakespeare text, and capture the output.
The workbook is planning input only. It does not establish that Ajay watched the
lectures, ran the command, inspected the sample, or created the screenshot.

## Repository capability

`ai-journey-day-38` verifies the exact public Tiny Shakespeare snapshot before
tokenization. It creates disjoint ordered train and held-out partitions, applies
explicit token budgets, trains the existing complete decoder-only transformer,
publishes an exact restart checkpoint, reloads that checkpoint, and generates a
seeded sample. Its report rejects nonfinite metrics, inconsistent counts,
contradictory findings, missing limitations, malformed runtime evidence, and
fingerprint mismatches.

Run the default bounded experiment:

```bash
mkdir -p artifacts
ai-journey-day-38 \
  --corpus artifacts/tinyshakespeare.txt \
  --download \
  --output artifacts/day-38-transformer.json \
  --checkpoint artifacts/day-38-transformer.pt
```

Resume only with the exact same source and configuration:

```bash
ai-journey-day-38 \
  --corpus artifacts/tinyshakespeare.txt \
  --output artifacts/day-38-transformer-resumed.json \
  --checkpoint artifacts/day-38-transformer.pt \
  --resume
```

The output and checkpoint paths must differ from the corpus and from each other.
Changing the model, optimizer, data limits, or seed invalidates checkpoint
compatibility rather than silently starting a different experiment.
`--max-parameters` limits model allocation (default: 1,000,000 parameters).
The corpus fingerprint now binds partition lengths; checkpoints from the older
partition fingerprint must be regenerated. New checkpoints additionally bind
the sampler's token stream, seed, context, and batch size.

## Verified bounded run

Audit or serve the trained checkpoint without running training again:

```bash
ai-journey-day-38-infer \
  --report artifacts/day-38-transformer.json \
  --checkpoint artifacts/day-38-transformer.pt \
  --corpus artifacts/tinyshakespeare.txt \
  --operation audit --output artifacts/day-38-audit.json
```

Use `--operation generate --new-tokens 80` for seeded samples, optionally
`--cached`; use `--operation benchmark --new-tokens 4` for full/cache timings.
The benchmark includes cache validation and context rollover but excludes
artifact loading. It measures bounded local CPU calls.

Run the predeclared dropout comparison with three configurations and three seeds:

```bash
ai-journey-day-38-ablate \
  --corpus artifacts/tinyshakespeare.txt \
  --protocol config/day-38-dropout-ablation.json \
  --output artifacts/day-38-dropout.json
```

The 2026-10-12 CPU run used 4,096 training tokens, 1,024 held-out tokens,
80 updates per trial, and seeds 38/39/40. Mean held-out NLL was 2.894318 for
dropout 0.0, 2.900121 for 0.1, and 2.897345 for 0.2. Both contrasts were
inconclusive against the predeclared 0.01 improvement threshold. The report
contains each trial's configuration, fingerprints, trace, metrics, and runtime;
these machine results do not establish learner completion.

Two clean CPU runs on 2026-10-12 used the pinned source digest
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`,
65,536 training tokens, 8,192 held-out tokens, an 80-step configuration, dropout
0.2, a 65-character vocabulary, four heads, two blocks, 32 embedding channels,
and 30,656 parameters. Both
runs produced model fingerprint
`e9171adbec47da0b0b0665ebc0feb4a1b91154626dd6e1423f3d27d8107d282b`,
training NLL 2.575125, held-out NLL 2.710894, and the same sampled token sequence.
Each final measured run completed in about 10 seconds on the local CPU. The
untrained reference NLLs were 4.184218 and 4.181267 respectively.

The short sample is visibly imperfect character text. Loss improvement and exact
repeatability support the bounded training/restart claim; they do not establish
useful language quality, convergence, statistical significance, GPU speed,
distributed behavior, production readiness, or similarity to a pretrained GPT.

## Learner evidence still required

Ajay must separately confirm or explicitly skip both assigned lectures, run the
command with a recorded seed and output paths, inspect the generated text, add
the requested screenshot, explain where dropout acts and why it is disabled at
evaluation, and write a brief personal reflection on the lecture's scaling and
real-GPT closing remarks.
