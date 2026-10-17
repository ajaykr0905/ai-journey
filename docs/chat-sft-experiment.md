# Assistant-only chat SFT comparison

This experiment turns a small authored chat fixture into bounded, reproducible
evidence for assistant-only supervised fine-tuning (SFT). It compares a frozen
baseline, full fine-tuning, rank-one LoRA, and rank-two LoRA after the same base
language-model warm-up. Every arm starts from the same warmed function for its
seed and receives the same scheduled windows.

It is deliberately a tiny CPU character-model diagnostic. It does not measure
factuality or hallucination, execute tools, establish production safety, or
generalize to large models. The fixture and split are authored in this repository;
no private data or external credentials are used.

## Run and replay

From an installed checkout:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ai-journey-chat-sft-compare \
  --config config/chat-sft-comparison.json \
  --output artifacts/chat-sft-comparison.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ai-journey-chat-sft-verify \
  --report artifacts/chat-sft-comparison.json
```

The comparison writes with durable temporary-file replacement. The verifier
rebuilds both warmed bases and all eight adaptation trials and requires exact
fingerprints, training traces, assistant metrics, and summaries. Python and
PyTorch version strings are recorded but excluded from numeric equality; device,
dtype, and thread controls remain validated. Exact replay may fail on a different
numerical backend, which is preferable to silently applying a tolerance.

## Data and supervision contract

Each conversation has a stable identifier and complete, alternating turns. A
conversation may start with one system message. A tool request must be emitted by
the assistant, followed by the matching tool result and a final assistant answer.
Call identifiers are unique and tool arguments use canonical finite JSON.

Rendering uses one canonical JSON object per line, so role-like text inside message
content cannot inject turns. The codec is fit on training conversations only.
Training windows never cross a conversation boundary, and their loss mask selects
only assistant characters. Validation enumerates every held-out assistant character
exactly once with bounded conversation-local context; system, user, and tool-result
characters provide context but never targets.

The checked-in split contains eight training and four validation conversations.
The validation evidence covers 593 assistant characters and is bound to digest
`d038e7491233444ffd11fd5ff42a39a0bebf8400cbf311db2f4bbde7222934c8`.

## Checked-in protocol result

On the declared two-seed CPU float64 protocol, the canonical report digest is
`7f7823e8da89adf8325dab2b7a9a1ca69e24b0b67773ab2b641558d3ca68cb3c`.
Lower assistant NLL is better.

| Arm | Trainable fraction of full | Mean assistant NLL | Sample standard deviation |
|---|---:|---:|---:|
| Frozen | 0.0000 | 3.116846 | 0.000002 |
| Full | 1.0000 | 2.569781 | 0.008312 |
| LoRA rank 1 | 0.0228 | 3.107952 | 0.001936 |
| LoRA rank 2 | 0.0456 | 3.108716 | 0.004367 |

Full fine-tuning improved over frozen in both seeds by a mean NLL delta of
`-0.547065`. Rank-one and rank-two LoRA also improved over frozen in both seeds,
but by only `-0.008894` and `-0.008130` respectively. Rank two did not consistently
beat rank one: it won one seed and lost one, with mean delta `+0.000764` relative
to rank one. That negative result is retained rather than reframed as a capacity
win.

These are descriptive results for two paired seeds on one fixed synthetic split.
The standard deviations and standard errors in the report are not confidence
intervals. Equal token exposure does not imply equal compute, memory, or latency.

## Failure boundaries

Configuration parsing rejects missing or extra controls, duplicate JSON fields,
nonfinite literals, unsafe dimensions, and experiments beyond the declared CPU
work budget. Report validation checks schema, fixture/schedule digests, target
coverage, parameter accounting, adapter merge equivalence, finite metrics, summary
recomputation, and the report digest. A self-consistent forged report can pass
internal validation; only exact replay checks it against re-executed computation.

This engineering evidence does not show that Ajay watched the scheduled lecture,
completed the manual note-taking exercise, ran the experiment personally, or wrote
a personal interpretation. Those remain learner-owned evidence.
