# Preference optimization experiment

This bounded CPU experiment extends the assistant-only SFT reference with offline
pairwise preference optimization. It compares a frozen control, full fine-tuning,
rank-one LoRA, and rank-two LoRA across two seeds and three prompt-family folds.
Related prompts stay in the same fold, so a reading and its threshold variant
cannot leak across train and held-out partitions.

The fixture is authored and public-safe. Each prompt supplies one reading, and
the preferred answer is grounded in that reading. Canonical JSON Lines make role
boundaries unambiguous. Each response character is scored exactly once with
bounded conversation-local context. The objective is DPO-style but uses mean
response log probability to make response length explicit:

`-logsigmoid(beta * ((policy chosen - policy rejected) - (reference chosen - reference rejected)))`

The reference is the same deterministic assistant-only SFT model for every fold
and arm within a seed. Adapter arms change only the two declared attention
projections. Every arm receives the same sampled preference schedule.

## Run and verify

From an installed checkout:

```bash
ai-journey-preference-compare \
  --config config/preference-comparison.json \
  --output /tmp/preference-comparison.json

ai-journey-preference-verify \
  --report /tmp/preference-comparison.json
```

The compare command validates the complete protocol, writes through an fsynced
temporary file, and atomically replaces the destination. The verify command
rebuilds all 24 seed-fold-arm trials and requires exact schedules, fingerprints,
training traces, held-out metrics, summaries, and negative findings. Runtime
version strings are the only excluded replay fields.

## Measured checked-in protocol

The 18 October 2026 local run used Python 3.13.9, PyTorch 2.14.1, CPU float64,
and one Torch thread. It produced six seed-fold observations per arm and report
SHA-256 `6922ed168cc6a701b3ce743a1340dec57f81326f339af65b2b1e7b8639035000`.

| Arm | Trainable parameters | Mean held-out preference accuracy | Sample standard deviation | Mean reward margin | Mean loss |
|---|---:|---:|---:|---:|---:|
| Frozen | 0 | 0.6250 | 0.2092 | 0.000000 | 0.693147 |
| Full | 4,464 | 0.4583 | 0.1021 | -0.000134 | 0.693234 |
| LoRA rank 1 | 96 | 0.5833 | 0.2582 | -0.000029 | 0.693162 |
| LoRA rank 2 | 192 | 0.5833 | 0.2582 | -0.000023 | 0.693159 |

The negative result is part of the evidence: none of the adapted arms exceeded
the frozen control's mean accuracy, and all had non-positive mean reward margins.
Rank two tied rank one on mean accuracy. Full fine-tuning also trailed both LoRA
arms on this tiny authored fixture. These observations do not establish a general
ranking between adaptation methods.

## Limits

This is offline preference optimization, not a reward-model pipeline, PPO,
on-policy reinforcement learning, human preference collection, or reasoning-trace
evaluation. The sample is an authored fixture with six prompt families. Fold and
seed statistics describe these runs only; they are not confidence intervals.
Parameter counts do not measure runtime, memory, or serving cost. The experiment
does not provide GPU, distributed, production, lecture-viewing, or learner evidence.
