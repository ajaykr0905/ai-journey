# Bounded HellaSwag evaluation

Day 49 adds a reproducible multiple-choice evaluation path for the small CPU
training lab. It validates a checksum-pinned 100-row slice from the MIT-licensed
[`Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag) mirror at
revision `218ec52e09a7e7462a5400043bb9a69a41d06b76`. The protocol records both the
full validation Parquet checksum and the exact dataset-server response used by
the bounded run.

The completion-only scorer concatenates each context and candidate ending, then
averages token negative log-likelihood over candidate bytes only. It never includes prompt
tokens in a candidate score. The report includes accuracy, a deterministic 95%
bootstrap interval, Brier score, expected calibration error, in-domain and
zero-shot strata, row-level scores, and source/model/protocol digests. Four
logical shards cover every selected row once without padding duplicates.

## Reproduce and verify

Fetch the exact bounded response, train the scale-limited CPU checkpoint, and
evaluate it:

```bash
curl --fail --location --retry 2 --max-time 60 \
  'https://datasets-server.huggingface.co/first-rows?dataset=Rowan%2Fhellaswag&config=default&split=validation' \
  --output /tmp/hellaswag-first-rows.json

ai-journey-ddp train \
  --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json \
  --checkpoint /tmp/hellaswag-model.pt \
  --output /tmp/hellaswag-training.json \
  --workdir /tmp/hellaswag-training \
  --steps 4

ai-journey-hellaswag evaluate \
  --dataset-response /tmp/hellaswag-first-rows.json \
  --protocol config/hellaswag-validation.json \
  --source data/day-19-demo-names.txt \
  --spec config/ddp-lab.json \
  --checkpoint /tmp/hellaswag-model.pt \
  --output /tmp/hellaswag-evidence.json

ai-journey-hellaswag verify --report /tmp/hellaswag-evidence.json
```

Omit `--checkpoint` to score the deterministic initial model. Compare two sealed
reports only when they cover the same protocol rows:

```bash
ai-journey-hellaswag compare \
  --baseline /tmp/hellaswag-initial.json \
  --candidate /tmp/hellaswag-evidence.json
```

The verified local Day 49 run scored 25/100 (`0.25`) with a deterministic 95%
bootstrap interval of `[0.17, 0.34]`. Its evidence digest is
`dad16b9b5a1534eba657c062a96ec3b240050a13fb7ac3844077f8e3140b7a75`.
This is a wiring and reliability result, not an official 10,042-example
HellaSwag score, a pretrained GPT-2 124M result, evidence of GPU or production
performance, or proof of lecture viewing and learner understanding.
