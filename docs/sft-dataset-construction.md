# Reproducible 500-example SFT dataset

This workflow builds a reviewable supervised fine-tuning dataset from the public
Databricks Dolly 15k snapshot. The Day 67 policy pins Hugging Face revision
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, the 13,085,339 source bytes, and
SHA-256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`.
Dolly is attributed to Databricks contributors and distributed under Creative Commons Attribution-ShareAlike 3.0 Unported. Keep the attribution, license, and share-alike obligations with any redistributed derivative dataset.

## Dataset contract

The loader accepts only the pinned newline-terminated UTF-8 JSONL bytes. Every
record must contain exactly `instruction`, `context`, `response`, and `category`.
Text is normalized to NFC with canonical newlines before content identities are
derived. The quality stage applies explicit byte limits and rejects empty,
oversized, or trivial identity answers with counted reasons. Exact duplicates are
removed independently of source order.

The selector apportions 500 examples across available Dolly task categories by a
deterministic largest-remainder policy, then ranks candidates by a seeded content
hash. A second seed produces 450 train examples and 50 validation examples while
retaining category coverage and preventing content identity overlap.

`dataset.jsonl` stores one conventional `messages` array per example with one
`user` turn and one `assistant` turn. Context is appended to the user instruction
under an explicit `Context:` delimiter. `chatml.jsonl` stores an unambiguous
JSON-escaped ChatML rendering plus the exact character span that may contribute to
assistant-only loss. User text and role markers are never included in that span.

## Build, verify, and replay

Download the immutable public snapshot directly from its pinned revision:

```bash
curl --fail --location --retry 2 --max-time 120 \
  https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl \
  --output /tmp/databricks-dolly-15k.jsonl
```

Run the installed operator from the repository root:

```bash
ai-journey-sft-dataset build \
  --source /tmp/databricks-dolly-15k.jsonl \
  --config config/day-67-sft-dataset.json \
  --output /tmp/sft-dataset-evidence

ai-journey-sft-dataset verify \
  --evidence /tmp/sft-dataset-evidence

ai-journey-sft-dataset replay \
  --source /tmp/databricks-dolly-15k.jsonl \
  --evidence /tmp/sft-dataset-evidence
```

`build` refuses to overwrite an output and atomically publishes
`dataset.jsonl`, `chatml.jsonl`, and `manifest.json`. The manifest binds the
public source identity and license, policy controls, rejection and duplicate
counts, category quotas, split and byte metrics, artifact digests, and its own
canonical seal.

`verify` does not trust the construction path. It checks the exact inventory,
bounded regular files, duplicate JSON fields, source and policy schemas, manifest
seal, pipeline arithmetic, artifact digests, message roles, stable identities,
split order and counts, template text, and assistant spans. `replay` additionally
reloads the pinned source, reruns every transformation, and requires byte-for-byte
artifacts and an identical final manifest.

## Evidence boundary

The repository tests and CI prove deterministic CPU construction and replay from
the named public snapshot. They do not prove that Ajay watched the Day 67 lecture,
personally reviewed the 500 selected examples, chose these policies for a real
deployment, trained a model, improved model quality, or ran a production system.
Learner-complete evidence still requires Ajay to run the installed workflow,
record its manifest digest, inspect selected and rejected examples across every
category, document any content risks, and explain the template, loss-mask,
selection, split, and license tradeoffs in his own words.
