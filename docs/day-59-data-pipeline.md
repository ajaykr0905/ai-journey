# Day 59 public data pipeline

This workflow builds a bounded decoder-training dataset from the immutable
English Parallel Universal Dependencies test snapshot. The snapshot is
CC-BY-SA-3.0 and is pinned by repository, revision, byte count, and SHA-256 in
`src/ai_journey/tokenizer_sources.py`. The source annotations are not treated as
training prose; only validated `sent_id` and `text` metadata enter the pipeline.
Every manifest retains attribution to Parallel Universal Dependencies, CoNLL
2017, DFKI, Google, the UD community, and the source contributor credits.

## Build and verify

Fetch the exact public snapshot:

```bash
curl --fail --location --retry 2 --max-time 60 \
  https://raw.githubusercontent.com/UniversalDependencies/UD_English-PUD/6e328e8195c04a297c1e375d21fe663be1ce6178/en_pud-ud-test.conllu \
  --output /tmp/en_pud-ud-test.conllu
```

Run the installed commands from an empty output path:

```bash
ai-journey-data-pipeline build \
  --snapshot /tmp/en_pud-ud-test.conllu \
  --config config/day-59-data-pipeline.json \
  --output /tmp/day-59-dataset

ai-journey-data-pipeline verify \
  --dataset /tmp/day-59-dataset

ai-journey-data-pipeline accept \
  --dataset /tmp/day-59-dataset \
  --contract config/day-59-data-pipeline-acceptance.json \
  --config config/day-59-data-pipeline.json \
  --snapshot /tmp/en_pud-ud-test.conllu
```

`build` validates the source bytes before parsing. It normalizes Unicode and
whitespace, records changed source hashes, applies explicit quality reasons,
deduplicates exact cleaned text, creates a seeded train/validation split,
assigns balanced shards, trains BPE on training documents only, and publishes
the complete directory atomically. Existing output directories are never
overwritten.

`verify` independently checks the manifest seal, exact file inventory, safe
shard paths, byte counts, content digests, document coverage, source binding,
tokenizer state, token counts, and compression observations. `accept` also
requires the exact pinned source, configuration, and expected public result.

## Pinned CPU observations

The acceptance contract records these deterministic observations:

- 1,000 source documents;
- 3 quality rejections and no exact duplicates;
- 897 training documents and 100 validation documents;
- 69,626 content tokens from 110,379 UTF-8 bytes;
- 1.5853129578031195 bytes per token;
- 0.3692097228639506 token reduction relative to one token per source byte.

Token counts exclude synthetic boundary markers because this artifact is the
cleaned content dataset, not a model-specific packed sequence. The manifest
records individual train and validation metrics so downstream packing can add
its own explicit boundary policy.

## Failure and evidence boundaries

Publication writes into a task-owned temporary sibling directory, syncs every
file, writes the manifest last, and renames the complete directory into place.
If the process stops before the rename, rerun `build` with the same empty output
path. If the output exists, run `verify`; never delete or overwrite it without
first deciding whether it is evidence that must be retained.

The observed values are deterministic CPU preprocessing results. They do not
show that the supporting quantization video was watched, that a manual learner
exercise was completed, or that any model was trained. They provide no GPU,
distributed-scale, throughput, production-readiness, or model-quality claim.
