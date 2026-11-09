# Replayable MinHash deduplication

This workflow finds near-duplicate documents in a verified Project 1 dataset and
reports the exact fraction removed. It reads the published train and validation
JSONL shards produced by `ai-journey-data-pipeline`; their manifest seal, shard
digests, byte budgets, row schemas, and document identities must pass independent
verification before candidate discovery begins.

## Policy and algorithm

The checked-in Day 66 policy uses normalized five-word shingles, 128 seeded
MinHash values, 16 bands of eight rows, and an exact Jaccard acceptance threshold
of 0.8. The LSH curve gives a collision probability above 94% at the acceptance
threshold and below 7% at 0.5 similarity. LSH only proposes candidates. Every
candidate is measured against the original shingle sets, and only exact Jaccard
similarity at or above 0.8 creates a duplicate edge.

Accepted edges form transitive clusters. The lexicographically smallest document
ID is retained in each cluster, making representative selection independent of
input order. The report preserves every candidate, its MinHash estimate, exact
similarity, acceptance decision, cluster membership, and the counts used to
calculate `removed_fraction`.

## Build, verify, and replay

Run the installed command against an existing verified Project 1 dataset:

```bash
ai-journey-minhash build \
  --dataset /path/to/data-pipeline-evidence \
  --config config/day-66-minhash-dedup.json \
  --output /tmp/minhash-evidence

ai-journey-minhash verify \
  --evidence /tmp/minhash-evidence

ai-journey-minhash replay \
  --dataset /path/to/data-pipeline-evidence \
  --evidence /tmp/minhash-evidence
```

`build` refuses to overwrite an output directory. It publishes
`deduplicated-corpus.jsonl` and `manifest.json` together by a durable atomic
rename. `verify` checks the exact inventory, duplicate JSON fields, report seal,
policy, candidate decisions, cluster coverage, removal arithmetic, corpus rows,
and artifact digests without trusting the build path. `replay` additionally
reloads the source shards, checks their identities against the manifest, and
recomputes shingles, signatures, candidates, exact decisions, clusters, output
bytes, metrics, and the final report seal.

The JSON summary prints `input_documents`, `kept_documents`,
`removed_documents`, `removed_fraction`, and `report_sha256`. Keep that output,
the source dataset manifest, and the evidence directory together when reporting
an actual run.

## Evidence boundary

Repository and CI fixtures prove deterministic CPU behavior on public-safe test
data. They do not prove that Ajay ran the command over his selected Project 1
corpus, inspected the clusters, chose the threshold for a particular domain, or
completed the lectures. A learner-complete record still needs the exact dataset
digest, command output, removed fraction, review of representative and rejected
pairs, a threshold tradeoff explanation, and personal notes. This workflow makes
no GPU, distributed-scale, data-quality, production, or model-quality claim.
