# Replayable HTML-to-text extraction

This workflow converts a caller-supplied snapshot of public HTML pages into a
content-addressed text corpus. It is intentionally offline: acquisition remains
separate so robots policies, terms, rate limits, and source licenses can be
handled for each source before any bytes enter this repository.

## Input contract

The snapshot is newline-terminated JSONL with one canonical, unique public URL
per row. Each row contains exactly `url`, `content_type`, `payload_base64`, and
`payload_sha256`. The digest is computed over the decoded payload bytes. URLs
must use HTTP or HTTPS without credentials, private addresses, fragments, or
non-default ports. The loader rejects symlinks, duplicate fields, duplicate
URLs, noncanonical base64, changed bytes, more than 1,000 pages, or more than
64 MiB of snapshot data.

The checked-in policy requires at least 100 pages, 200 extracted characters per
accepted page, and at most a 0.5% decoding-replacement fraction:

```bash
ai-journey-html-extract build \
  --snapshot /path/to/public-pages.jsonl \
  --config config/html_extraction_100_pages.json \
  --output /tmp/html-extraction-evidence

ai-journey-html-extract verify \
  --evidence /tmp/html-extraction-evidence
```

`build` checks the snapshot and policy, chooses an explicit charset, removes
scripts, styles, forms, navigation, headers, footers, templates, and other page
chrome, then records an outcome for every input page. Short text, non-HTML
payloads, undecodable content, and empty visible output remain visible as
rejections instead of disappearing from the denominator.

The output directory is published by durable sibling-directory rename and is
never overwritten. `manifest.json` seals the exact snapshot digest, thresholds,
page observations, accepted/rejected counts, extracted character total, corpus
descriptor, and evidence boundary. `corpus.jsonl` retains stable document IDs,
canonical source URLs, text, and text digests. `verify` independently recomputes
the report seal, exact inventory, page/count consistency, corpus digest, row
identities, source coverage, and text hashes.

## Evidence boundary

Repository tests exercise a deterministic 100-page batch, but they are not a
claim that Ajay selected or inspected 100 real pages. A learner-complete record
still needs the public source list and license/terms review, the snapshot digest,
the command output, inspection of poor extractions, and a personal account of
what the default extractor got wrong. This workflow makes no Common Crawl-scale,
GPU, model-training, production, lecture-viewing, or learner-completion claim.
