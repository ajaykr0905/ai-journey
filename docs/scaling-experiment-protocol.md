# Bounded scaling experiment protocol

The Day 62 protocol turns a model-size and token-budget grid into a complete,
identity-bound experiment plan. Every coordinate runs every declared seed. The
report aggregates replicates, reserves the largest model at each token budget as
a holdout, fits validation loss against the declared `6 × parameters × tokens`
compute convention, and evaluates the fit without refitting on holdout rows.

## Verify the deterministic reference fixture

```bash
ai-journey-scaling-experiment build \
  --protocol config/day-62-scaling-experiment-fixture.json \
  --output artifacts/day-62-scaling-experiment.json
ai-journey-scaling-experiment verify \
  --report artifacts/day-62-scaling-experiment.json
```

The checked fixture contains 12 deterministic reference rows: three model
sizes, two token budgets, and two seeds. Its losses follow an exact authored
power law so the complete analysis and verification path can run in CPU-only
CI. They are not training measurements. The protocol labels them
`deterministic_reference_fixture`, and the report keeps
`observations_measured=false` and `execution_authenticated=false`.

## Required real-run inputs

A measured protocol must retain the immutable public source URL and digest,
tokenizer digest, full code revision, exact model sizes, token budgets, seeds,
completed token counts, optimizer steps, validation losses, and elapsed times.
Run identities bind every coordinate to those source and code identities. A
report rejects missing, duplicate, relabelled, partial, or off-grid runs.

At least two replicates are required per coordinate. The analysis reports the
mean and sample standard deviation separately for each model/token pair. It
never pools different token budgets or silently drops seeds.

## Fit and extrapolation controls

The largest model at each token budget is held out before fitting. The training
fit reports its coefficient, exponent, log-space R², absolute error, percentage
error, and log residual error. Holdout evaluation reports percentage error and
the fraction covered by a descriptive residual band.

Targets at or below the largest fitted compute are interpolation. Targets up to
four times that compute are bounded extrapolation under the checked policy.
Larger targets are speculative and fail admission. The residual band is not a
confidence interval. Acceptance also requires the predeclared fit-quality,
holdout-error, and holdout-coverage thresholds.

## Integrity and limitations

Reports are canonically sealed, independently recomputed, size-bounded, and
atomically published. A valid seal establishes integrity and internal
consistency only. It does not authenticate that training occurred, identify
hardware, prove GPU or distributed execution, establish model quality, or
justify predictions outside the observed range.

Ajay must separately provide lecture-viewing evidence, personal protocol
interpretation, and any real experiment logs. A future measured sweep needs an
appropriate public corpus, executable training configuration, hardware and
software manifest, cost approval, losses, utilization, memory, failures, and
limitations. The reference fixture is not a substitute for that evidence.
