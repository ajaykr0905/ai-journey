# Day 23: Model selection without leakage

## Implemented

- Split complete records into deterministic train, development, and test sets.
- Build all encoded sets from one vocabulary while fingerprinting each partition.
- Train shuffled minibatches with optional clipping, weight decay, and early stopping.
- Sweep logarithmically spaced learning rates from identical seeded initializations.
- Select on development loss, then evaluate the selected model once on test data.
- Persist a JSON report, verified checkpoint, and dependency-free SVG sweep plot.

The test set does not participate in hyperparameter selection. A widening positive
development gap is reported separately from a late train/development divergence
signal so the evidence does not reduce overfitting to one threshold.

## Pending learner evidence

- Lecture viewing and type-along confirmation.
- A learner-run sweep and explanation of the selected learning rate.
- Interpretation of the train/development/test losses and sweep plot.
