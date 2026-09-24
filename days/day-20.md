# Day 20 — Neural Character Bigram and Negative Log-Likelihood

**Date:** 2026-09-24

## Goal

Finish the neural version of the character bigram model with independently
written NumPy code. The model turns integer character ids into one-hot vectors,
multiplies those vectors by a weight matrix, applies softmax, and minimizes mean
negative log-likelihood. A deterministic equivalence test confirms that a
particular neural weight matrix produces exactly the same loss as the
additively smoothed Day 19 count model.

The canonical exercise is a type-along. The repository implementation is a
tested reference and does not establish that Ajay completed the type-along.

## Training examples

Day 19 already converted every word into boundary-aware transitions. Day 20
uses the same transitions as supervised examples:

```text
input character id → target character id
```

The 20-word public-safe corpus contains 120 examples and an 18-token
vocabulary. Let `n = 120` and `v = 18`:

```text
input ids:    (n,)    = (120,)
target ids:   (n,)    = (120,)
one-hot X:    (n, v)  = (120, 18)
weights W:    (v, v)  = (18, 18)
logits X @ W: (n, v)  = (120, 18)
```

Each one-hot input row contains one `1`. Multiplying by `W` therefore selects
the weight row for the current character. This is a matrix-multiplication view
of a lookup table, not a hidden layer.

## Stable softmax

Raw logits are unconstrained real values. Softmax turns each row into a
probability distribution:

```text
p[i, j] = exp(logit[i, j]) / sum_k exp(logit[i, k])
```

The implementation subtracts the largest logit in each row before
exponentiating. Adding or subtracting the same constant from all logits in one
row does not change that row's softmax probabilities, while max subtraction
avoids overflow for large positive logits.

Validators require finite inputs, strictly positive probabilities, and rows
that sum to one. The strict positivity matters because the next step takes a
logarithm.

## Negative log-likelihood

For example `i`, the model assigns probability `p[i, target[i]]` to the correct
next character. Mean negative log-likelihood is:

```text
loss = -(1 / n) * sum_i log(p[i, target[i]])
```

Minimizing this loss increases the probability of observed transitions. A
perfect probability of `1` contributes zero loss. Smaller correct-target
probabilities contribute larger positive penalties. Perplexity is `exp(loss)`;
it is included as a readable transformation of the same objective, not as an
independent quality metric.

## Exact bridge from counts to neural weights

Additive smoothing replaces each bigram count `N[i, j]` with
`N[i, j] + alpha`, where `alpha > 0`. The smoothed count probability is:

```text
P_count[i, j] = (N[i, j] + alpha) / sum_k (N[i, k] + alpha)
```

Now choose neural weights:

```text
W[i, j] = log(N[i, j] + alpha)
```

Exponentiating these logits recovers `N[i, j] + alpha`. The softmax denominator
is the same smoothed row total, so `softmax(W) = P_count`. Because the two
models assign the same probability to every transition, their corpus NLLs must
also match.

With `alpha = 1` on the checked-in corpus, the deterministic run reports:

```text
smoothed count-model NLL: 2.051851633208
log-count neural NLL:     2.051851633208
absolute difference:     below 1e-12
```

This is an exact mathematical identity for the fixed corpus. Floating-point
implementations can differ in their last few bits, so the validator accepts an
absolute loss difference below `1e-12`. It is not a claim that every neural
language model is equivalent to a count model.

## Analytic gradient and numerical oracle

For one-hot target matrix `Y`, the mean-loss gradient with respect to logits is:

```text
dL/dlogits = (P - Y) / n
dL/dW      = X.T @ (P - Y) / n
```

The repository checks six selected weights with centered finite differences:

```text
dL/dW[r, c] ≈ (L(W + epsilon) - L(W - epsilon)) / (2 * epsilon)
```

The default experiment's maximum absolute gradient error is `4.973e-10`, below
the validator threshold of `1e-8`.

## Deterministic training result

Full-batch gradient descent starts with an all-zero `18 x 18` weight matrix.
There is no random initialization, shuffled minibatch, or hidden state. With 200
steps and learning rate `10.0`, the checked-in smoke configuration reports:

```text
initial NLL:        2.890371758
trained NLL:        1.408091052
loss reduction:    51.28%
initial perplexity: 18.000000
trained perplexity: 4.088144
terminated samples: 10/10 with seed 2020
```

The loss decrease proves that the implementation can optimize this training
set. It does not prove generalization. The corpus is intentionally tiny, there
is no train/dev/test split, and sample appearance is not a model-quality
benchmark.

## Reproducible experiment

```bash
python scripts/run_day_20.py \
  --output artifacts/day-20-neural-bigram.md \
  --json-output artifacts/day-20-neural-bigram.json
```

The Markdown report records tensor shapes, the exact count/neural comparison,
training checkpoints, gradient-check error, and seeded samples. The JSON report
adds integer examples, probability and weight matrices, per-probe values, and
sample traces. The command accepts the Day 19 corpus format plus explicit
smoothing, step count, learning rate, sample count, and seed arguments.

## What the evidence does and does not establish

The repository verifies boundary-aware example construction, one-hot encoding,
explicit matrix multiplication, numerically stable softmax, probability
invariants, negative log-likelihood, perplexity, additive smoothing, exact
count-to-neural probability and loss agreement, the analytic gradient, a
finite-difference oracle, deterministic optimization, sampling compatibility,
reports, and the command-line workflow. It does not establish lecture viewing,
Ajay's independent type-along, a manual derivation, model generalization, or a
completed Week 3 gate.

## Learner evidence still needed

1. Confirm or explicitly skip the assigned second makemore segment and calculus
   support video.
2. Finish the neural bigram type-along independently rather than copying this
   reference.
3. Run the Day 20 command and record the exact command, output paths, smoothing,
   learning rate, step count, seed, and resulting metrics.
4. Hand-calculate the NLL for a two-example, two-class probability table.
5. Explain why multiplying a one-hot row by `W` selects one row of `W`.
6. Derive why `log(count + alpha)` followed by softmax recovers the additively
   smoothed count probabilities.
7. Explain the derivative formula lesson in Ajay's own words and connect it to
   one entry of the neural bigram gradient.

No lecture viewing, type-along, personal explanation, or milestone pass is
claimed here.

## Sources

- Curriculum lecture and notebook: Andrej Karpathy, *The spelled-out intro to
  language modeling: building makemore*:
  https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb
- Course index and lecture description: Andrej Karpathy, *Neural Networks: Zero
  to Hero*: https://github.com/karpathy/nn-zero-to-hero
- NumPy broadcasting rules:
  https://numpy.org/doc/stable/user/basics.broadcasting.html
- NumPy matrix multiplication API:
  https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
- NumPy exponential API:
  https://numpy.org/doc/stable/reference/generated/numpy.exp.html
- Scheduled support lesson: 3Blue1Brown, *Derivative formulas through
  geometry*: https://www.youtube.com/watch?v=S0_qX4VJhMQ

The workbook assigns the lectures. Listing them records curriculum provenance;
it does not prove that Ajay watched or completed them. The code here was written
independently and does not copy the reference notebook or repository.
