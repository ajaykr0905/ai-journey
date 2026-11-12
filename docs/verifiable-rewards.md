# Verifiable reward and recovery lab

This Day 69 workflow applies deterministic rewards to AI-engineering answers that
can be checked without a judge model. It exercises three verifier classes:
normalized exact text, finite decimal values, and strict JSON. Candidate groups
produce group-relative advantages using the GRPO normalization shape. The code
does not train a policy or claim that the fixture responses came from a model.

Run the pinned fixture and seal its evidence:

```bash
ai-journey-verifiable-rewards build \
  --config config/day-69-verifiable-rewards.json \
  --output /tmp/day-69-reward-report.json
ai-journey-verifiable-rewards verify \
  --config config/day-69-verifiable-rewards.json \
  --report /tmp/day-69-reward-report.json
ai-journey-verifiable-rewards replay \
  --config config/day-69-verifiable-rewards.json \
  --report /tmp/day-69-reward-report.json
```

The fixture injects one transient verifier failure and one permanent failure.
Transient work retries only within the declared attempt budget. Group-boundary
checkpoints bind the task set and committed attempt stream, and the recovery test
requires an exact match with uninterrupted evaluation. Reports include canonical
input identities, a hash-chained attempt journal, reconciled final-candidate
metrics, group-relative advantages, and a SHA-256 seal.

These rewards fit tasks whose correctness can be recomputed: checkpoint loss,
tensor shapes, resource arithmetic, structured recovery decisions, and exact
protocol terms. They do not establish helpfulness, safety, reasoning quality, or
general model quality. Real GRPO evidence would also require a policy, reference
controls, training measurements, held-out evaluation, and hardware provenance.

Ajay still needs to personally watch CS336 Lecture 17 and the support video,
choose a domain task, justify why its reward cannot be gamed, inspect false positives and false negatives,
and write his own explanation of GRPO and verifiable rewards. The repository
report is engineering evidence, not learner evidence or a personal reflection.
