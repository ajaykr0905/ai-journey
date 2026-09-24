# Open-source bridge

This file connects repository exercises to public AI-infrastructure problems
without claiming that a small local experiment solves an upstream system.

## Active contribution

| Project | Work | Status | Local evidence |
|---|---|---|---|
| PyTorch TorchTitan | [PR #4864](https://github.com/pytorch/torchtitan/pull/4864): collect CPU-safe RL unit tests and lazy-load optional vLLM exports | Open; CLA and Meta checks passed; maintainer workflow approval and review pending | 147 focused tests passed after syncing with current `main` |

## Project connections

| Public problem | Repository analogue | Honest boundary |
|---|---|---|
| TorchTitan [#4801](https://github.com/pytorch/torchtitan/issues/4801): packed-document isolation in sparse attention | Day 21 audits per-record boundaries and an intentionally broken packed character stream | The lab is not transformer attention and does not fix the issue |
| vLLM [#27433](https://github.com/vllm-project/vllm/issues/27433): output invariance across batch layouts | Days 13 and 21 use deterministic inputs, independent controls, and exact repeatability checks | No supported-GPU vLLM validation has been run |
| TorchTitan [#4828](https://github.com/pytorch/torchtitan/issues/4828): RL tests missing from CPU/GPU collection | Day 21 and earlier days wire focused tests into the repository-wide readiness path | Local CI structure is much smaller than TorchTitan's workflow matrix |

## Working policy

1. Maintain active upstream work before claiming another issue.
2. Reproduce an issue or verify a documentation gap before posting publicly.
3. Prefer deterministic CPU tests when they genuinely cover the behavior.
4. State hardware, dependency, and scope limitations beside every result.
5. Keep one focused change per upstream pull request.
6. Do not create activity-only commits or repeated maintainer pings.
