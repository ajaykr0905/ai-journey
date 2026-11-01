# AI infrastructure program ledger

This ledger counts only independently reviewable source commits created after
`0bb84ad3e0fa37b91bdb19ff050e8ac7977c634b`. Merge, setup, glossary, generated,
and mechanical commits are excluded. The milestone parent is
[#53](https://github.com/ajaykr0905/ai-journey/issues/53).

All listed commits use `Ajay Kumar <206070437+ajaykr0905@users.noreply.github.com>`.
GitHub attribution is expected from that public noreply address; final remote
association must be confirmed after publication. The focused checks below are
the public behavioral seams rerun on the integrated milestone. Full-suite,
installed-wheel, readiness, and static-check status is recorded after the table.

| Count | Ticket | Source commit | Independent problem solved | Changed files | Focused checks | Evidence |
|---:|---|---|---|---|---|---|
| 1 | [#54](https://github.com/ajaykr0905/ai-journey/issues/54) | `b10f527a1d33996113bff7e6f9340412eade2acc` — Define hybrid parallel topology contracts | Admits topology dimensions and derives deterministic rank coordinates and peer groups. | `src/ai_journey/hybrid_topology.py`; `tests/test_hybrid_topology.py` | `pytest -q tests/test_hybrid_topology.py` | Analytical topology evidence only. |
| 2 | [#57](https://github.com/ajaykr0905/ai-journey/issues/57) | `d8dff3b18dc978ac4ec83a7737ea242929ea14e7` — Partition transformer layers across pipeline stages | Assigns every layer once to a contiguous stage and reports imbalance. | `src/ai_journey/pipeline_partition.py`; `tests/test_pipeline_partition.py` | `pytest -q tests/test_pipeline_partition.py` | Analytical partition evidence only. |
| 3 | [#56](https://github.com/ajaykr0905/ai-journey/issues/56) | `c065a22531b6a7296cd7cb4cfbda689c980a9e80` — Plan reversible sequence dimension shards | Preserves token order and explicit padding across deterministic sequence shards. | `src/ai_journey/sequence_sharding.py`; `tests/test_sequence_sharding.py` | `pytest -q tests/test_sequence_sharding.py` | Analytical sharding evidence only. |
| 4 | [#55](https://github.com/ajaykr0905/ai-journey/issues/55) | `6f2bf3a4c80eae65eeadd790d4fff6647662efa1` — Plan reversible tensor parameter shards | Preserves every tensor element across bounded even and uneven shards. | `src/ai_journey/tensor_sharding.py`; `tests/test_tensor_sharding.py` | `pytest -q tests/test_tensor_sharding.py` | Analytical sharding evidence only. |
| 5 | [#62](https://github.com/ajaykr0905/ai-journey/issues/62) | `ca1ee2006baafbc6ed53430cb9e9079101d9c222` — Validate activation handoff contracts between stages | Binds bounded activation bytes to adjacent-stage metadata and digests. | `src/ai_journey/activation_handoff.py`; `tests/test_activation_handoff.py` | `pytest -q tests/test_activation_handoff.py` | Contract admission; no communication claim. |
| 6 | [#58](https://github.com/ajaykr0905/ai-journey/issues/58) | `11b9b9a48249b25058f19ddded91bd3823385afd` — Generate validated GPipe microbatch schedules | Emits every dependency-valid GPipe operation exactly once. | `src/ai_journey/pipeline_schedule.py`; `tests/test_pipeline_schedule.py` | `pytest -q tests/test_pipeline_schedule.py` | Logical schedule evidence, not runtime performance. |
| 7 | [#63](https://github.com/ajaykr0905/ai-journey/issues/63) | `d0bec000bcc181203bd93bc4c3b715347fd12fc0` — Verify pipeline handoffs over CPU Gloo | Supervises one bounded two-rank CPU/Gloo transfer and reaps failures. | `src/ai_journey/pipeline_handoff_probe.py`; `tests/test_pipeline_handoff_probe.py` | `pytest -q tests/test_pipeline_handoff_probe.py` (real local Gloo) | Executed local CPU/Gloo observation; no accelerator or performance claim. |
| 8 | [#59](https://github.com/ajaykr0905/ai-journey/issues/59) | `edfa5b8bb5e592b2ee1c2847b01c7f6a891bfef2` — Generate validated 1F1B pipeline schedules | Emits dependency-valid warmup, steady, and cooldown operations. | `src/ai_journey/pipeline_schedule.py`; `tests/test_pipeline_schedule.py` | `pytest -q tests/test_pipeline_schedule.py` | Logical schedule evidence, not executed pipeline training. |
| 9 | [#60](https://github.com/ajaykr0905/ai-journey/issues/60) | `987ad71a57207fa9060766f18a82f181dab1af64` — Measure pipeline bubbles from executable schedules | Derives busy, idle, bubble, and utilization values from validated slots. | `src/ai_journey/pipeline_metrics.py`; `tests/test_pipeline_metrics.py` | `pytest -q tests/test_pipeline_metrics.py` | Analytical schedule utilization, not measured device utilization. |
| 10 | [#61](https://github.com/ajaykr0905/ai-journey/issues/61) | `1187a7267461e455435f4d5c6fdbe097bf51a367` — Account for hybrid parallel communication volume | Separates tensor, sequence, and pipeline traffic with explicit modeled wire assumptions. | `src/ai_journey/parallel_communication.py`; `tests/test_parallel_communication.py` | `pytest -q tests/test_parallel_communication.py` | Analytical bytes; no bandwidth, latency, or collective execution claim. |
| 11 | [#64](https://github.com/ajaykr0905/ai-journey/issues/64) | `22291d888cf920774efa1ee3eb5c74372e64df25` — Publish sealed hybrid parallelism evidence | Canonically seals bounded primary inputs and replayable analytical fields. | `src/ai_journey/hybrid_parallel_io.py`; `src/ai_journey/hybrid_parallel_report.py`; `tests/test_hybrid_parallel_io.py`; `tests/test_hybrid_parallel_report.py` | `pytest -q tests/test_hybrid_parallel_io.py tests/test_hybrid_parallel_report.py` | Seal proves integrity and internal consistency, not execution authenticity. |
| 12 | [#65](https://github.com/ajaykr0905/ai-journey/issues/65) | `6ac850dcc48709d7161fb2cf54650cbcd548ff39` — Gate installed parallelism workflows in CI | Operates generation and offline verification from an isolated installed wheel. | `.github/workflows/ci.yml`; `config/deployment_readiness.json`; `docs/hybrid-parallelism.md`; `pyproject.toml`; `src/ai_journey/hybrid_parallel_cli.py`; `src/ai_journey/hybrid_parallel_verify.py`; `tests/test_hybrid_parallel_ci_contract.py`; `tests/test_hybrid_parallel_cli.py`; `tests/test_hybrid_parallel_docs.py`; `tests/test_hybrid_parallel_package.py` | `pytest -q tests/test_hybrid_parallel_cli.py tests/test_hybrid_parallel_package.py tests/test_hybrid_parallel_ci_contract.py tests/test_hybrid_parallel_docs.py` | Installed generation executes CPU/Gloo; offline verification does not rerun or authenticate it. |

## Milestone verification and publication

- Full repository suite: **3,878 passed and 680 subtests passed**. The required
  final integration-tip merge was already up to date; the post-merge affected
  suite passed **42 tests**.
- Real CPU/Gloo probe and isolated-wheel generation/verification: **passed**.
- Scoped Ruff, formatting, compilation, deployment readiness, complete-diff
  inspection, and secret-sensitive review: recorded in the repair commit's final
  verification; publication CI remains pending.
- Pull request and CI URLs: pending publication of the milestone integration branch.
- Cumulative qualifying source commits: **12 / 70**.
- Exact remaining program shortfall: **58 meaningful source commits**.

## Evidence Boundaries

The topology, sharding, partition, scheduling, Pipeline Bubble, and communication
results are deterministic analytical evidence. The generation command performs
one bounded local two-rank CPU/Gloo activation handoff. A retained report records
that observation, while its seal and offline verifier establish integrity and
internal consistency only; the retained report does not independently authenticate execution.

This milestone provides no NVIDIA, CUDA, NCCL, multi-GPU, multi-node, bandwidth,
throughput, latency, measured utilization, memory, cost, production-readiness,
lecture-completion, manual-exercise, or learner-understanding evidence. Ajay must
provide direct hardware or learner evidence before any such claim can be added.
