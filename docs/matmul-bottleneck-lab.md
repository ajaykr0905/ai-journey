# Top bottleneck of the naive Metal matmul

Day 75 of the curriculum asks for the top bottleneck of the naive matmul, by
name. This lab names a bottleneck for each naive Metal kernel at each Day 74
sweep shape. It uses only measurements and a declared rule.

These names are inferences from timings and measured ceilings. The lab used
no hardware counters, no Nsight and no Instruments. Thus the lab did not
observe a stall, a cache hit or the occupancy of the GPU. Every Metal number in this
guide is a host-specific Apple GPU (Metal) measurement. Nothing here runs a
CUDA kernel or uses an NVIDIA GPU.

The rule gives one of these names:

- `dispatch-bound`: a fixed cost of one call fills most of the call.
- `memory-bound`: the rate is near the measured bandwidth roof at the
  intensity of the point.
- `compute-bound`: the rate is near the measured FMA ceiling.
- `latency or occupancy-bound`: the rate is below both roofs.
- `indeterminate`: the measurements do not support a name. The report gives
  the reason.

## Commands

```bash
python -m ai_journey.matmul_bottleneck_cli classify \
  --matmul-report /tmp/day-74-matmul-metal.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json \
  --gap-report /tmp/day-74-matmul-roofline.json \
  --timing /tmp/day-75-matmul-timing.json \
  --config config/day-75-matmul-bottleneck.json \
  --output /tmp/day-75-matmul-bottleneck.json
python -m ai_journey.matmul_bottleneck_cli verify \
  --report /tmp/day-75-matmul-bottleneck.json \
  --config config/day-75-matmul-bottleneck.json \
  --matmul-report /tmp/day-74-matmul-metal.json \
  --ceiling-report /tmp/day-74-gpu-ceilings.json \
  --gap-report /tmp/day-74-matmul-roofline.json \
  --timing /tmp/day-75-matmul-timing.json
```

The three Day 74 paths are the output paths in the guides of the
[matmul-metal lab](matmul-metal-lab.md), the
[gpu-ceilings lab](gpu-ceilings-lab.md) and the
[roofline gap](matmul-roofline-gap.md). The parser uses the program name
`ai-journey-matmul-bottleneck`.

The `classify` command prints these lines:

- one line with the two ceilings and the load averages of the ceiling run and
  of the timing run
- for each sweep point and kernel, one line with the name, the reason code
  and the numbers that set the name
- for each sweep point, one line with the verdict on the two mappings
- for each kernel, one line with its top name over the sweep
- one summary line with the report digest

In a measured report, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`matmul_bottleneck_measurement=`. When an input is synthetic, the lines use
`synthetic_timing` and `matmul_bottleneck_synthetic_timing=` instead.

The `verify` command calculates the report again from the inputs in the
report. With all four input paths, it also builds the report again from them.
The report must then have the same bytes, and the summary line says
`sources_checked=true`. The `verify` command takes all four input paths or
none of them.

The exit codes are:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks. A Day 74
  report that fails its own verifier also gives code 1.
- 2: a usage error, an unreadable or invalid input, inputs that do not pair,
  or a refused output.

Neither command runs a kernel or imports torch. Both commands need NumPy for
the Day 74 verifiers and the warp model.

## Inputs

The command reads three sealed Day 74 reports. Each report must pass its own
verifier before the command uses a number from it:

1. The matmul-metal report gives the threadgroup size and the SIMD width of
   each kernel.
2. The gpu-ceilings report gives the measured FMA ceiling and the measured
   bandwidth ceiling.
3. The roofline gap report gives the median sample time, the calls per
   sample, the median rate and the roofs of each point. The command builds
   the gap again from the two other reports, and the gap must be equal.

The fourth input is a timing file with a small declared schema. The
gpu-timing workstream of Day 75 makes the real file. For each shape and
kernel, the file holds three medians of one call, in nanoseconds:

| Field | Meaning |
| --- | --- |
| `median_gpu_ns` | GPU time of one call, for example between two `torch.mps.Event` records. Such an interval also holds the event signals. |
| `median_host_ns` | host time from an idle queue until the host sees the call complete |
| `median_dispatch_ns` | host time to issue the call, before the host waits |

The file also holds its labels, the environment tokens, the repeat count and
the load averages before and after its run. The file must hold each naive
kernel at each gap shape. It can also hold other entries. The environment
tokens must be equal to the tokens of the Day 74 reports.

## Rule

The rule takes these values for each point and kernel:

- The steady time is the Day 74 median sample time divided by the calls per
  sample. When a sample holds many calls, this value is near the time of one
  call in a series of calls.
- The overhead share is `1 - steady time / median_host_ns`.
- The fixed-share bound is the smallest `median_host_ns` of the kernel in the
  timing file, divided by `median_host_ns`.
- The rate is the Day 74 median rate.
- The bandwidth roof is the bandwidth ceiling times the intensity of the
  pinned traffic model. The lower roof is the lower of this roof and the FMA
  ceiling.

The rule applies these steps in this order. The first step that applies sets
the name:

1. `indeterminate` if the dispatch time is more than the host time. The same
   applies if the GPU time or the steady time exceeds the host time by more
   than the timing tolerance. The tolerance is a share of the host time.
2. `dispatch-bound` if the overhead share is at least the dispatch threshold.
3. `indeterminate` if the fixed-share bound is at least the dispatch
   threshold. Then a fixed cost can still fill most of the call.
4. `indeterminate` if the GPU time and the steady time differ by more than
   the agreement ratio.
5. `indeterminate` if the rate is above the FMA ceiling or the bandwidth roof
   by more than the roof tolerance.
6. `memory-bound` or `compute-bound` if the rate is at least the near-roof
   share of the lower roof. The lower roof sets the name. If the lower roof
   is a flagged FMA ceiling, the name is `indeterminate`.
7. `latency or occupancy-bound` in all other cases.

Step 3 is necessary because a Day 74 sample can hold one call. The steady
time of that sample includes the synchronization of the call. Thus the
overhead share is near 0, also for a small kernel. The fixed-share bound is
true only if the fixed cost does not change with the shape. The rule
cannot exclude a fixed cost if the timing file has no fast call.

Step 2 cannot find the cause of the overhead. The cause can be the host
dispatch, the completion wait, the launch latency of the GPU or a lower GPU
clock in one isolated call. Each decision records two parts of the overhead.
The first part is the host time outside the GPU time. The second part is the
GPU time above the steady time.

The name `latency or occupancy-bound` is a residual. It means that no
measured roof explains the rate. It does not prove that latency or occupancy
is the cause. A limit that the labs did not measure can also cause it, for
example the bandwidth of an on-chip cache.

A measured ceiling is a rate that the Day 74 kernels reached. The true limit
of the host can be higher. Thus "below a roof" stays true for the true limit.
"Near a roof" needs a ceiling that increases with the work. The rule gives
`indeterminate` for a point near the FMA ceiling if the ceiling lab flags its
FMA ceiling as fixed-cost bound.

The pinned config `config/day-75-matmul-bottleneck.json` holds each
threshold. The code has no default threshold.

| Field | Value | Use |
| --- | --- | --- |
| `rule.intensity_model` | `ideal_reuse` | traffic model of the bandwidth roof |
| `rule.dispatch_overhead_share_min` | 0.5 | dispatch threshold of steps 2 and 3 |
| `rule.timing_tolerance` | 0.05 | step 1 |
| `rule.gpu_agreement_max_ratio` | 1.5 | agreement ratio of step 4 |
| `rule.roof_excess_tolerance` | 0.05 | roof tolerance of step 5 |
| `rule.near_roof_share_min` | 0.7 | near-roof share of step 6 |
| `rule.require_fma_scales_with_work` | true | flagged FMA ceiling in step 6 |
| `mapping.slower_ratio_min` | 1.1 | mapping verdict |

The ideal-reuse model moves each element of A, B and C one time. No kernel
moves fewer bytes. Thus its roof is the highest bandwidth roof that a kernel
can have. A rate far below that roof is also below the roof of each real
traffic.

Each decision records its reason code, an explanation, every value above and
the value and limit of the step that set the name.

## Two mappings

The two naive kernels differ only in the axis that x covers. In `x_columns`,
x covers the columns of C. In `x_rows`, x covers the rows of C. The timings
show that `x_rows` is slower. They do not show why. The per-warp access model
of the [matmul model lab](matmul-model-lab.md) gives the reason.

The lab runs that model on the first warp of threadgroup (0, 0) of each
mapping at each shape. The model lists the lane addresses of each loop step
from the index formulas of the kernels. It counts the 32-byte sectors of each
request and models no cache. The model needs a SIMD group of 32 consecutive
x-threads. The matmul-metal lab makes the same assumption, and both pipelines
reported a SIMD width of 32.

| Mapping | A loads | B loads | C store |
| --- | --- | --- | --- |
| `x_columns` | broadcast, 1 sector each | coalesced, 4 sectors each | coalesced, 4 sectors |
| `x_rows` | strided by 4K bytes, 32 sectors each | broadcast, 1 sector each | strided by 4N bytes, 32 sectors |

For an aligned shape, one `x_columns` warp touches `5K + 4` sectors. One
`x_rows` warp touches `33K + 32` sectors, about 6.6 times as many. The verdict
`x_rows_worse_as_modeled` needs two facts:

- The model gives the `x_rows` warp more sectors.
- The median rate of `x_columns` is at least 1.1 times the median rate of
  `x_rows`.

Other combinations give a verdict that claims nothing and names the missing
fact. The model predicts the direction of the difference, not its size. The
model counts no cache, so caches probably served part of the strided loads.

## Results on the development host

The lab made these values on the development host on 2026-11-18. The steps
were:

1. The gpu-ceilings command ran the pinned Day 74 config. Its run began at
   08:14 IST.
2. The matmul-metal command ran the pinned Day 74 config. Its run began at
   08:15 IST.
3. A scratch capture timed one call of each kernel 11 times at each pinned
   shape and at 129 x 65 x 1000. It used a `torch.mps.Event` pair and
   `time.perf_counter_ns`. Its run began at 08:15 IST.
4. The roofline gap command and the `classify` command ran on the CPU.

All three timed runs held the GPU lock of the workstreams. The `uptime` load
averages were 11.70, 10.23 and 9.18 before the ceiling run. They were 12.51,
10.47 and 9.28 before the matmul-metal run. They were 10.90, 10.26 and 9.25
before the timing run and 17.61, 11.86 and 9.85 after it. The ceiling report
recorded 11.41, 10.19 and 9.17 before its run. The timing file recorded 10.51,
10.19 and 9.23 before its run.

The repository does not keep the scratch capture or the three Day 74 reports.
The tests keep the timing file and the bottleneck report as fixtures. The
report digest is
`bad8d9707745344c26a16676a757e1bb6c60c880afd3a10557cbdc34aa0ec2f5`. The gap
digest is
`94ca519c6033bb5dedbfca46da9111f4bdfaf5e1816d1453629aced833598341`.

The FMA ceiling was 4757 GFLOP/s, and the bandwidth ceiling was 102.6 GB/s.
The ceiling report flags its FMA ceiling as one that increases with the work.

| M x N x K | Mapping | Name | Overhead share | Fixed-share bound | GPU time / steady time | Lower-roof share |
| --- | --- | --- | --- | --- | --- | --- |
| 128 x 128 x 128 | `x_columns` | dispatch-bound | 0.956 | 1.000 | 12.41 | 0.095 |
| 128 x 128 x 128 | `x_rows` | dispatch-bound | 0.941 | 1.000 | 7.86 | 0.061 |
| 256 x 256 x 256 | `x_columns` | dispatch-bound | 0.858 | 0.583 | 4.92 | 0.068 |
| 256 x 256 x 256 | `x_rows` | dispatch-bound | 0.830 | 0.330 | 4.12 | 0.028 |
| 512 x 512 x 512 | `x_columns` | dispatch-bound | 0.515 | 0.297 | 1.75 | 0.075 |
| 512 x 512 x 512 | `x_rows` | dispatch-bound | 0.546 | 0.146 | 1.94 | 0.034 |
| 1024 x 1024 x 1024 | `x_columns` | latency or occupancy-bound | 0.038 | 0.077 | 0.99 | 0.079 |
| 1024 x 1024 x 1024 | `x_rows` | latency or occupancy-bound | 0.019 | 0.037 | 0.99 | 0.032 |
| 2048 x 2048 x 2048 | `x_columns` | latency or occupancy-bound | -0.008 | 0.011 | 0.99 | 0.084 |
| 2048 x 2048 x 2048 | `x_rows` | latency or occupancy-bound | 0.041 | 0.005 | 1.04 | 0.033 |
| 4096 x 4096 x 4096 | `x_columns` | latency or occupancy-bound | 0.120 | 0.001 | 1.14 | 0.070 |
| 4096 x 4096 x 4096 | `x_rows` | latency or occupancy-bound | 0.166 | 0.001 | 1.20 | 0.039 |
| 4096 x 1024 x 1024 | `x_columns` | latency or occupancy-bound | 0.023 | 0.021 | 1.00 | 0.084 |
| 4096 x 1024 x 1024 | `x_rows` | latency or occupancy-bound | -0.012 | 0.009 | 0.98 | 0.031 |
| 1024 x 4096 x 2048 | `x_columns` | latency or occupancy-bound | 0.035 | 0.010 | 1.03 | 0.083 |
| 1024 x 4096 x 2048 | `x_rows` | latency or occupancy-bound | 0.005 | 0.005 | 1.00 | 0.033 |
| 2000 x 3000 x 1000 | `x_columns` | latency or occupancy-bound | -0.030 | 0.015 | 0.96 | 0.081 |
| 2000 x 3000 x 1000 | `x_rows` | latency or occupancy-bound | -0.006 | 0.005 | 0.99 | 0.025 |

A negative overhead share shows a steady time above the one-call host time.
The two values are from separate runs, so they can differ by noise.

These are the results:

- The top name of each naive kernel was `latency or occupancy-bound`, at 6
  of 9 shapes.
- At 128, 256 and 512 cubed, both kernels were `dispatch-bound`. The 512
  cubed points were near the threshold, with overhead shares of 0.515 and
  0.546.
- At the 256 and 512 cubed points, the GPU time was 1.75 to 4.92 times the
  steady time. Thus most of their overhead was inside the event interval.
  The event signals, a GPU launch latency or a lower GPU clock in an
  isolated call can cause it.
- At the other 6 shapes, both kernels used at most 0.084 of the FMA ceiling.
  The ideal-reuse roof was above the FMA ceiling at these shapes.
- At all 9 shapes, the verdict was `x_rows_worse_as_modeled`. The model gave
  `x_rows` 6.60 to 6.61 times as many sectors. `x_columns` ran 1.57 to 3.24
  times as fast as `x_rows`.
- No point was `memory-bound`, `compute-bound` or `indeterminate`.

### Limits of the event timing

The `torch.mps.Event` pair does not read a GPU clock. The torch 2.15 headers
show that each event keeps a completion time that the host records. Thus
`median_gpu_ns` is a host interval between two event signals. It includes
the signal latency, so it is longer than the kernel time.

The scratch capture read the host clock before the first event record and
after the wait. Thus `median_host_ns` and `median_dispatch_ns` include the
cost of the two event records. The steady time does not include it, so the
event records increase the overhead share.

A check run measured that cost on 2026-11-18 from 08:54 IST. It held the GPU
lock and timed each naive kernel 21 times, with and without the event pair.
The `uptime` load averages were 22.12, 17.36 and 15.02 before it and 24.43,
17.92 and 15.23 after it. This load was about two times the load of the
recorded run. The repository does not keep the check script or its output.

The check run gave these results:

- At 128, 256 and 512 cubed, the event pair added 38 to 211 microseconds to
  the median host time of one call.
- At 128 and 256 cubed, the overhead shares without the event pair were at
  least 0.81. At 512 cubed, the `x_columns` share was 0.54.
- At 512 cubed, one `x_rows` call took 2.51 ms, against 3.65 ms in the
  recorded file. Its overhead share was 0.34 with the event pair and 0.33 without it.

Thus the `dispatch-bound` name of the 512 cubed `x_rows` point did not
repeat in the check run. The two runs had different loads, so the check run
does not replace the recorded result. The names at 128 and 256 cubed did not
change.

The tests also join this timing file with the small Day 74 fixture from
2026-11-17. That fixture holds one call per sample at 256 x 256 x 256. Step 3
names that `x_columns` point `indeterminate`. At 129 x 65 x 1000, the steady
times of the fixture are above the host times of this file. Thus step 1
names both points `indeterminate`.

## Reports and verification

A bottleneck report is canonical JSON with sorted keys, compact separators
and ASCII escapes. It holds a SHA-256 digest over every other field. The
digest shows drift and tampering. It is not a signature. The report holds
these sections:

- the full config and its digest
- the digest of each Day 74 report, the digest of the timing file and the
  load averages of the ceiling run and the timing run
- the fastest call of each kernel in the timing file
- the two ceilings, the threadgroup size and the SIMD widths
- for each point, the inputs, the decision of each kernel and the warp model
  of both mappings
- a summary with the top name of each kernel and the count of each verdict

The verifier calculates each decision again with its own copy of the rule.
It calculates the warp model again with NumPy from the index formulas. A
report with changed inputs and a new digest still passes these checks. The
`verify` command with all four inputs finds that change, because it builds
the report again from the verified inputs.

## Evidence boundary

The evidence boundary of a measured report states these facts:

- Each name is an inference from Apple GPU (Metal) timings and from the
  ceilings that the Day 74 ceiling lab measured on the same host.
- The lab used no hardware counters, no Nsight and no Instruments. Thus it
  observed no stall, no cache hit and no occupancy.
- The mapping explanation uses the Day 74 per-warp access model, not
  measured traffic.
- No CUDA kernel ran, and the lab used no NVIDIA GPU.
- The names do not describe CUDA, NVIDIA, distributed or production
  performance.

The names are from one Apple GPU on one shared host with load averages near
10. They are not portable figures. They are not evidence of CUDA or NVIDIA
performance.
