# Three clocks for one kernel on the Apple GPU

Day 75 of the curriculum asks how long a GPU kernel takes. On a GPU, that
question has more than one answer. A launch call on the host only puts the
work in a queue. The GPU runs the work later. This lab times the same launch
on three clocks and keeps the clocks apart.

The lab times the two naive Metal matmul kernels of Day 74 and `torch.matmul`
on the MPS device. It also times a near-empty kernel. Every timing is a
host-specific Apple GPU (Metal) measurement at the load averages that the
report records. Nothing here runs a CUDA kernel or uses an NVIDIA GPU. The lab
reads no hardware counters.

## Commands

```bash
python -m ai_journey.gpu_timing_cli run \
  --config config/day-75-gpu-timing.json \
  --output /tmp/day-75-gpu-timing.json
python -m ai_journey.gpu_timing_cli verify \
  --report /tmp/day-75-gpu-timing.json \
  --config config/day-75-gpu-timing.json
```

The parser uses the program name `ai-journey-gpu-timing`. An installed
wheel gives the same commands under that name.

The `run` command compiles the kernels, times them, checks its own report and
writes the report. It prints these lines:

1. One line with the load averages before and after the run.
2. One line for the empty kernel, with the median and the IQR on each clock.
3. One line for each shape and side, with the median and the IQR on each
   clock, the regime and the cross-domain difference.
4. One line for each side, with the fitted fixed overhead and the largest
   relative residual on each clock.
5. A summary line with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`gpu_timing_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `gpu_timing_synthetic_timing=` instead. The `verify`
command recomputes a report on the CPU and prints the summary line with the
prefix `verified_`. With `--config`, `verify` also requires that the report
agrees with that exact config.

The exit codes are:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or used all of
  its time budget.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it timed anything.

## Hosts without MPS

The `run` command needs an Apple GPU on macOS and a PyTorch build with MPS and
`torch.mps.compile_shader`. The runtime uses the Day 73 probe. On a host
without these, `run` writes the reason to standard error and exits with code
3. A Linux CI runner gets `unsupported_operating_system` without a torch
import. The `verify` command never imports torch, so it works on any host
with NumPy.

The tests that need the GPU skip on other hosts and give the reason. The other
tests use a stand-in backend with a virtual clock. One test file holds a real
report from the development host, and that report verifies on Linux as a
measurement.

## The three clocks

Each sample holds one launch and reads one clock. Thus no clock reads its
time inside the interval of a different clock.

| Clock | Domain | Interval |
| --- | --- | --- |
| `mps_event` | `mps_event_stream` | From an event before the launch to an event after it |
| `host_synchronized` | `host_perf_counter_ns` | From the launch call to the return of `torch.mps.synchronize()` |
| `host_dispatch` | `host_perf_counter_ns` | The launch call alone, with no synchronize in the interval |

The event clock uses two `torch.mps.Event(enable_timing=True)` objects. The
lab records the first event, launches the kernel, records the second event
and then waits for the queue. Then it reads `elapsed_time` in milliseconds. The report keeps
the raw milliseconds and the integer nanoseconds from `round(ms * 1e6)`.

The PyTorch header says that a completion handler of the command buffer
stores the time of each event. The header does not name the clock that the
handler reads. Thus the lab treats the event clock as a different domain from
`time.perf_counter_ns`.

The event interval is not GPU time only. It also holds the host time between
the two `record()` calls, and that time includes the launch call. A test in
`tests/test_gpu_timing_runtime.py` shows it. The test waits 20 ms on the host
between the two calls, and the event interval is then 15 ms or more.

The lab never subtracts an event time from a host time without a label. It
does one such subtraction, `host_minus_event`. Its record has
`cross_domain: true` and names both clocks. The two medians come from
different samples on two clocks. Thus the difference is only an estimate of
the host cost around the GPU work.

The queue is empty at the beginning of each sample. The lab uses these steps for one
repeat:

1. It visits the sides in an order that moves by one side at each repeat.
2. For each side, it visits the clocks in an order that moves by one clock at
   each repeat.
3. It records each value in the list of that side and clock.

Before the timed repeats, each side runs `warmup` samples on each clock
without a record. These samples pay the one-time costs, for example the
creation of pipelines and MPS graphs.

## The empty kernel

The empty kernel is in `gpu_timing_source.py`. It has one buffer and one
compare. It writes to the buffer only at a thread position that no launch of
this lab has. The compiler cannot know the thread position, so it must keep
the store and the buffer. The lab launches one thread in one threadgroup.

This kernel gives the floor of one `torch.mps.compile_shader` launch on this
host. The lab times it before the sweep, on all three clocks.

## Fixed overhead and the size-dependent part

The lab fits one straight line for each side and clock:
`median_ns = intercept_ns + slope_ns_per_flop * flops`. The intercept is the
estimate of the fixed overhead. The rest of each median is the
size-dependent part.

The sweep goes from 2 FLOPs to more than 10^11 FLOPs. An unweighted line
follows the largest points and gives an intercept with large noise. Thus the
fit uses weighted least squares with the weight `1 / median_ns**2`. This
weight makes the fit minimize relative residuals. Each sum uses `math.fsum`,
so the verifier gets the same bits with its own code.

The report gives these values for each point:

- `fixed_ns`: the intercept of the line for that side and clock.
- `size_ns`: the median minus `fixed_ns`.
- `fixed_share`: `fixed_ns` divided by the median.
- `regime`: `fixed_overhead_dominant` when `fixed_share` is 0.5 or more,
  otherwise `size_dependent_dominant`.

A straight line is a model. The naive kernels use the caches in different
ways as the matrices grow. Thus the report gives the largest relative
residual of each line. A `fixed_share` above 1 means that the line is higher
than the median at that point. The regime is an inference from the line, not
a hardware counter reading.

The report also gives the direct floor. This floor is the empty-kernel median
on each clock and the median of each side at the smallest shape. The ratio
`intercept_to_empty_kernel_ratio` compares each intercept with the empty
kernel on the same clock.

## Statistics

For each side and clock, the report holds `repeats` timed samples. The statistics use the
nearest-rank rule of Days 73 and 74, with no interpolation. The median is the
50th percentile. The IQR is the 75th percentile minus the 25th percentile.

A dispatch sample times a strict part of what a synchronized sample times.
Thus a side with a dispatch median above its synchronized median stops the
run, and the verifier rejects it.

## Bottleneck inference

For each point and side, the report divides the work by the event median.
That gives `event_gflops_per_second`. It divides that rate by the Day 74 FMA
ceiling, which the config holds, to give `event_fraction_of_fma_ceiling`.

The Day 74 FMA ceiling is 4618 GFLOP/s, from run 1 in
[the ceilings lab](gpu-ceilings-lab.md). It is a measurement on the same host,
not a vendor peak. A small fraction shows that a side is far below the
compute ceiling. It does not show the cause. This lab reads no hardware
counters, so every bottleneck statement here is an inference from timings and
from the Day 74 ceilings.

## Reports and verification

The report records these items:

- the config and its SHA-256 digest
- the digest of the Day 74 matmul source, and the empty-kernel source with its
  digest
- the pipeline limits of the three kernels
- the clock rules, the domain rules and the sample order
- the load averages before and after the run, from `os.getloadavg()`
- the raw samples of every side on every clock
- the analysis and a summary

A SHA-256 digest seals the report. The digest shows changes to the file. It is
not a signature, and it cannot prove that a timing occurred.

The verifier is in `gpu_timing_verify.py`. It does not import torch or call a
module that times anything. It does these checks:

1. It checks that each clock has `repeats` positive integer samples.
2. It checks that each event sample is `round(ms * 1e6)` of its raw value.
3. It calculates each statistic again from the raw samples.
4. It calculates each fit, split, ratio, rate and fraction again with its own
   code.
5. It calculates the launch geometry, the planned work and the run totals
   again.
6. It compares each section with the stored section as canonical JSON.

Only a report from `torch.mps.compile_shader` with `time.perf_counter_ns`
verifies as a measurement, with the label
`host-specific Apple GPU (Metal) measurement` and the scope
`apple_gpu_metal_this_host`. A report from a stand-in or an injected clock
verifies only as synthetic, with the scope `synthetic_backend_or_clock`.

### Schema versions

New reports have `schema_version` 2. The two versions are different only in
the sealed text of the `mps_event_stream` domain:

- The version 1 text calls the event interval "device-side". But the
  section about the three clocks shows that the interval also holds host
  time.
- The version 2 text says that the interval is between two points in the MPS
  command stream. It also says that the interval can hold host time between
  the two `record()` calls, and that this time includes the launch call.

The verifier accepts each version only with its own text. Thus the recorded
report in `tests/test_gpu_timing_recorded.py` still verifies as version 1. A
report with the text of the other version fails the checks. The runs in this
guide are from 2026-11-18, before version 2, so their reports have version 1.

## Controls

The pinned config is `config/day-75-gpu-timing.json`.

| Field | Pinned value | Bounds |
| --- | --- | --- |
| `seed` | 75 | 0 to 2^32 - 1 |
| `launch.group_size` | [32, 8] | at most 1024 threads |
| `sweep.shapes` | 12 cubes, 1 to 4096 | 3 to 16 shapes, FLOPs strictly increase, the first shape has at most 1024 FLOPs |
| `sweep.warmup` | 3 | 0 to 10 |
| `sweep.repeats` | 15 | 3 to 50 |
| `sweep.max_runtime_seconds` | 600 | 1 to 600 |
| `reference.fma_ceiling_gflops_per_second` | 4618.0 | a float above 0 and at most 10000000 |

The pinned sweep uses the cubes 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
and 4096. The planned work has 37 sides, 1665 timed samples, 1998 launches and
25445840525892 FLOPs. Each sample holds one launch, because the cost of one
launch is the subject of this lab.

## Two runs on the development host

These runs used the pinned config on 2026-11-18, under the shared GPU lock.
The host was a shared Darwin arm64 host with an Apple GPU, macOS 27.2, CPython
3.11.14, NumPy 1.26.4 and PyTorch 2.15.0. Other jobs ran on the host at the
same time, and the load averages show it. All values are host-specific Apple
GPU (Metal) measurements, not portable figures.

| Run | Time (IST) | Load averages before (1, 5, 15 min) | Load averages after |
| --- | --- | --- | --- |
| 1 | 08:32:09 to 08:33:27 | 9.87, 13.62, 12.65 | 10.53, 12.82, 12.41 |
| 2 | 08:33:54 to 08:35:10 | 11.40, 12.80, 12.42 | 11.66, 13.06, 12.58 |

Each run took 76 to 78 seconds, with the torch import. Each report was about
98 KB, and `verify` accepted both reports as measurements.

### The launch floor

| Run | Empty kernel event | Empty kernel synchronized | Empty kernel dispatch | `host_minus_event` |
| --- | --- | --- | --- | --- |
| 1 | 64.9 us | 115.5 us | 4.9 us | 50.6 us |
| 2 | 69.6 us | 136.5 us | 5.9 us | 66.9 us |

The dispatch call of the empty kernel took about 5 to 6 us. The caller
waited about 115 to 137 us for a kernel that did no work. The event interval
was about 65 to 70 us. Thus the event interval includes more than the
kernel. It includes the launch call on the host, as the section about the
three clocks shows. It probably also includes a part of the submission path of
the MPS stream.

### Medians of run 1

The table gives the median in microseconds on the event, synchronized and
dispatch clocks.

| Shape | `matmul_x_columns` | `matmul_x_rows` | `torch.matmul` |
| --- | --- | --- | --- |
| 1 x 1 x 1 | 65.8 / 115.9 / 5.5 | 67.1 / 114.2 / 5.3 | 69.2 / 123.5 / 4.9 |
| 4 x 4 x 4 | 89.2 / 194.7 / 12.5 | 103.9 / 176.2 / 7.0 | 106.7 / 229.8 / 48.0 |
| 8 x 8 x 8 | 68.7 / 130.5 / 7.2 | 70.0 / 133.5 / 5.8 | 89.6 / 162.2 / 27.9 |
| 16 x 16 x 16 | 88.4 / 148.2 / 7.7 | 81.0 / 156.1 / 7.6 | 151.8 / 225.1 / 68.0 |
| 32 x 32 x 32 | 87.9 / 162.0 / 6.0 | 89.4 / 159.6 / 5.3 | 107.8 / 168.6 / 24.6 |
| 64 x 64 x 64 | 108.5 / 170.2 / 5.9 | 109.9 / 164.3 / 5.5 | 102.6 / 162.9 / 30.1 |
| 128 x 128 x 128 | 141.5 / 200.8 / 5.8 | 179.9 / 235.2 / 6.0 | 111.0 / 172.5 / 32.9 |
| 256 x 256 x 256 | 212.3 / 282.5 / 6.2 | 468.7 / 569.9 / 6.4 | 120.2 / 163.1 / 32.3 |
| 512 x 512 x 512 | 762.4 / 864.7 / 8.1 | 1820.6 / 1940.1 / 11.2 | 174.6 / 253.2 / 47.9 |
| 1024 x 1024 x 1024 | 5571.8 / 5667.9 / 36.5 | 11359.9 / 11518.6 / 36.9 | 681.0 / 878.8 / 105.7 |
| 2048 x 2048 x 2048 | 42204.1 / 42388.2 / 38.8 | 106773.3 / 107356.4 / 39.0 | 4337.0 / 4597.6 / 141.0 |
| 4096 x 4096 x 4096 | 428201.3 / 416757.0 / 46.9 | 737336.6 / 737962.9 / 62.3 | 35697.0 / 36202.5 / 157.4 |

### Fits and regimes

| Side | Fixed overhead, event (runs 1, 2) | Fixed overhead, synchronized (runs 1, 2) | Fixed overhead, dispatch (runs 1, 2) | First size-dependent shape |
| --- | --- | --- | --- | --- |
| `matmul_x_columns` | 83.3, 86.8 us | 150.0, 155.0 us | 6.7, 9.3 us | 256 x 256 x 256 |
| `matmul_x_rows` | 84.3, 83.0 us | 150.0, 158.9 us | 6.3, 7.8 us | 128 x 128 x 128 |
| `torch.matmul` | 97.4, 113.3 us | 164.3, 191.3 us | 9.2, 11.0 us | 1024 x 1024 x 1024 |

The first size-dependent shape is the first shape whose event median has the
regime `size_dependent_dominant`. Both runs gave the same shape for each side.
These are the results:

- The fitted event overhead of the naive kernels was 83 to 87 us. That is
  1.2 to 1.3 times the empty-kernel event median.
- Below 128 x 128 x 128, the fixed overhead was the larger part of every
  event median.
- `torch.matmul` stayed in the fixed regime up to 512 x 512 x 512.
- The largest relative residual of the event lines was 0.33 to 0.41 in run 1
  and 0.43 to 0.66 in run 2. The line is a rough model at the small shapes.

The dispatch lines had residuals of 0.75 to 0.91, because the dispatch time
does not follow the work. These are the dispatch medians in run 1:

- The naive kernels took 5.3 to 12.5 us up to 512 cubed. From 1024 cubed,
  they took 36.5 to 62.3 us.
- `torch.matmul` took 4.9 us at 1 cubed and 24.6 to 68.0 us from 4 to 512
  cubed. From 1024 cubed, it took 105.7 to 157.4 us.

Run 2 had the same pattern with higher values at some small shapes, up to
24.4 us for a naive kernel and 119.4 us for `torch.matmul`.

### Rates at the largest shape

| Side | Event GFLOP/s at 4096 (runs 1, 2) | Fraction of the 4618 GFLOP/s FMA ceiling |
| --- | --- | --- |
| `matmul_x_columns` | 321.0, 351.3 | 0.069, 0.076 |
| `matmul_x_rows` | 186.4, 187.2 | 0.040, 0.041 |
| `torch.matmul` | 3850.2, 3926.7 | 0.834, 0.850 |

At 2048, the event rate of `matmul_x_columns` was 407 GFLOP/s in run 1. Day 74
measured 406 GFLOP/s at that shape with batched, synchronized samples. At
4096, both naive kernels were below 8% of the FMA ceiling. Their time grew
with the work, not with the fixed overhead. From these timings, the lab infers
that a size-dependent cost limits the naive kernels, and that cost is not the
FMA rate. The [Day 74 roofline gap](matmul-roofline-gap.md) puts the naive
kernels under the bandwidth roof. This lab does not measure memory traffic.

### Two clocks on a loaded host

At 4096, `matmul_x_columns` in run 1 had an event IQR of 70.0 ms. Its
synchronized median was 11.4 ms below its event median. Thus
`host_minus_event` was negative for that side. The two medians came from
different samples, and the host load changed between them. This result shows
why the lab labels each cross-domain difference. In run 2, the same value was
2.0 ms.

## Evidence boundary and exclusions

- Every number is a host-specific Apple GPU (Metal) measurement at the load
  averages in its report.
- No CUDA kernel ran. The lab used no NVIDIA GPU and read no hardware counter.
- The lab did not use Nsight, Instruments or the MPS profiler signposts.
- The FMA ceiling is a measurement from Day 74 on this host, not a vendor
  peak.
- A fixed overhead, a regime or a bottleneck is an inference from timings.
- The lab makes no claim about distributed, production or portable
  performance.

## Sources

- [The Day 74 naive matmul lab](matmul-metal-lab.md), for the kernels, the
  workload and the agreement bound.
- [The Day 74 ceilings lab](gpu-ceilings-lab.md), for the FMA ceiling and the
  load-average record.
- [The Day 74 roofline gap](matmul-roofline-gap.md), for the bandwidth roof of
  the naive kernels.
- [The Day 73 Metal lab](kernel-metal-lab.md), for the probe, the launch
  geometry and the timing rules.
- The PyTorch MPS header `ATen/mps/MPSEvent.h` in PyTorch 2.15.0, for the
  completion handler that stores event times.
