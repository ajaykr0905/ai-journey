# Profile traces of the naive matmul on the Apple GPU

Day 75 of the curriculum is GPU MODE Lecture 1, "Profiling and Integrating
CUDA kernels in PyTorch", with the NVIDIA Nsight Systems video. The exercise
is "Profile your naive matmul. Identify the top bottleneck by name." This lab
is the trace half of that exercise, for a Mac with an Apple GPU.

The lab profiles the two naive Metal kernels of the Day 74 lab and
`torch.matmul` on the MPS device. It uses `torch.profiler`, exports Chrome
trace files, and analyzes them without torch. Then it names the top
bottleneck of each shape and compares it with the measured Day 74 ceilings.
Every timing is a host-specific Apple GPU (Metal) measurement. Nothing here
runs a CUDA kernel, uses an NVIDIA GPU, or uses Nsight.

## What the profiler can see on this host

The lecture uses the torch profiler first and a kernel profiler second. This
host has only the first. These are the tools on the development host:

- `torch.profiler` with `ProfilerActivity.CPU`. This PyTorch build has no MPS
  activity.
- `torch.mps.Event(enable_timing=True)`, which records events on the MPS
  stream.
- `torch.mps.profiler`, which writes OS signposts. Instruments is not on the
  host, so no tool can read the signposts. The lab does not use them.

There is no Nsight, no CUDA toolkit and no access to hardware counters.

`ProfilerActivity.CPU` sees only host-side work. It records the dispatch
calls and the time that the host waits in `torch.mps.synchronize()`. It
records no GPU kernel and no GPU kernel time. A kernel from
`torch.mps.compile_shader` does not appear as an op in the trace at all.
Thus the GPU work of a kernel shows only as host wait time.

The lab adds one device-side figure. For each call, it records two MPS
events around the dispatch and reads their elapsed time. That time comes from
the MPS stream, not from a hardware counter.

## Commands

```bash
python -m ai_journey.profile_trace_cli capture \
  --config config/day-75-profile-trace.json \
  --output-dir /tmp/day-75-profile-trace
python -m ai_journey.profile_trace_cli verify \
  --report /tmp/day-75-profile-trace/report.json \
  --config config/day-75-profile-trace.json
python -m ai_journey.profile_trace_cli analyze \
  --trace /tmp/day-75-profile-trace/trace-2-2048x2048x2048.json
```

The parser uses the program name `ai-journey-profile-trace`. An installed
wheel gives the same commands under that name.

The `capture` command profiles each shape of the config. It seals a report,
checks the report, and writes the report with its trace files into the
output directory. It prints these lines:

1. One line with the load averages before and after the capture.
2. One line for each shape, with the top op, its kind and its share of the
   iteration host time.
3. One line for each side of each shape, with the median host, dispatch,
   synchronize and MPS event times. The line also gives the host-observed
   GFLOP/s and the fraction of the Day 74 FMA ceiling.
4. A summary line with the report digest and the bottleneck of each shape.

In a measured run, the lines have the prefix `apple_gpu_metal_measurement`.
The summary line has the prefix `profile_trace_measurement=`. A run with a
stand-in session uses `synthetic_timing` instead.

The `analyze` command reads a Chrome trace with the event phases of the
parser, for example a `torch.profiler` export with `ProfilerActivity.CPU`.
It refuses a trace with other event phases, for example flow events.
It prints the top ops by self time, the gaps between top-level calls, the
host time of each iteration and a text timeline. The `verify` command checks
a report against the trace files in its directory. With `--config`, the
report must also come from that exact config. The `analyze` and `verify`
commands do not import torch.

The exit codes are:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a capture that failed an output check or had a device error.
- 3: this host cannot run Metal kernels through PyTorch, so `capture` stopped
  before it profiled anything.

Code 3 uses the probe of the Day 73 Metal lab. The probe names the first
missing requirement: macOS, torch, an MPS backend, an MPS device or
`torch.mps.compile_shader`.

## Capture

The capture uses the Day 74 helpers directly. `MatmulMetalRuntime` compiles
both kernels and binds one workload for each shape. The Day 74 functions
draw the operands and check each launch against its own pipeline. For each
shape, the steps are:

1. Run 2 untimed warmup iterations without the profiler.
2. Run 5 iterations under `torch.profiler` with `ProfilerActivity.CPU`.
3. Check the outputs as Day 74 does. The two naive outputs must have the same
   bits. The `torch.matmul` output must agree with them within its bound.
4. Export the trace. Replace the host name and the export path in it.
5. Parse the trace strictly and check its annotation layout.

The profiler cannot name the phases of a compiled kernel, so the capture
names them with `record_function`:

| Span | What it holds |
| --- | --- |
| `profile_trace/iteration` | One profiled iteration with all three sides |
| `profile_trace/<side>` | One side: a naive kernel or `torch.matmul` |
| `<side>/dispatch` | The call that puts the work in the queue |
| `<side>/synchronize` | The host wait in `torch.mps.synchronize()` |

The order of the sides moves by one place at each iteration, as in the Day
74 benchmark. Thus no side always runs first. For `torch.matmul`, the aten
ops `aten::matmul` and `aten::mm` nest in the dispatch span. The dispatch
spans of the naive kernels hold no child span.

## Parser

The parser accepts the event phases that `torch.profiler` writes: complete
events (`X`), instant events (`i`) and metadata events (`M`). It also
accepts pairs of `B` and `E` events, which open and close one span. It
refuses all other phases by name. The parser checks these items:

- the type of each field, and unknown or missing fields
- the process and thread IDs, as integers or strings
- the timestamps and durations, in microseconds
- the nesting of the spans on each thread

Torch writes times in microseconds with three decimals. The parser reads
them as exact decimals and converts them to integer nanoseconds. It refuses
a time that is not a whole nanosecond. Two spans on one thread must nest or
must not overlap. A partial overlap is an error.

The reader opens only regular files that are not symbolic links. The limits
are 32 MiB and 262144 events for each trace. The parser refuses duplicate
JSON keys, `NaN` and `Infinity`. Each refusal raises a named error, for
example `TracePhaseError` or `TraceNestingError`.

## Analysis

All times in the analysis are integer nanoseconds. The analysis gives these
figures for each trace:

- The total time and the self time of each op. Self time is the duration of
  a span minus the durations of its direct children on the same thread.
- The top N ops by self time. Total time counts a parent and its children, so
  it does not rank the ops.
- The gaps between consecutive top-level calls on each thread.
- The host time of each iteration, with the time of each direct child.
- A text timeline with one row for each thread and depth.

The analysis does not include the recording-window span of the profiler.
That span has the category `Trace` and covers the full capture.

## Bottleneck inference

The bottleneck of a shape is the op with the most self time. Its kind comes
from its name:

- `gpu_wait`: a synchronize span. The host waits for the GPU work of that
  side.
- `host_dispatch`: a dispatch span or an aten op in it.
- `host_other`: host time outside the two phases.

This is an inference from host timings and the measured Day 74 ceilings. It
is not a reading of hardware counters. For each side, the lab divides the
FLOPs by the median host time of the side. It compares that time with three
floors at the Day 74 ceilings:

| Floor | Calculation |
| --- | --- |
| FMA floor | `2 * M * N * K` FLOPs at 4618 GFLOP/s |
| Compulsory-traffic floor | A, B and C, each moved once, at 100.7 GB/s |
| No-reuse floor | Two operand reads for each multiply-add, plus C, at 100.7 GB/s |

The config holds the three Day 74 ceilings. They are the FMA ceiling and the
bandwidth ceiling from `docs/gpu-ceilings-lab.md`, and the `torch.matmul`
ceiling of 3963 GFLOP/s from `docs/matmul-metal-lab.md`. These values are
measurements on the development host, not vendor peaks.

## Reports and verification

A report is canonical JSON with a SHA-256 digest over all other fields. For
each shape, it names the trace file and gives its SHA-256 digest and its
size. It also holds the analysis, the timeline, the output check, the MPS
event times and the bottleneck. The digest shows drift and tampering. It is
not a signature.

The writer checks the report against the trace bytes before it writes
anything. Then it writes each trace and the report last, each with an atomic
replace. The verifier does not import torch. For each shape, the steps are:

1. Read the trace file and check its size and its SHA-256 digest.
2. Parse the trace and check the annotation layout. Make sure that the trace
   holds the replaced host name and export path.
3. Calculate the analysis, the timeline and the bottleneck again from the
   bytes. Each section must match the report as canonical JSON.
4. Calculate the op times and the gaps with a second method, an interval
   sweep. The two methods share no code.

The verifier cannot observe a timing again. It checks the MPS event times,
the load averages and the output check for structure and bounds only.

## One capture on the development host

This capture used the pinned config on 2026-11-18. The host was a shared
Darwin arm64 host with an Apple GPU, macOS 27.2, CPython 3.11.14, NumPy
1.26.4 and PyTorch 2.15.0. The capture held the GPU lock of this host. Other
jobs used the CPU at the same time. The capture took 8.1 seconds with the
torch import.

The load averages were 11.00, 12.79 and 12.49 before the capture. They were
11.32, 12.82 and 12.50 after the capture. These values are host-specific
Apple GPU (Metal) measurements from one capture, not portable figures.

Each cell gives the median of 5 profiled iterations, in microseconds. The
GFLOP/s column divides the FLOPs by the median host time of the side.

| Shape | Side | Host | Dispatch | Synchronize | MPS events | GFLOP/s |
| --- | --- | --- | --- | --- | --- | --- |
| 256 x 256 x 256 | `x_columns` | 686.8 | 21.2 | 617.5 | 550.3 | 48.9 |
| 256 x 256 x 256 | `x_rows` | 1426.7 | 67.1 | 1268.7 | 1196.9 | 23.5 |
| 256 x 256 x 256 | `torch.matmul` | 520.2 | 151.1 | 287.7 | 319.0 | 64.5 |
| 1024 x 1024 x 1024 | `x_columns` | 5832.3 | 37.6 | 5760.7 | 5530.8 | 368.2 |
| 1024 x 1024 x 1024 | `x_rows` | 11837.5 | 42.6 | 11628.7 | 11531.4 | 181.4 |
| 1024 x 1024 x 1024 | `torch.matmul` | 1169.4 | 225.9 | 874.7 | 768.6 | 1836.4 |
| 2048 x 2048 x 2048 | `x_columns` | 49682.0 | 140.1 | 49210.2 | 49003.9 | 345.8 |
| 2048 x 2048 x 2048 | `x_rows` | 120789.3 | 81.0 | 120612.6 | 120504.8 | 142.2 |
| 2048 x 2048 x 2048 | `torch.matmul` | 5527.1 | 257.8 | 5099.6 | 5086.6 | 3108.3 |

### Top bottleneck by name

At 2048 x 2048 x 2048, the top op is `matmul_x_rows/synchronize`. Its self
time was 602.3 ms of 879.4 ms of iteration host time, a share of 0.685. Its
kind is `gpu_wait`. Thus the host spent most of its time in a wait for the
`matmul_x_rows` kernel on the GPU.

At 1024 x 1024 x 1024, the top op is also `matmul_x_rows/synchronize`, with
a share of 0.616. In both shapes, `matmul_x_columns/synchronize` came second.

At the smallest shape, the top op is `matmul_x_columns/synchronize`, with a
share of 0.351. That result is not stable:

- Its self time was 6.74 ms, and `matmul_x_rows/synchronize` had 6.69 ms.
- The first profiled iteration took 8.6 ms. The other four took 2.3 to 3.2 ms.
- In the first iteration, the MPS events of `matmul_x_columns` gave 3.88 ms.
  In the other four, they gave 0.50 to 0.59 ms.

The warmup did not remove this cost of the first profiled iteration.

### What the figures show

These are the results of the capture:

- At 1024 and 2048, the synchronize span held 98% to 99.9% of the host time
  of each naive side. The dispatch took 38 to 140 us.
- At 2048, the MPS events of each side were within 0.5% of the synchronize
  time. Thus, at that size, the host wait probably shows the time of the
  GPU work. Neither timer reads a hardware counter, so this is an inference.
- At 256, a synchronize call took 288 to 1269 us, but a naive kernel needs
  much less time at the Day 74 rates. The fixed cost of the command and
  the wait probably sets the time at that size.
- The `torch.matmul` dispatch took 151 to 258 us, more than each naive
  dispatch. The aten ops nest in it.
- The gaps between iterations were 18 to 26 us at most, and 38 to 92 us in
  total for each shape. The host overhead between iterations is small.

### Comparison with the Day 74 ceilings

At 2048 x 2048 x 2048, `matmul_x_rows` reached 142.2 GFLOP/s. That rate is
0.031 of the FMA ceiling and 0.036 of the `torch.matmul` ceiling. Its median
host time was 32.5 times the FMA floor and 242 times the compulsory-traffic
floor. It was 0.177 of the no-reuse floor.

`matmul_x_columns` reached 345.8 GFLOP/s, 0.075 of the FMA ceiling. Its host
time was 13.4 times the FMA floor and 99 times the compulsory-traffic floor.
`torch.matmul` reached 3108.3 GFLOP/s, 0.673 of the FMA ceiling.

From these numbers, the lab makes this inference. Neither the FMA ceiling
nor the compulsory traffic explains the time of the naive kernels. Both
kernels are faster than the no-reuse floor, so caches probably serve most
of the operand reads. Both kernels do the same math and give the same bits.
At 2048, `matmul_x_rows` is 2.4 times slower, and its consecutive threads read
elements of A that are K floats apart. Thus the memory access pattern of
`matmul_x_rows` is the most probable cause. A check of this cause needs
hardware counters, and this host has none.

The Day 74 benchmark gave 406 GFLOP/s for `matmul_x_columns` and 161 GFLOP/s
for `matmul_x_rows` at 2048. This capture gave lower rates. Each side here
has one call and one synchronize call for each iteration, and the profiler
was active. Thus these rates include more host overhead than the Day 74
batched samples.

## Tests and recorded traces

The tests that need the GPU skip on other hosts. The other tests use a
stand-in session that writes the torch trace layout on a fake clock. One
recorded capture from the development host is in
`tests/test_profile_trace_recorded.py`. It holds the exact trace files and the
canonical report, so a host without torch can parse genuine torch exports
and check a genuine report.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `seed` | 75 | 0 to 2^32 - 1 |
| `group_size` | [32, 8] | [x, y], with at most 1024 threads in total |
| `shapes` | 256, 1024 and 2048, square | 1 to 3 different [m, n, k] triples |
| `warmup` | 2 | 0 to 10 untimed iterations |
| `iterations` | 5 | 2 to 20 profiled iterations |
| `analysis.top_n` | 8 | 1 to 64 ops |
| `analysis.gap_count` | 4 | 1 to 64 gaps for each thread |
| `analysis.timeline_width` | 96 | 16 to 240 columns |
| `analysis.timeline_depth` | 3 | 1 to 8 rows for each thread |
| `day_74_ceilings` | 4618.0 GFLOP/s, 100.7 GB/s, 3963.0 GFLOP/s | finite values above 0, at most 10^6 |
