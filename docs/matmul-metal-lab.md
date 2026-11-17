# Naive matmul kernels on the Apple GPU

Day 74 of the curriculum asks for a naive matmul kernel and its GFLOP/s. It
also asks how far below peak the kernel is. This lab is the Metal half of that
exercise, for a Mac with an Apple GPU. It compiles two naive float32 kernels at
run time through `torch.mps.compile_shader`. It checks them against a float64
reference and against `torch.matmul` on the MPS device and on the CPU. Then it
times them against `torch.matmul` on the MPS device.

In this lab, "peak" means a measured ceiling: the highest median rate of
`torch.matmul` in the same run on the same host. It is not a hardware peak or
a vendor peak, and the lab quotes no vendor figure. Every timing is a
host-specific Apple GPU (Metal) measurement. Nothing here runs a CUDA kernel or
uses an NVIDIA GPU.

## Commands

```bash
python -m ai_journey.matmul_metal_cli run \
  --config config/day-74-matmul-metal.json \
  --output /tmp/day-74-matmul-metal.json
python -m ai_journey.matmul_metal_cli verify \
  --report /tmp/day-74-matmul-metal.json \
  --config config/day-74-matmul-metal.json
```

The parser uses the program name `ai-journey-matmul-metal`. This lab does not
add a console script to `pyproject.toml`, so the commands use the module form.

The `run` command compiles the kernels, checks them, times them, checks its own
report, and writes the report. It prints these lines:

1. One line for each correctness case, with the largest relative errors and
   the bound.
2. One line for each sweep point, with the median time, the IQR and the
   GFLOP/s of each side. The line also gives the ratio of each naive kernel to
   `torch.matmul`.
3. One line for the measured ceiling and the fraction of it that each naive
   kernel reached.
4. A summary line with the report digest.

In a measured run, each line before the summary has the prefix
`apple_gpu_metal_measurement`. The summary line has the prefix
`matmul_metal_measurement=`. A run with a stand-in backend or clock uses
`synthetic_timing` and `matmul_metal_synthetic_timing=` instead.

The `verify` command recomputes a report on the CPU and prints the same summary
line with the prefix `verified_`. With `--config`, `verify` also requires that
the report comes from that exact config.

The exit codes are:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. PyTorch reports an MPS device without enough memory as a
  device error.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured anything.

On the development host, one run of the pinned config took about 30 seconds,
with the torch import. It wrote a report of about 50 KB. The `verify` command
took about 2 seconds for that report.

## Requirements and hosts without MPS

The `run` command needs an Apple GPU on macOS. It also needs a PyTorch build
with an MPS device and `torch.mps.compile_shader`. The development host used
PyTorch 2.15.0. The runtime uses the probe of the Day 73 Metal lab. The probe
reads the operating system first, and it imports torch only on macOS. It names
the first requirement that is missing:

- `unsupported_operating_system`: the host does not run macOS.
- `torch_not_importable`: Python cannot import torch.
- `pytorch_without_mps`: the PyTorch build has no MPS backend.
- `mps_unavailable`: PyTorch has an MPS backend, but no MPS device is ready.
- `compile_shader_unavailable`: PyTorch has no `torch.mps.compile_shader`.

When a requirement is missing, `run` writes the reason to standard error and
exits with code 3. A Linux CI runner gets `unsupported_operating_system`
without a torch import. The `verify` command and the report reader never
import torch, so they work on any host with NumPy.

The tests that need the GPU skip on other hosts and report the reason. The
other tests use a NumPy stand-in for the GPU and one measured report from the
development host. The stand-in returns the CPU replay of the kernels, which
is what the GPU returned on that host.

## Kernels

The kernel source, in the Metal Shading Language, is in
`matmul_metal_source.py`. The lab computes `C = A @ B`, with A of shape (M, K),
B of shape (K, N) and C of shape (M, N). All three matrices are row-major
float32. Each kernel runs one thread for each element of C and gets the thread
position from `thread_position_in_grid`. That attribute is equal to the CUDA
index `blockIdx * blockDim + threadIdx`.

The two kernels differ only in the two lines that read the position:

- `matmul_x_columns` takes the column of C from x and the row from y.
- `matmul_x_rows` takes the row of C from x and the column from y.

The guard, the loop and the store come from one template in the source module.
When `row < m && column < n`, the kernel sets `sum` to 0. For each i from 0 to
k - 1, it applies `sum = fma(a[row * k + i], b[i * n + column], sum);`. Then it
stores `sum` in `c[row * n + column]`. The parameter k holds K. The Metal Shading Language defines `fma(a, b,
c)` as `a * b + c` with one rounding. Thus the source alone decides each
rounding of each output. The Day 73 probe showed that the compiler on this
host fuses `x * y + z` on its own. The explicit call removes that choice.

The parameters `m`, `n` and `k` are 32-bit `constant uint&` values. PyTorch
2.15.0 passes a Python integer as 64 bits unless `arg_casts` gives another
type. Thus the runtime gives `arg_casts` with `int32` for each of these
parameters. Each dimension is at most 8192, so each index that the kernels
calculate fits in 32 bits.

## Thread mappings and address steps

The grid's x axis changes fastest. The table gives the address step of each
matrix when x increases by 1. The steps come from the index formulas `A[row * K + k]`, `B[k * N +
column]` and `C[row * N + column]`.

| Mapping | A step | B step | C step |
| --- | --- | --- | --- |
| `x_columns` | 0 bytes: one shared element | 4 bytes | 4 bytes |
| `x_rows` | 4K bytes | 0 bytes: one shared element | 4N bytes |

In `x_columns`, consecutive threads read consecutive elements of B, and
they all read one element of A. In `x_rows`, consecutive threads read elements
of A that are K floats apart, and they all read one element of B. Thus the
strided reads of `x_rows` are reads of A, not of B. In both mappings, each
thread reads its row of A with a step of 1 element. It reads its column of B
with a step of N elements.

The pinned threadgroup is 32 x 8 threads. On the development host, both
pipelines reported a SIMD width of 32. The lab assumes that a SIMD group holds
32 consecutive threads of a threadgroup, with x first. With that assumption,
each row of 32 threads in a threadgroup is one SIMD group. The report records
the address steps of each sweep point in bytes.

## Launch geometry

The geometry follows the CUDA rule for a grid, on both axes. The group count on
each axis is `ceil(extent / group_size)`. The launch passes `groups *
group_size` threads on each axis, so the grid holds whole threadgroups. The
threads past the data are padding threads, and the guard stops them.

The x extent is N for `x_columns` and M for `x_rows`. Both mappings use the
same group size, so a time difference between them comes from the mapping. For
example, the case 17 x 33 x 9 with 32 x 8 groups gets these launches:

- `x_columns`: an extent of 33 x 17, 2 x 3 groups, 64 x 24 threads and 975
  padding threads.
- `x_rows`: an extent of 17 x 33, 1 x 5 groups, 32 x 40 threads and 719
  padding threads.

The runtime refuses a launch that was not checked against the pipeline of its
kernel and the grid of its mapping.

## Correctness

Each case gets its operands from `numpy.random.default_rng([seed, 0, case])`.
The generator draws A and then B from a standard normal distribution in
float32. The reference adds the K products of each output in float64, in
increasing order of k. A product of two float32 values is exact in float64.
Thus this order gives the same reference on each IEEE host, with any BLAS
library.

The relative error of an output element is `|C - R| / (|A| @ |B|)`. R is the
reference, and `|A| @ |B|` is the float64 sum of the absolute products. A case
records the largest relative error of each output. The bound is a worst case
that holds for each summation order, with or without fused multiply-add
operations. The bound is:

`(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)`, with
`gamma_K(u) = K u / (1 - K u)`.

The first term covers the float32 output, and the second term covers the
float64 reference. The bound increases linearly with K. Its relative value is
about 5.96e-8 at K = 1, 5.96e-5 at K = 1000 and 2.44e-4 at K = 4096. The source
of the bound is N. J. Higham, section 3.1.

A CPU replay gives the exact bits that each naive output must have. The replay
applies one `fma` step for each k, as the kernels do:

1. It calculates the product in float64, where the product is exact.
2. It adds the float64 partial sum and gets the exact rounding error of that
   addition from the TwoSum algorithm.
3. It rounds the sum to float32. When the float64 sum is exactly between two
   float32 values, the sign of the rounding error decides the direction.

Without step 3, a second rounding can go in the wrong direction. The tests hold
two operand pairs that cause this condition, one in each direction. On the
development host, the GPU gave the result of one rounding for both pairs.

Four implementations run each case:

- `matmul_x_columns` on the MPS device
- `matmul_x_rows` on the MPS device
- `torch.matmul` on the MPS device
- `torch.matmul` on the CPU

Each output must obey these rules, and a case that breaks a rule stops the
run:

- Each naive output must match the CPU replay bit for bit. Thus the two
  mappings give the same bits.
- Each naive output buffer is longer than C by its padding threads. The tail
  holds sentinel values before the launch, and it must keep them after the
  launch.
- The largest relative error of each output must not be more than the bound.

The record of each output also counts the elements whose bits differ from the
naive kernels. The pinned config has 9 cases. These cases have padding
threads on both axes of both mappings:

- 1 x 1 x 1
- 17 x 33 x 9
- 31 x 47 x 129
- 129 x 65 x 1000
- 255 x 257 x 511

The shape 32 x 32 x 16 is an exact multiple of the group size, for contrast.
The config parser refuses a config without padding on each axis of each
mapping. It also refuses a config without a non-square shape.

## Correctness results

These results come from one run of the pinned config on the development host
on 2026-11-17:

- Each naive output matched the CPU replay bit for bit, in each of the 9
  cases. Each guard tail kept its sentinel values.
- The largest relative error of a naive output was between 3.07e-8, at
  1 x 1 x 1, and 3.32e-7, at 1000 x 999 x 100.
- The largest error was 0.516 of the bound at K = 1. It was 0.00094 of the
  bound at K = 4096. The errors increased much more slowly than the bound.
- `torch.matmul` on the CPU gave the same bits as the naive kernels in each
  of the 9 cases.
- `torch.matmul` on the MPS device gave the same bits in 7 cases. It gave
  other bits in the cases 64 x 48 x 4096 and 3 x 1000 x 2048. These two cases
  have a small output and a long K. There, its largest errors were 2.78e-8 and
  1.83e-8, and the naive errors were 2.30e-7 and 2.36e-7.

## Benchmark

The benchmark times the two naive kernels and `torch.matmul` on the same MPS
device, with the same operands. The pinned sweep has 6 square shapes, from 128
to 4096, and 3 shapes that are not square. For each sweep point, the steps
are:

1. Each side runs `warmup` samples that are not timed.
2. Each timed sample calls `torch.mps.synchronize()` and reads the clock.
3. The sample then launches a batch of calls, calls `torch.mps.synchronize()`
   again, and reads the clock again.
4. The three sides take turns, and the order moves by one side at each repeat.
5. After the last sample, the two naive outputs must have the same bits. The
   `torch.matmul` output must agree with them within `4 * gamma_K(2**-24) *
   (|A| @ |B|)`, as the device calculates it in float32.

A launch returns when its command is in the queue. The two synchronize calls
make each sample include the completion of its own work. This timing rule is
the rule of the Day 73 Metal lab, with three sides instead of two. The lab
uses the clock reader, the time budget and the timing records of that lab.

A batch holds the fewest whole calls whose FLOPs reach
`target_flops_per_sample`, with at most 4096 calls. One call does `2 * M * N *
K` FLOPs, one multiply and one add for each term of each output. The GFLOP/s of
a side is the FLOPs of one sample divided by the median sample time in
nanoseconds. The IQR of the sample times gives a range of rates.

The report gives the ratio of each naive kernel to `torch.matmul` at each
point. It also gives the measured ceiling, the highest median rate of
`torch.matmul` over the sweep. For each naive kernel, it gives the highest
median rate and the fraction of the ceiling.

Only `time.perf_counter_ns` with the real `MatmulMetalRuntime` makes a
measurement. Such a report has the label `host-specific Apple GPU (Metal)
measurement` and the scope `apple_gpu_metal_this_host`. In Python, `main` and
`measure_report` accept a stand-in backend and a clock for tests. A report from
either one has `measured: false` and the scope `synthetic_backend_or_clock`.
Its boundary denies that any value in it is a measurement.

## One run on the development host

This run used the pinned config on 2026-11-17. The host was a shared Darwin
arm64 host with an Apple GPU, macOS 27.2, CPython 3.11.14, NumPy 1.26.4 and
PyTorch 2.15.0. The PyTorch float32 matmul precision was `highest`. The run
held a GPU lock that the other jobs on this host take before a timed GPU run.
Short GPU tests of other jobs can run without the lock. Other jobs also used
the CPU.

The load averages were 7.98, 9.06 and 8.98 before the run. They were 8.24,
8.97 and 8.95 after the run. These values are host-specific Apple GPU (Metal)
measurements from one run, not portable figures. To get the figures for your
host, run the command on it.

Each cell gives the median GFLOP/s of 11 samples. The range in parentheses
comes from the IQR of the sample times. The ratio columns divide the median
rate of a naive kernel by the median rate of `torch.matmul`.

| M x N x K | `x_columns` | `x_rows` | `torch.matmul` | `x_columns` / torch | `x_rows` / torch |
| --- | --- | --- | --- | --- | --- |
| 128 x 128 x 128 | 213 (209 to 214) | 134 (134 to 135) | 150 (147 to 161) | 1.416 | 0.891 |
| 256 x 256 x 256 | 302 (297 to 305) | 122 (121 to 123) | 766 (585 to 796) | 0.394 | 0.160 |
| 512 x 512 x 512 | 363 (352 to 366) | 165 (164 to 167) | 1503 (1362 to 1597) | 0.241 | 0.110 |
| 1024 x 1024 x 1024 | 381 (378 to 383) | 153 (152 to 153) | 2398 (2372 to 2499) | 0.159 | 0.064 |
| 2048 x 2048 x 2048 | 406 (406 to 407) | 161 (160 to 161) | 3692 (3563 to 3744) | 0.110 | 0.044 |
| 4096 x 4096 x 4096 | 385 (381 to 388) | 188 (187 to 188) | 3963 (3891 to 3987) | 0.097 | 0.047 |
| 4096 x 1024 x 1024 | 406 (401 to 407) | 153 (151 to 154) | 3494 (3313 to 3539) | 0.116 | 0.044 |
| 1024 x 4096 x 2048 | 401 (395 to 403) | 159 (157 to 159) | 3655 (3518 to 3753) | 0.110 | 0.043 |
| 2000 x 3000 x 1000 | 401 (395 to 405) | 120 (120 to 122) | 3556 (3514 to 3652) | 0.113 | 0.034 |

These are the results of the run:

- The measured ceiling was 3963 GFLOP/s, from `torch.matmul` at 4096 x 4096 x
  4096.
- The highest naive rate was 406 GFLOP/s, from `matmul_x_columns`. That rate
  is 0.103 of the measured ceiling, about one tenth.
- The highest rate of `matmul_x_rows` was 188 GFLOP/s, 0.047 of the measured
  ceiling.
- At each point, `matmul_x_rows` had 0.30 to 0.63 of the rate of
  `matmul_x_columns`. Both mappings gave the same bits, so the mapping changed
  the speed and not the math.
- From 256 x 256 x 256 up, `matmul_x_columns` had 0.097 to 0.394 of the rate
  of `torch.matmul`. For the square shapes, the ratio decreased as the size
  increased.
- At 128 x 128 x 128, `matmul_x_columns` was faster than `torch.matmul`, with a
  ratio of 1.416. The overhead of each call probably sets the time of
  `torch.matmul` at that shape.
- At each sweep point, the timed `torch.matmul` output had the same value as
  the naive output at each element. At 4096 x 4096 x 4096, that is 16,777,216
  elements. This result suggests that `torch.matmul` adds the products in the
  same order at these shapes.

## Reports and verification

A report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature, and it cannot prove that a timing
occurred or that a GPU ran.

The writer checks a report before it writes it. It writes a temporary file,
flushes the file to the disk, and moves it into place with `os.replace`. It
refuses an output that aliases the config through a path, a symbolic link or a
hard link. The readers accept only regular files that are not symbolic links.
The limit is 64 KiB for a config and 1 MiB for a report. The readers refuse
duplicate JSON keys, `NaN`, `Infinity`, unknown fields and missing fields. They
also refuse booleans or floats in place of integers.

The verifier uses only NumPy and its own code. It does not call the modules
that made the report. It shares the config parser, which only checks bounds.
It recomputes the correctness facts on the CPU from the seed and the stored
digests:

1. It draws the operands again from the seed and checks their digests.
2. It calculates the float64 reference again and checks its digest.
3. It replays the fma chain with its own method. Each step rounds the float64
   sum to odd before the rounding to float32, which gives the exact result.
4. It adds the sentinel tail to the replay and requires each naive buffer to
   match its stored digest.
5. It calculates the largest relative error and the bound again and holds each
   error to the bound.

The replay of the run repairs the midpoints, and the replay of the verifier
rounds to odd. Thus the two methods check each other. A `torch.matmul` output
has an unknown summation order, so the verifier cannot rebuild it. For these
outputs, the verifier checks the digest format and holds the recorded error to
the bound. It also requires a count of 0 differing elements exactly when the
digest is the digest of the replay. When the digest is the digest of the
replay, the output has the bits of the replay. Then the verifier also requires
the error of the replay.

The verifier cannot observe a timing again, so it checks the timings for
structure only. It checks the sample counts, the positive integer values, and
the statistics, rates, ratios, batch sizes, ceiling and totals that come from
the samples. It also checks the launch geometry, the address steps, the
planned work, the shader source and its digest, the environment and the
labels. Each section must match the rebuilt section as canonical JSON, so `1`,
`1.0` and `true` stay different.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `seed` | 74 | 0 to 2^32 - 1 |
| `launch.group_size` | [32, 8] | [x, y], each 2 or more, with at most 1024 threads in total |
| `correctness.shapes` | 9 shapes, 1 x 1 x 1 to 1000 x 999 x 100 | 1 to 16 different [m, n, k] triples |
| `benchmark.shapes` | 9 shapes, 128 x 128 x 128 to 4096 x 4096 x 4096 | 1 to 16 different [m, n, k] triples |
| `benchmark.warmup` | 2 | 0 to 10 samples that are not timed |
| `benchmark.repeats` | 11 | 3 to 50 timed samples for each side |
| `benchmark.target_flops_per_sample` | 1073741824 | 2^20 to 2^40, with at most 4096 calls in a sample |
| `benchmark.max_runtime_seconds` | 300 | 1 to 600 |

Each dimension can be 1 to 8192. The correctness shapes need an M and an N that
are not multiples of either group size. Then both mappings have padding threads
on both axes. A group size of 1 divides every extent, so the parser refuses
it. A correctness output with its guard tail can have at most 33555455
elements.

The config parser counts the planned work before a run, with the batch sizes
of the run. For the pinned config, the counts are 9 correctness cases,
160709684 replay steps, 9 sweep points, 297 timed samples, 11622 kernel calls
and 7712536086528 FLOPs. A run can have at most 2^28 replay steps, 2^22 calls
and 2^45 FLOPs. The time budget covers the cases and the sweep, but not the
torch import or the kernel compile.

## Evidence boundary and exclusions

The boundary of a measured report states: "These are wall-clock timings on the
Apple GPU of the host that made this report. The naive kernels came from
torch.mps.compile_shader, and torch.matmul ran on the MPS device. The timings
are host-specific Apple GPU (Metal) measurements. No CUDA kernel ran, and no
NVIDIA GPU was used. The measured ceiling is the best median rate of
torch.matmul in this run, not a hardware or vendor peak. These timings do not
measure or predict CUDA, NVIDIA, distributed or production performance."

The lab reads no hardware counters. It does not record the GPU model, the clock
frequency, the thermal state or the other jobs on the host. The rates count
the FLOPs of the matmul, not the instructions of the GPU. The report is
evidence from one Apple GPU on one host. It is not evidence of CUDA or NVIDIA
performance, lecture viewing, completed exercises or learner understanding.

## Sources

- PyTorch 2.15.0, the docstring and source of `torch.mps.compile_shader` in
  `torch/mps/__init__.py`.
- Apple, Metal Shading Language Specification, the `thread_position_in_grid`
  attribute and the `fma` function:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
  2002, section 3.1, for the bound `gamma_K`.
- D. E. Knuth, The Art of Computer Programming, Volume 2, 3rd ed., section
  4.2.2, for the TwoSum algorithm.
- S. Boldo and G. Melquiond, "Emulation of FMA and correctly rounded sums:
  proved algorithms using rounding to odd", IEEE Transactions on Computers
  57(4), 2008, for the rounding to odd.
