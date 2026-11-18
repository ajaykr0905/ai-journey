# Metal kernels on the Apple GPU

Day 73 of the curriculum asks for a benchmark against the PyTorch equivalent.
This lab is the GPU half of that exercise, for a Mac with an Apple GPU. It
compiles two Metal kernels at run time through `torch.mps.compile_shader`, the
Metal counterpart of CUDA's `load_inline`. It checks the kernels against
PyTorch on the MPS device and on the CPU. Then it times them against PyTorch on
the MPS device.

Every timing is a measurement on the Apple GPU of one host. Nothing here runs
a CUDA kernel or uses an NVIDIA GPU.

## Commands

```bash
python -m ai_journey.kernel_metal_cli run \
  --config config/day-73-kernel-metal.json \
  --output /tmp/day-73-kernel-metal.json
python -m ai_journey.kernel_metal_cli verify \
  --report /tmp/day-73-kernel-metal.json \
  --config config/day-73-kernel-metal.json
```

An installed wheel gives the same commands as `ai-journey-kernel-metal`.

`run` compiles the kernels, checks them, times them, checks its own report,
and writes the report. It prints one line for the correctness of each kernel
and one line for the contraction probe. It also prints one line for each sweep
point and a summary line with the report digest. In a measured run, each line
before the summary has the prefix `apple_gpu_metal_measurement`, and the
summary line has the prefix `kernel_metal_measurement=`. A run with a
stand-in backend or clock uses `synthetic_timing` and
`kernel_metal_synthetic_timing=` instead.

The `verify` command recomputes a report on the CPU and prints the same
summary line with the prefix `verified_`. With `--config`, `verify` also
requires that the report comes from that exact config.

The exit codes are:

- 0: the command succeeded.
- 1: the command read a report, but the report failed the checks.
- 2: a usage error, an unreadable or invalid input, or a refused output. Code
  2 also marks a run that failed a check, had a device error or exceeded its
  time budget. PyTorch reports an MPS device without enough memory as a
  device error.
- 3: this host cannot run Metal kernels through PyTorch, so `run` stopped
  before it measured anything.

On the development host, one run of the pinned config took 4 to 5 seconds,
with the torch import. It wrote a report of about 37 KB.

## Requirements and hosts without MPS

The `run` command needs an Apple GPU on macOS. It also needs a PyTorch build
with an MPS device and `torch.mps.compile_shader`. The development host used
PyTorch 2.15.0. Before `run` measures anything, a probe reads the operating
system. The probe imports torch only on macOS. It names the first requirement
that is missing:

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
other tests use observations from the development host. These are the
pipeline limits, the probe result and one complete measured report.

## Kernels

The kernel source, in the Metal Shading Language, is in
`kernel_metal_source.py`. Each kernel gets its global index from
`thread_position_in_grid`. That attribute is equal to the CUDA index
`blockIdx * blockDim + threadIdx`.

- `vector_add` runs one thread for each element. When `i < n`, it computes
  `out[i] = a[i] + b[i]` in float32.
- `rgb_to_gray` runs one thread for each pixel, with the column on the x axis
  and the row on the y axis. It reads a row-major uint8 image of shape
  (height, width, 3). When `column < width && row < height`, it writes one
  uint8 level for the pixel.

The parameters `n`, `width` and `height` are 32-bit `constant uint&` values.
PyTorch 2.15.0 passes a Python integer as 64 bits unless `arg_casts` gives
another type. Thus the runtime gives `arg_casts` with `int32` for each of these
parameters.

Grayscale uses the weights 0.2989, 0.5870 and 0.1140 as exact decimals. It
applies the rule of the CUDA half of this lab, with the same integer
statements as the CUDA source of that half:

1. Compute the integer `2989 R + 5870 G + 1140 B`, which is the exact weighted
   sum times 10,000.
2. Round half up: add 5,000, then divide by 10,000 with floor division.
3. Clamp the result to 0..255.

The sum of the weights is 0.9999, so the weighted sum is at most 254.9745. The
clamp thus never changes a value, but the kernel keeps it because the rule
includes it. The largest integer in these steps is 2,554,745, so a 32-bit
`uint` holds each step exactly.

Binary floating point cannot hold the decimal weights exactly. Of the 2^24 RGB
triples, 1,703 have an exact weighted sum that ends in .5. The float32 form
`floor(0.2989f * r + 0.5870f * g + 0.1140f * b + 0.5f)` rounds 19 of these
ties down. A compiler that fuses a multiply and an add can round a tie in the
other direction. The integer rule has no such case, so the kernel, the
PyTorch equivalent and the NumPy reference all use it.

The source also has a third kernel, `contraction_probe`. It computes
`x[0] * x[1] + x[2]` for three exact float32 inputs. Two separate float32
roundings give 0, and one fused multiply-add gives 2^-24. On the development
host, the probe gave 2^-24, so the compiler fused the multiply and the add. A
fused multiply-add cannot change an integer result, so the probe result does
not affect `rgb_to_gray`.

## Launch geometry

The geometry follows the CUDA rule for a grid. The group count on each axis is
`ceil(extent / group_size)`, and `group_size` is the CUDA block size. The
launch passes `groups * group_size` threads on each axis, so the grid holds
whole threadgroups. The threads past the data are padding threads, and the
bounds guard stops them.

The API shows the limits only through each compiled pipeline. These limits are
`max_threads_per_threadgroup` and `thread_execution_width`. On the development
host, both pipelines gave 1024 and 32. The runtime refuses a launch that has
more threads in a threadgroup than its pipeline allows. The report records the
padding threads and the SIMD groups of each threadgroup.

For example, a vector of 1000 elements with a group size of 256 gets 4 groups,
1024 threads and 24 padding threads. An image of 17 rows and 33 columns uses
16 x 16 groups. It gets 3 x 2 groups, 48 x 32 threads and 975 padding threads.

## Correctness

Each case gets its inputs from `numpy.random.default_rng([seed, kernel, case])`.
Vector add uses standard-normal float32 vectors, and grayscale uses uniform
uint8 pixels. One more grayscale case holds each of the 2^24 RGB triples once,
in a 4096 x 4096 image.

Three implementations run each case:

- the Metal kernel on the MPS device
- the PyTorch equivalent on the MPS device
- the PyTorch equivalent on the CPU

The PyTorch equivalent of `vector_add` is `torch.add`. The PyTorch equivalent
of `rgb_to_gray` is ten eager operations. These operations do the integer
steps of the kernel in int32, and the kernel does them in 32-bit `uint`. A
NumPy reference on the CPU computes the expected outputs: float32 sums for
vector add, and the integer rule for grayscale. Each output must obey these
rules:

- Each vector-add output must match the reference bit for bit.
- Each grayscale output must match the reference bit for bit. The report
  keeps a list of the mismatched pixels of each output, and in a valid report
  each list is empty.
- The Metal output buffer is longer than the data by the padding threads. Its
  tail holds sentinel values before the launch. After the launch, the tail
  must still hold the sentinels, which shows that the guard stopped each
  padding thread.

A case that breaks a rule stops the run, so a report never holds a failed
check. The pinned lengths 1, 255, 257, 1000, 65537 and 1048579 are not
multiples of the group size. The pinned shapes 1 x 1, 15 x 17 and 1023 x 1025
are not multiples on either axis. The shape 480 x 641 is not a multiple on the
x axis. The config parser refuses a config without such extents on each axis.
The length 256 and the shape 16 x 16 are exact multiples, for contrast.

In three runs on the development host, each of the 13 cases was exact on all
three implementations. These cases include the all-triples image, so the Metal
kernel matched the rule of the CUDA half on each possible RGB input. The GPU
tests also run the all-triples image with 16 x 16, 32 x 8 and 8 x 32
threadgroups. Each of these threadgroup shapes gave the same exact output.

## Benchmark

The benchmark times the Metal kernel against the PyTorch equivalent on the
same MPS device. The pinned sweep has 7 vector lengths and 6 image shapes. For
each sweep point, the steps are:

1. Each side runs `warmup` samples that are not timed.
2. Each timed sample calls `torch.mps.synchronize()` and reads the clock.
3. The sample then launches a batch of calls, calls `torch.mps.synchronize()`
   again, and reads the clock again.
4. The two sides alternate, and the order flips at each repeat.
5. After the last sample, the latest outputs of the two sides must be equal.

A launch returns when its command is in the queue. The two synchronize calls
make each sample cover the completion of its own work. A batch holds the
fewest whole calls whose compulsory bytes reach `target_bytes_per_sample`,
with at most 4096 calls. Thus a small point takes enough time to measure
against the cost of the synchronize calls.

The report keeps the raw samples. From them, it gives the nearest-rank median
and IQR of each side, the median time of each call, and the effective
bandwidth. The effective bandwidth is the compulsory bytes of a sample divided
by the median sample time, in decimal GB/s.

Vector add reads 8 bytes and writes 4 bytes for each element. Grayscale reads
3 bytes and writes 1 byte for each pixel. The PyTorch equivalent of grayscale
moves more bytes than that through its int32 temporary tensors. Both sides
use the same byte count, so their rates compare their times directly.

Only `time.perf_counter_ns` with the real `MetalRuntime` makes a measurement.
Such a report has the label `host-specific Apple GPU (Metal) measurement` and
the scope `apple_gpu_metal_this_host`. In Python, `main` and `measure_report`
accept a stand-in backend and a clock for tests. A report from either one has
`measured: false` and the scope `synthetic_backend_or_clock`. Its boundary
denies that any value in it is a measurement.

## Reports and verification

A report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature, and it cannot prove that a timing
occurred or that a GPU ran.

The writer checks a report before it writes it. It writes a temporary file,
flushes the file to the disk, and moves it into place with `os.replace`. It
refuses an output that aliases the config through a path, a symbolic link or a
hard link.

The readers accept only regular files that are not symbolic links. The limit
is 64 KiB for a config and 1 MiB for a report. The readers refuse duplicate
JSON keys, `NaN`, `Infinity`, unknown fields and missing fields. They also
refuse booleans or floats in place of integers.

The verifier uses only NumPy and its own code. It does not call the modules
that made the report. It shares the config parser, which only checks bounds.
It recomputes the correctness facts on the CPU from the seed and the stored
digests:

1. It draws the inputs again from the seed and checks their digests.
2. It computes the reference again and checks its digest.
3. It rebuilds each output from the reference and the recorded mismatches. It
   adds the sentinel tail to the Metal buffer.
4. It requires each rebuilt buffer to match its stored digest. A match shows
   that the list of mismatches is complete.
5. It recomputes the mismatch counts and the largest differences, and it holds
   them to the bound.

The verifier cannot observe a timing again, so it checks the timings for
structure only. It checks the sample counts, the positive integer values, and
the statistics, rates, batch sizes and totals that come from the samples. It
also checks the launch geometry, the planned work, the shader source and its
digest, the probe result, the environment and the labels. Each section must
match the rebuilt section as canonical JSON, so `1`, `1.0` and `true` stay
different.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `seed` | 73 | 0 to 2^32 - 1 |
| `launch.vector_add_group_size` | 256 | 2 to 1024 |
| `launch.rgb_to_gray_group_size` | [16, 16] | [x, y], each 2 or more, with at most 1024 threads in total |
| `correctness.vector_add_lengths` | 7 lengths, 1 to 1048579 | 1 to 16 increasing lengths of 1 to 2^25 elements |
| `correctness.rgb_to_gray_shapes` | 5 shapes, 1 x 1 to 1023 x 1025 | 1 to 16 different [height, width] pairs |
| `correctness.rgb_to_gray_all_triples` | true | true or false |
| `benchmark.vector_add_lengths` | 7 lengths, 4096 to 16777216 | 1 to 16 increasing lengths |
| `benchmark.rgb_to_gray_shapes` | 6 shapes, 64 x 64 to 4096 x 4096 | 1 to 16 shapes, each with more pixels than the one before |
| `benchmark.warmup` | 2 | 0 to 10 samples that are not timed |
| `benchmark.repeats` | 11 | 3 to 50 timed samples for each side |
| `benchmark.target_bytes_per_sample` | 33554432 | 1 KiB to 1 GiB, with at most 4096 calls in a sample |
| `benchmark.max_runtime_seconds` | 120 | 1 to 600 |

An image can have at most 8192 pixels on a side and 2^24 pixels in total. The
correctness cases need one length that is not a multiple of the vector group
size. They also need one width and one height that are not multiples of the
image group size. A group size of 1 divides every extent, so the parser
refuses it.

The config parser counts the planned work before a run, with the batch sizes
of the run. For the pinned config, the counts are 13 correctness cases, 13
sweep points, 286 timed samples, 81354 kernel calls and 17363214336
compulsory bytes. A run can have at most 2^22 calls and 2^38 bytes. The time
budget covers the probe, the cases and the sweep, but not the torch import or
the kernel compile.

## Three runs on the development host

These runs used the pinned config on 2026-11-16. The host was a shared Darwin
arm64 host with an Apple GPU, macOS 27.2, CPython 3.11.14, NumPy 1.26.4 and
PyTorch 2.15.0. Other jobs ran on the host at the same time. The values are
medians from three runs on one host, not portable figures. To get figures for
your host, run the command on it.

- Vector add: the ratio of the PyTorch median to the Metal median stayed
  between 0.85 and 1.24 at each length. A ratio above 1 means that the Metal
  kernel was faster. The ratio was 1 or more at 18 of the 21 points. Neither
  side was faster at every point.
- At 16777216 elements, 192 MiB for each call, the Metal kernel gave 92 to 116
  GB/s and `torch.add` gave 99 to 109 GB/s.
- The highest Metal rate for vector add was 193 GB/s, at 262144 elements. The
  same point gave 161 to 193 GB/s in the three runs. The 3 MiB of one call
  probably stay in the GPU caches from call to call. Thus this rate does not
  measure the memory bandwidth of the host.
- At 4096 elements, a vector-add call took 2.4 to 2.5 µs, which gives 19 to 20
  GB/s. The fixed cost of each call probably sets this time. At 100 GB/s, the
  48 KiB of one call would move in about 0.5 µs.
- Grayscale: the Metal kernel was 12.8 to 23.6 times faster than the PyTorch
  equivalent at each shape in each run.
- At 4096 x 4096, the Metal kernel gave 79 to 82 GB/s and the PyTorch
  equivalent gave 3.9 to 4.1 GB/s.
- Later runs, while other test jobs used the host, gave a wider spread. The
  grayscale speedup went from 8.1 to 24.3 times, and the vector-add ratio
  reached 1.30. The values change with the load on the host.

## Evidence boundary and exclusions

The boundary of a measured report states: "These are wall-clock timings on the
Apple GPU of the host that made this report. The Metal kernels came from
torch.mps.compile_shader, and their PyTorch equivalents ran on the MPS device.
The timings are host-specific Apple GPU (Metal) measurements. No CUDA kernel
ran, and no NVIDIA GPU was used. Effective bandwidth counts the bytes that
each kernel must read and write, not the hardware traffic. These timings do
not measure or predict CUDA, NVIDIA, distributed or production performance."

The lab reads no hardware counters. It does not record the GPU model, the
clock frequency, the thermal state or the other jobs on the host. The
bandwidths count compulsory bytes, not the hardware traffic. The report is
evidence from one Apple GPU on one host. It is not evidence of CUDA or NVIDIA
performance, lecture viewing, completed exercises or learner understanding.

## Sources

- PyTorch 2.15.0, the docstring and source of `torch.mps.compile_shader` in
  `torch/mps/__init__.py`.
- Apple, Metal Shading Language Specification, the `thread_position_in_grid`
  attribute: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- ITU-R Recommendation BT.601, which gives the luma weights 0.299, 0.587 and
  0.114: https://www.itu.int/rec/R-REC-BT.601
