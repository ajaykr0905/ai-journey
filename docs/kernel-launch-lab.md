# Kernel launch model lab

This lab is the launch-model half of the Day 73 lab. The day pairs GPU MODE
Lecture 3, `Getting Started With CUDA for Python Programmers`, with an
exercise: vector add, then RGB-to-grayscale. The lab writes those two kernels
as per-thread Python functions. The lecture's own grayscale kernel uses a 1D
grid over planar channels. This lab's grayscale kernel uses a 2D grid over
interleaved HxWx3 pixels. A simulator runs them
thread by thread on the CPU and records each out-of-bounds access and each
data race. No GPU runs, and the lab compiles no CUDA source and measures no
time. The lab imports only the Python standard library, so it runs without
torch or NumPy.

## Commands

Run the pinned plan, then check the report:

```bash
PYTHONPATH=src python -m ai_journey.kernel_launch_cli run \
  --config config/day-73-kernel-launch.json \
  --output /tmp/day-73-kernel-launch-report.json

PYTHONPATH=src python -m ai_journey.kernel_launch_cli verify \
  --report /tmp/day-73-kernel-launch-report.json \
  --config config/day-73-kernel-launch.json
```

An installed wheel gives the same commands as `ai-journey-kernel-launch`.
`run` prints one line:

```text
kernel_launch_report=95b6791cc71fddc96e0e8e45bba60f1f95951921a8b382fbcae383e447ed9d9a runs=8 runs_matching_reference=8 order_independent_kernels=2 out_of_bounds_writes_caught=128 write_conflicts_caught=232 cuda_checks=10/10 simulated_on_cpu=true gpu_executed=false cuda_compiled=false
```

`verify` prints the same line with the prefix `verified_kernel_launch_report=`.
The exit codes are:

- `0`: the command succeeded.
- `1`: the command read or built a report, but the report failed verification.
- `2`: the arguments are wrong, the plan is invalid, the report file cannot be
  read, or the command refused the output path.

`--config` is optional for `verify`. With it, `verify` also requires that the
report embeds that exact plan. `run` refuses an output path that names the
plan, also through a symlink or a hard link.

## Launch model

`kernel_launch_config` holds the launch model. `Dim3` is a CUDA `dim3` extent.
Its components are positive integers, and unset axes are 1. `Index3` is a CUDA
`uint3` index such as `blockIdx` or `threadIdx`. `LaunchConfig` holds a grid
and a block. It accepts only values inside these named limits of compute
capability 8.0:

| Limit | Value |
| --- | --- |
| `warp_size` | 32 |
| `max_threads_per_block` | 1024 |
| `max_block_dim_x`, `max_block_dim_y`, `max_block_dim_z` | 1024, 1024, 64 |
| `max_grid_dim_x`, `max_grid_dim_y`, `max_grid_dim_z` | 2^31 - 1, 65535, 65535 |

The lab copies these values from Table 21, "Technical Specifications per
Compute Capability", of the NVIDIA CUDA C++ Programming Guide v12.6 (archive
12.6.3). A test compares them with the compute capability 8.0 record of the
SIMT lab. A rejected launch raises `LaunchLimitError`. Its message and its
`limit` attribute name the broken limit, for example
`block.z = 65 breaks max_block_dim_z = 64 for compute capability 8.0`.

`ceil_div(n, b)` is the `cdiv` of the lecture: `(n + b - 1) // b`.
`launch_1d(n, threads)` and `launch_2d(width, height, block_x, block_y)` use it
to size the smallest grid that covers an extent. When an extent is not a
multiple of the block, the last block on that axis reaches past the extent.
The kernel then needs a bounds guard.

## Simulator

`launch(kernel, config, *args, order=...)` calls `kernel(t, *args)` once for
each `(blockIdx, threadIdx)` pair. The context `t` holds the CUDA built-ins
`blockIdx`, `threadIdx`, `blockDim` and `gridDim` as `block_idx`,
`thread_idx`, `block_dim` and `grid_dim`. The arguments are device buffers or
`int` and `float` scalars. A kernel must return nothing, like a `__global__`
function.

Threads run one at a time in one of two orders:

- The canonical order runs the blocks by linear block ID, and the threads of
  each block by linear thread ID. Both IDs count x fastest, then y, then z.
- A permuted order shuffles all threads of the grid with Fisher-Yates. SHA-256
  counter blocks drive the shuffle instead of `random`, so one seed gives one
  order on every platform.

CUDA does not order the threads of one launch. The canonical order is only
the declared default of the simulator.

`DeviceBuffer` is a flat `uint8` or `float32` array. A kernel indexes it like
a pointer, and it has no length. During a launch, the buffers record these
events:

- An out-of-bounds read or write, with the buffer, the index, the block and
  the thread. The read returns zero, and the simulator drops the write.
- A write conflict: two threads write one element.
- A read-write conflict: one thread writes an element, and a different thread
  reads it.

Both conflicts are data races. The buffers record them also when the two
threads store the same value. In a launch without conflicts, each element has
at most one writer. If a thread writes an element, no other thread reads it.
A kernel whose result depends only on its context and the values it reads
then gives the same result in every order. The tests also show the other
case: a kernel where every thread writes `out[0]` gives different results in
different orders.

## Kernels

`kernel_launch_kernels` writes the two exercise kernels in the per-thread
style:

```python
def vector_add_kernel(t, a, b, out, n):
    i = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    if i < n:
        out[i] = to_float32(a[i] + b[i])


def rgb_to_grayscale_kernel(t, rgb, gray, width, height):
    col = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    row = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    if col < width and row < height:
        pixel = row * width + col
        gray[pixel] = luma_u8(rgb[3 * pixel], rgb[3 * pixel + 1], rgb[3 * pixel + 2])
```

The vector add stores each sum through `to_float32`. This function rounds to
the nearest float32 with ties to even, as a CUDA `float` store does. For the
seeded inputs the double sum is exact, so the stored value equals the result
of a CUDA `float` add. The grayscale kernel reads an `H x W x 3` `uint8`
image: interleaved RGB in row-major order. Its 2D blocks run x over the
columns and y over the rows, and its guard tests both axes.

The grayscale weights are 0.2989, 0.5870 and 0.1140. GPU MODE Lecture 3 uses
these weights in its rgb-to-grayscale code. Each weight differs from the
Rec. ITU-R BT.601 luma weight (0.299, 0.587 or 0.114) by at most 0.0001. The
lab reads the weights as exact decimals and applies this rule:

1. Compute the integer `2989 R + 5870 G + 1140 B`, which is the exact weighted
   sum times 10,000.
2. Round half up: add 5,000, then divide by 10,000 with floor division.
3. Clamp the result to [0, 255].

The sum of the weights is 0.9999. Thus the largest weighted sum is 254.9745,
and the clamp never changes the result for `uint8` input.

Binary floating point cannot hold the decimal weights exactly. Of the
16,777,216 RGB triples, 1,703 have an exact sum that ends in .5. For 292 of
these ties, the double formula `floor(0.2989 * r + 0.5870 * g + 0.1140 * b +
0.5)` rounds down. The integer rule has no such case. C converts an in-range
float to `unsigned char` by truncation toward zero. Thus a kernel that stores
the float sum directly can give a value one lower.

The unguarded variants `vector_add_unguarded_kernel` and
`rgb_to_grayscale_unguarded_kernel` drop the guard. Only the tests and the
report use them.

## Inputs, references and comparisons

`kernel_launch_reference` makes the inputs from SHA-256 counter blocks. Block
k of a stream is `SHA-256("ai-journey-kernel-launch-input:<stream>:<seed>:<k>")`.
The image bytes are the stream bytes. Each float32 value uses 4 stream bytes:
a sign bit, a 4-bit exponent offset and a 23-bit mantissa. Thus the
magnitudes are in [2^-8, 2^8), and many float32 sums round.

The references compute each output element with plain loops and without the
simulator:

- The vector add forms each sum in double and rounds it to float32 with
  `array('f')`. For these inputs the exponents differ by at most 15, so the
  double sum is exact.
- The grayscale reference applies the rule with `Fraction` arithmetic.

`compare_exact` compares `float32` values by bit pattern and `uint8` values by
value. An output digest is the SHA-256 of the little-endian buffer bytes.

## CUDA source

`kernel_launch_cuda.cuda_source` writes CUDA C++ text for both kernels and
their host launchers. The text uses the launch math of the simulator. It
gives one thread to each element and sizes each grid axis as
`cdiv(extent, block)`. It has the same guards and the integer grayscale rule.

The text is source for a later run on an NVIDIA GPU. This lab does not compile
or run it. The development host has no NVIDIA GPU, no CUDA toolkit and no
`nvcc`.

`check_cuda_source` reads the text without a compiler. It runs 10 named
checks:

- `cdiv_definition`: `cdiv` returns `(a + b - 1) / b`.
- `vector_add_signature` and `rgb_to_grayscale_signature`: each `__global__`
  signature occurs once, with the expected parameter types.
- `vector_add_index`: `int i = blockIdx.x * blockDim.x + threadIdx.x;`.
- `rgb_to_grayscale_index`: the `col` expression on x and the `row`
  expression on y.
- `vector_add_guard` and `rgb_to_grayscale_guard`: each memory access is
  inside `if (i < n)` or inside `if (col < width && row < height)`.
- `rgb_to_grayscale_rule`: the integer weights, the 5,000 offset, the 10,000
  divisor and the clamp to 255.
- `vector_add_grid` and `rgb_to_grayscale_grid`: each launcher divides its
  extents by its block with `cdiv`. The check evaluates the `dim3`
  expressions for the plan extents and requires the launch configuration of
  the simulator.

Each function must hold exactly its pinned statements, in the pinned order.
Only whitespace and comments can differ. Thus one extra statement fails a
check of its function. For example, `i += 1;` after the index statement fails
`vector_add_index`.

Each `dim3` literal must be a decimal literal without a leading zero, because
C reads `0256` as the octal value 174. The text must hold the five
definitions and no other code. Other code, such as an `int` overload of
`cdiv`, can change the five functions. Thus such text fails every check.

The check finds comments from left to right, as a compiler does. Thus `/*` in
a `//` comment does not open a block comment. A `//` comment stops at a line
feed or at a carriage return. A compiler joins a line that ends with a
backslash to the next line, so a backslash can move a guard into a comment.
Text with a backslash, a `??` trigraph or an unterminated `/*` comment fails
every check.

The pinned text has the SHA-256
`cf14adedc20211d11c978cf0d6af291a447853ec01e4d0dee1ecfd02c8631bcc`. The report
holds the full text.

## Report and verification

`run` writes canonical JSON (sorted keys, compact separators and ASCII
escapes) with a SHA-256 seal over all other fields. The seal detects drift.
It is not a signature. The report has no floating-point fields, so its bytes
do not depend on how a platform prints floats. The report holds these
sections:

- `plan`, `limits` and `grayscale_rule`: the declared inputs, the named limits
  and the stated rule.
- `vector_add` and `rgb_to_grayscale`: the launch configuration, the launched,
  useful and idle thread counts, the input and reference digests, and one row
  for each order.
- `unguarded_demos`: the misuse counts, the digests of the out-of-bounds
  records and conflicts, and the first four out-of-bounds writes.
- `cuda_source`: the text, its digest and the results of the structural
  checks.
- `summary` and `evidence_boundary`.

Each order row gives the order digest, the output digest, the useful threads,
and the counts of accesses, out-of-bounds accesses and conflicts.

`verify` trusts no number in the report. It does these steps:

1. Check the field set, the seal and the fixed evidence boundary.
2. Parse the embedded plan. With `--config`, require that plan.
3. Recompute the kernel, demo and summary sections with its own code, and
   name the first field that differs. The code makes the input streams, the
   float32 sums, the integer rule and the order digests again. It also
   enumerates the threads of each unguarded demo directly. It reads the CUDA
   text with `check_cuda_source`, and it compares the limits and the rule
   with the lab constants.
4. Require that the report equals a fresh build, byte for byte.

The reader accepts only regular non-symlink files: plans of at most 64 KiB and
reports of at most 1 MiB. It rejects duplicate keys, `NaN`, unknown or missing
fields, and booleans in place of integers.

## Pinned results

The pinned plan `config/day-73-kernel-launch.json` uses the input seed 73 and
the permuted order seeds 1, 2 and 3.

- Vector add: 1,000 float32 elements in blocks of 256 threads need a grid of 4
  blocks. The launch has 1,024 threads: 1,000 useful and 24 idle. In all four
  orders the output matches the reference bit for bit. Its digest is
  `879aaad6e7345f9a1f64a662323d767deb8b3322d70bc3df550b19d0e47bf870`. Float32
  rounding changed 851 of the 1,000 sums.
- Grayscale: a 40 x 30 image in 16 x 16 blocks needs a 3 x 2 grid. The launch
  has 1,536 threads: 1,200 useful and 336 idle. In all four orders the output
  matches the reference. Its digest is
  `681b798d88c39d1160f671a756c5a9df66f84f1bb1e6ea2763a70d703e735726`. No pixel
  is a tie, and truncation would give 612 pixels a value one lower.

The unguarded demos give these results in all four orders:

- The unguarded vector add over 1,000 elements makes 24 out-of-bounds writes
  and 48 out-of-bounds reads. The first write goes to index 1000, from block
  3, thread 232.
- The unguarded grayscale kernel over 40 x 30 pixels makes 104 out-of-bounds
  writes and 312 out-of-bounds reads. The first write goes to index 1200, from
  block (0, 1), thread (0, 14).
- In the same launch, each thread of rows 0 to 28 past column 39 writes a
  pixel of the next row. This gives 232 write conflicts. Both writers of such
  a pixel read the same RGB values, so the output still matches the
  reference.
- At the aligned extents, 1,024 elements and 48 x 32 pixels, the unguarded
  kernels make no out-of-bounds access and no conflict.

A matching output does not prove that a kernel is free of races. The conflict
records show the race. The tests check that the out-of-bounds writes and the
extra writes equal the closed forms of the SIMT lab. The report digest is
`95b6791cc71fddc96e0e8e45bba60f1f95951921a8b382fbcae383e447ed9d9a`.

## Boundaries

The simulator runs one thread at a time. It does not model warps, divergence,
shared memory, barriers, the CUDA memory model, caches or timing. It finds
conflicts only inside one launch. It records out-of-bounds accesses only in
simulated buffers. On a GPU, an out-of-bounds access can corrupt other memory
or stop the kernel.

The lab does not compile or run the CUDA text, so it gives no NVIDIA,
production or distributed evidence. The report is not evidence of lecture
viewing, a completed exercise or learner understanding.
