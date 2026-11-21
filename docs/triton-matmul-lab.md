# A Triton port of the tiled matmul

Day 79 of the curriculum follows GPU MODE Lecture 14, "Practitioner's Guide to
Triton", and Lecture 18, "Fused Kernels". The tracker notes name the topics:
Triton program IDs, block pointers, masks, autotuning and Python-level kernel
authoring. The exercise is "Port your matmul to Triton. Compare lines of code
AND performance vs your CUDA version." This lab is the Triton half of Day 79.

The lab does these steps:

1. It writes one Triton kernel text and its host launcher, with a SHA-256
   digest.
2. It runs the text in the Triton CPU interpreter on the nine Day 74 shapes,
   with six declared configs. A Linux container ran this check, because Triton
   has no macOS wheel.
3. It counts the lines of the Triton text and of the Day 76 and Day 77 CUDA and
   Metal texts with one declared rule.
4. It states the performance side without a GPU number. The only rates come
   from the recorded Day 76 and Day 77 Metal reports.
5. It seals one comparison report, and an independent verifier checks the
   report on any CPU.

Nothing here runs on a GPU. The lab compiles no Triton kernel for a GPU, tunes
no config, compiles no CUDA text and calls no cuBLAS. The interpreter time is
not a performance number, so the report holds no time.

Ajay must still watch Lecture 14 and Lecture 18. This lab does not cover the
fused-kernel topic of Lecture 18.

## Commands

```bash
PYTHONPATH=src python -m ai_journey.triton_matmul_cli loc
PYTHONPATH=src python -m ai_journey.triton_matmul_cli check \
  --config config/day-79-triton-matmul.json \
  --output /tmp/day-79-triton-matmul.json
PYTHONPATH=src python -m ai_journey.triton_matmul_cli verify \
  --report /tmp/day-79-triton-matmul.json \
  --config config/day-79-triton-matmul.json
```

The parser uses the program name `ai-journey-triton-matmul`. The package has
no console script for this lab, so the commands use `python -m`.

- `loc` prints one line for each counted text. It needs no triton.
- `check` loads the plan, runs each case in the Triton CPU interpreter and
  builds the report. It verifies the report and then writes it atomically. It
  never writes over the plan. It prints one line for each case and one summary
  line with the report digest.
- `verify` recomputes a report on the CPU. With `--config`, it also requires
  that the report comes from that plan. It never imports triton or torch.

The exit codes are the codes of the Day 73 to 77 labs:

- 0: the command succeeded.
- 1: the command read or built a report, but the report failed verification.
- 2: a usage error, an unreadable or invalid input, a refused output, or a
  check that broke a rule. Such a check writes no report.
- 3: this host cannot run the Triton CPU interpreter. Triton or torch is
  missing, or `TRITON_INTERPRET` turns the interpreter off. Then `check` writes
  nothing.

## Hosts without triton

Triton has no macOS wheel, so on the development Mac `check` exits with code 3.
It prints the reason, `triton_not_importable`, and it does not make the output
directory. The `loc` and `verify` commands work on that host.

The backend imports triton and torch only when `check` makes the backend. It
sets `TRITON_INTERPRET=1` when the variable is unset. It refuses a host where
`triton.jit` does not give an `InterpretedFunction`, for example a host with
`TRITON_INTERPRET=0`.

The tests that need triton skip on a host without it. One CLI test runs a
fresh process that hides triton, and it requires exit code 3 and no output.
The other tests use a NumPy stand-in that does the arithmetic of the
interpreter.

## The kernel

`triton_matmul_kernel.TRITON_SOURCE` holds the kernel `matmul_kernel` and the
launcher `launch_matmul`. Its SHA-256 digest is
`9eb35e273b0fd229e83e223861ef7ce13c4f15ea46d2694baf13920851a31d56`. One program
computes one BLOCK_M x BLOCK_N tile of `C = A @ B`. All three matrices are
row-major float32. This is the K loop:

```python
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        # A slot past M, past N or past the end of K loads 0.
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k0)
        b_mask = (offs_k[:, None] < K - k0) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
```

The kernel has four parts:

- Program IDs: the launch grid has one axis with
  `cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)` programs. Nine lines map each program
  ID to one tile of the 2D tile grid of C, in the grouped order below.
- Pointers and masks: each program makes a BLOCK_M x BLOCK_K block of pointers
  into A and a BLOCK_K x BLOCK_N block into B. A masked load reads only the
  slots inside the matrix, and `other=0.0` puts 0 in the other slots. The store
  mask keeps each write inside C. Thus a ragged M, N or K needs no padding.
- The K loop: `tl.dot` multiplies one A tile by one B tile and adds the product
  to a float32 accumulator. Then the loop moves both blocks of pointers by
  BLOCK_K.
- Precision: the `tl.dot` docstring of Triton 3.9.0 gives `tf32` as the float32
  default on a GPU with Tensor Cores. The kernel sets
  `input_precision="ieee"`. Thus the Day 74 float32 bound stays the contract of
  a later GPU run.

The masks do the work of the guarded loads of the Day 76 CUDA text, for
example `(row < m && a_col < k) ? a[row * k + a_col] : 0.0f`. The Triton matmul
tutorial wraps the row and column offsets with `% M` and `% N` and masks only
K. This kernel masks all three axes, as the CUDA text guards its loads.

## Program IDs and the grouped order

Consecutive program IDs cover GROUP_M rows of one tile column, and then they
continue in the next tile column. The Triton matmul tutorial uses this order
to get more reuse of data in the L2 cache. This lab measures no cache, so it
claims no cache effect.

`program_tile` replays the nine lines on the CPU. A test runs the lines of the
text with a stand-in for `tl` and compares each result with `program_tile`.
Another test requires that the order covers each tile once. It tries each grid
from 1 x 1 to 17 x 17 tiles with each GROUP_M from 1 to 9.

For example, take 10 tile rows, 3 tile columns and GROUP_M = 4. Then program
IDs 0 to 3 give the tiles (0, 0) to (3, 0). IDs 4 to 7 give the tiles (0, 1)
to (3, 1).

## Block pointers and tensor descriptors

The tracker notes name block pointers. In Triton 3.9.0, `tl.make_block_ptr`
raises `NotImplementedError` with the message "Block pointers have been removed
in favor of the tensor descriptor API". `tl.advance` raises the same error.
Thus the lab cannot use block pointers with the pinned Triton version.

The `tl.make_tensor_descriptor` docstring requires leading strides that are
multiples of 16 bytes. A row of N = 33 float32 values spans 132 bytes. In the
container, the interpreter refused that stride with the message "Tensor
descriptor strides must be 16-byte aligned". Six of the nine Day 74 shapes have
a row stride of A or B that is not a multiple of 16 bytes. Thus the kernel uses
pointer arithmetic with masks, which works for each Day 74 shape.

## Declared configs and autotune keys

`AUTOTUNE_CONFIGS` declares six configs. They come from the CUDA configs of
the Triton matmul tutorial at tag v3.9.0. The lab keeps each config with K
blocks of 32 and at most 128 rows and 128 columns in a tile. The other
tutorial configs have larger tiles, and a float32 tile has twice the bytes of
a float16 tile. Each config has GROUP_M = 8.

| Config | BLOCK_M x BLOCK_N x BLOCK_K | Warps | Stages | Tile bytes |
| --- | --- | --- | --- | --- |
| `m128_n128_k32_g8_w4_s4` | 128 x 128 x 32 | 4 | 4 | 32,768 |
| `m128_n64_k32_g8_w4_s4` | 128 x 64 x 32 | 4 | 4 | 24,576 |
| `m64_n128_k32_g8_w4_s4` | 64 x 128 x 32 | 4 | 4 | 24,576 |
| `m128_n32_k32_g8_w4_s4` | 128 x 32 x 32 | 4 | 4 | 20,480 |
| `m64_n32_k32_g8_w2_s5` | 64 x 32 x 32 | 2 | 5 | 12,288 |
| `m32_n64_k32_g8_w2_s5` | 32 x 64 x 32 | 2 | 5 | 12,288 |

The tile bytes are the float32 bytes of one A tile and one B tile. The compiler
decides how many copies of them it keeps for `num_stages`. Thus the value is
not the shared memory of a launch.

The `triton.autotune` docstring of Triton 3.9.0 states that the tuner
evaluates all configs again when the value of a key argument changes. The lab
declares the key M, N, K, as the tutorial does. A container test wraps the
interpreted kernel in `triton.autotune` with these configs and this key. The
tuner holds the six configs and an empty cache, and no test calls it. Thus the
lab records the configs and tunes nothing.

The interpreter drops `num_warps` and `num_stages` before it calls the kernel.
Thus the interpreter check tests only the block sizes of each config.

## What Triton hides compared with CUDA and Metal

The Triton introduction compares two models. A CUDA kernel is a scalar program
with blocked threads. A Triton kernel is a blocked program with scalar threads.
The lab texts show the difference:

| Concern | Day 76 and Day 77 CUDA | Day 76 and Day 77 Metal | Day 79 Triton |
| --- | --- | --- | --- |
| Work of one thread or program | one element or one micro-tile | one element or one block of outputs | one tile as a block value |
| Index of the work | `threadIdx` and `blockIdx` | thread and threadgroup positions | `tl.program_id(axis=0)` |
| Shared tiles | `__shared__` arrays | `threadgroup` arrays | none in the text |
| Barriers | two `__syncthreads()` calls in each phase | two `threadgroup_barrier` calls in each phase | none in the text |
| Bounds | a guard on each element | a guard on each element | a mask on each block |
| Wide loads | a `float4` path in the blocked text | a `float4` path from S4 | none in the text |
| Threads of one block | `dim3 block` in the launcher | the group size of the launch | `num_warps` in the config |
| Tile sizes | one text for each tile or design | one kernel for each tile or design | one text for all configs |

The Triton introduction at v3.9.0 lists work that its compiler does. The list
holds coalescing, thread swizzling, prefetching, vectorization and the choice
of Tensor Core instructions. It also holds the allocation of shared memory,
synchronization, and the scheduling of asynchronous copies. The `triton.Config`
docstring states that `num_warps=8` gives one program 8 * 32 = 256 threads. It
also states that `num_stages` sets the stages of software pipelining.

The lab did not compile the text for a GPU. Thus it cannot show which of these
steps the compiler takes for this kernel. Triton does not hide the order of
the tiles: the nine lines of the grouped order are code that the programmer
writes.

## Correctness in the CPU interpreter

The plan `config/day-79-triton-matmul.json` names the nine correctness shapes
of `config/day-74-matmul-metal.json`, seed 74 and all six configs. Each case
draws two operand families:

- normal: the standard-normal float32 operands of Day 74 case i, from
  `matmul_inputs(74, 0, i, shape)`. With seed 74, these are the operands of the
  cases of the Day 76 CUDA contract.
- exact: integers from -8 to 8 as float32, from stream 2 of the seed.

Each run has these rules, and a run that breaks one stops the check:

1. The output buffer is C and a guard tail of sentinel quiet NaNs. The tail
   covers each index that a store without its mask could reach. After the run,
   the tail must keep its sentinels.
2. Each element of C must be finite. An element that kept its sentinel shows a
   tile that no program stored.
3. The largest relative error of a normal output must not be more than the
   Day 74 bound, `(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)`.
4. An exact output must hold the bits of the float64 reference.

The order of the additions inside `tl.dot` is not fixed. The compiler or the
interpreter chooses it. In Triton 3.9.0, the interpreter computes `tl.dot` as
`numpy.matmul(a, b, dtype=float32) + acc`. No CPU replay names these bits, so a
normal output gets a bound and not a bit pattern. Higham shows that the bound
holds for every order of the additions.

The exact family has a different rule. Each partial sum is an integer of at
most 64 * 8192 = 2**19 in size. Float32 holds each integer up to 2**24
exactly. Thus every order of the additions gives the exact sum.

### The container run

The interpreter ran in a linux/aarch64 python:3.11-slim container on Docker
Desktop on the development Mac. The image had CPython 3.11.18 and glibc 2.41.
The container mounted the worktree read-only. These commands made the
container, ran the tests and the check, and removed the container:

```bash
docker run -d --name d79-triton-interp --platform linux/arm64 \
  -v "$PWD":/src:ro -v /tmp/d79-out:/out \
  -e TRITON_INTERPRET=1 -e PYTHONDONTWRITEBYTECODE=1 -e PIP_NO_CACHE_DIR=1 \
  python:3.11-slim sleep 86400
docker exec d79-triton-interp pip install numpy==1.26.4 pytest==9.0.2 triton==3.9.0
docker exec d79-triton-interp pip install torch==2.15.0 \
  --index-url https://download.pytorch.org/whl/cpu
docker exec -w /src -e PYTHONPATH=/src/src d79-triton-interp bash -c \
  "python -m pytest -q -p no:cacheprovider --basetemp=/tmp/pt tests/test_triton_matmul_*.py"
docker exec -w /src -e PYTHONPATH=/src/src d79-triton-interp \
  python -m ai_journey.triton_matmul_cli check \
  --config config/day-79-triton-matmul.json \
  --output /out/day-79-triton-matmul.json
docker rm -f d79-triton-interp
```

The container had triton 3.9.0, torch 2.15.0+cpu, NumPy 1.26.4 and pytest
9.0.2. In the container, all 177 tests of the 10 Triton test modules passed,
with no skip. On the Mac, 164 tests passed and 2 skipped: the interpreter
module and the CLI test that runs the interpreter.

The check ran on 2026-11-22 at 04:47 IST. It did 108 runs: 9 cases, 6 configs
and 2 families. All 54 normal runs kept the bound, and all 54 exact runs held
the exact bits. Each guard tail kept its sentinels. The report digest is
`7d9b3c3005859ba6672b98c73c44869befbbbcec2e3af48761546f4d579de3af`.

| Case | M x N x K | Bound | Largest relative error | Share of the bound | Elements off the rounded reference | Exact runs with exact bits |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 x 1 x 1 | 5.96e-08 | 3.07e-08 | 0.516 | 0 of 1 | 6 of 6 |
| 1 | 17 x 33 x 9 | 5.36e-07 | 1.08e-07 | 0.201 | 288 of 561 | 6 of 6 |
| 2 | 32 x 32 x 16 | 9.54e-07 | 1.64e-07 | 0.171 | 634 of 1,024 | 6 of 6 |
| 3 | 31 x 47 x 129 | 7.69e-06 | 6.65e-08 | 0.00864 | 1,115 of 1,457 | 6 of 6 |
| 4 | 129 x 65 x 1000 | 5.96e-05 | 3.77e-08 | 0.000633 | 6,966 of 8,385 | 6 of 6 |
| 5 | 255 x 257 x 511 | 3.05e-05 | 4.99e-08 | 0.00164 | 52,498 of 65,535 | 6 of 6 |
| 6 | 64 x 48 x 4096 | 0.000244 | 3.33e-08 | 0.000136 | 2,698 of 3,072 | 6 of 6 |
| 7 | 3 x 1000 x 2048 | 0.000122 | 3.03e-08 | 0.000248 | 2,545 of 3,000 | 6 of 6 |
| 8 | 1000 x 999 x 100 | 5.96e-06 | 1.7e-07 | 0.0286 | 756,529 of 999,000 | 6 of 6 |

In each case, the six configs gave the same largest error and the same count,
so one row holds them all. The rounded reference is the float64 reference
rounded to float32. At 1000 x 999 x 100, 756,529 of the 999,000 elements
differ from it, and each element stays inside the bound. Thus bit equality is
not a usable test for a normal output.

For the six cases with K of 100 or more, the largest error was less than the
error of the k-order replay in `docs/gpu-tiled-cuda-contract.md`. At
64 x 48 x 4096, the two values are 3.33e-08 and 2.295e-07. The interpreter adds
the products in blocks of 32 and then adds each block to the accumulator. A
probable cause of the smaller errors is that each product then passes through
fewer roundings. These errors describe the NumPy arithmetic on this CPU, and
they do not predict the errors of a GPU.

The check took about 40 seconds of wall time. That time measures the CPU
interpreter, so it is not a performance number. Inside the container, the load
averages were 0.74, 0.40 and 0.55 before the check, and 2.03, 0.80 and 0.68
after it. On the Mac, `uptime` printed 7.81, 7.12 and 6.99 before, and 8.18,
7.35 and 7.08 after. On the Mac, `verify` checked the report in about 1 second,
without triton.

A second check ran on 2026-11-22 at 05:05 IST in the same container, with the
final code of this lab. Its report had the digest
`bcc9859fb1fda754e6c3bb0f11dc8c3d73aef7ab8269ce2726f0747288579c7d`, and the
repository does not keep that file. Each section was equal to the recorded
report, except the time stamp, the load averages and the digest. Thus each
output buffer of the interpreter had the same bits in both checks.

## A CI install of triton

On 2026-11-22, the PyPI JSON API for triton 3.9.0 listed these CPython 3.11
wheels:

| Platform | File | Bytes | SHA-256 |
| --- | --- | --- | --- |
| Linux x86_64 | `triton-3.9.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | 285,777,556 | `1622118a038b0c6e16fa4a0305da62c8b3c15904ee1cc4d4e65846a25969e41e` |
| Linux aarch64 | `triton-3.9.0-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` | 263,373,771 | `feaaec13578ca84301d561d034cb08b3ae293d7f0b7ab6b8ddbb2d6796363716` |

The metadata gives `Requires-Python: >=3.11,<3.16` and one dependency,
`importlib-metadata; python_version < "3.10"`. Thus triton needs no other
package on Python 3.11. The core metadata file of the x86_64 wheel has the
SHA-256 `5dfdd39550d7c33bb319d2427993a7cd259ea1d9209213938e5cab6dfc7373ae`.

The resolver uv 0.9.7 ran `uv pip compile` for `triton==3.9.0` with Python 3.11,
the platform `x86_64-manylinux_2_28` and `--only-binary :all:`. It resolved
`triton==3.9.0` and no other package. The manylinux tags need glibc 2.27 or
later. Thus `pip install triton==3.9.0` can install the x86_64 wheel on a Linux
CI host with Python 3.11 and such a glibc. The container install used the
aarch64 wheel. The repository CI has no job for this lab yet.

## Lines of code

`triton_matmul_loc` counts each text with one declared rule:

- A line is code when it still holds code after the counter removes its
  comments. Blank lines and lines with only a comment are not code. A Python
  docstring counts as a comment.
- Kernel lines are the code lines of a `__global__`, `kernel void` or
  `@triton.jit` function, with its decorator. A helper that a kernel calls also
  counts as kernel.
- Host lines are the code lines of the launcher and of each helper that only
  host code calls, such as `cdiv` in the CUDA texts. A Metal text has no host
  code, because the Day 76 and Day 77 runtimes launch its kernels from Python.
- Other lines are includes, imports and other top-level statements.
- The counts are physical lines, so they depend on the format of each text.

The rule does not count the `cublasSgemm` call text, because it is not a
kernel. Each counted text is the text whose digest its own lab guide pins.

| Lab | Language | Text | Kernel lines | Host lines | Other | Comment | Blank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Day 79 | Triton | `matmul_kernel` | 33 | 9 | 2 | 10 | 4 |
| Day 76 | CUDA C++ | `tiled_matmul_t8_kernel` | 24 | 9 | 0 | 10 | 3 |
| Day 76 | CUDA C++ | `tiled_matmul_t16_kernel` | 24 | 9 | 0 | 10 | 3 |
| Day 76 | CUDA C++ | `tiled_matmul_t32_kernel` | 24 | 9 | 0 | 10 | 3 |
| Day 77 | CUDA C++ | `coarsened_matmul_t16_c2_kernel` | 33 | 9 | 0 | 11 | 3 |
| Day 77 | CUDA C++ | `coarsened_matmul_t16_c4_kernel` | 33 | 9 | 0 | 11 | 3 |
| Day 77 | CUDA C++ | `coarsened_matmul_t32_c2_kernel` | 33 | 9 | 0 | 11 | 3 |
| Day 77 | CUDA C++ | `blocked_matmul_m64_n64_k16_t4x4_kernel` | 84 | 9 | 0 | 15 | 3 |
| Day 77 | CUDA C++ | `blocked_matmul_m128_n128_k8_t8x8_kernel` | 84 | 9 | 0 | 15 | 3 |
| Day 76 | Metal | `matmul_tiled_8` | 31 | none | 0 | 4 | 0 |
| Day 76 | Metal | `matmul_tiled_16` | 31 | none | 0 | 4 | 0 |
| Day 76 | Metal | `matmul_tiled_32` | 31 | none | 0 | 4 | 0 |
| Day 77 | Metal | `matmul_s2_coarse_4x1` | 126 | none | 0 | 3 | 0 |
| Day 77 | Metal | `matmul_s3_block_4x4` | 192 | none | 0 | 3 | 0 |
| Day 77 | Metal | `matmul_s4_vec4` | 222 | none | 0 | 3 | 0 |
| Day 77 | Metal | `matmul_s5_pad4` | 222 | none | 0 | 3 | 0 |
| Day 77 | Metal | `matmul_s6_group_8x8` | 316 | none | 0 | 3 | 0 |
| Day 77 | Metal | `matmul_s6_group_32x32` | 186 | none | 0 | 3 | 0 |
| Day 77 | Metal | `matmul_s7_interior` | 420 | none | 0 | 3 | 0 |

These are the results of the count:

- The Triton kernel has 33 code lines. Nine of them are the grouped order,
  which no CUDA text has. Without them, the kernel has 24 code lines, the count
  of the Day 76 CUDA tiled kernel.
- The host launch has 9 lines in each language. In CUDA, these lines are the
  launcher and its `cdiv` helper. In Triton, they are `launch_matmul`.
- One Triton text serves all six configs. The Day 76 lab writes one CUDA text
  for each tile, and the Day 77 lab writes one for each design.
- The Day 77 CUDA blocked text has 84 kernel lines for its micro-tiles and its
  `float4` path. The Triton text has no such lines, because its compiler
  decides that work. This lab did not see what the compiler did.
- A generator writes the Day 77 Metal texts. Thus their counts, 126 to 420,
  measure generated text and not the work of a programmer.
- A line count is not a speed. The lab measured no rate for any of these texts.

## Performance

This host has no NVIDIA or AMD GPU. Thus the lab gives no Triton GPU rate, no
CUDA rate and no cuBLAS rate. The CPU interpreter runs each program as Python
and NumPy code. Thus its time is not a performance number, and the report holds
no time.

The only matmul rates of this host come from two recorded Metal runs on its
Apple GPU. Each rate is a host-specific Apple GPU (Metal) measurement from one
run. Each cell gives the median GFLOP/s of 11 samples. The range in parentheses
comes from the IQR of the sample times. The lab copies these rates from the
guides. It measured none of them.

- The Day 76 report has the digest
  `371d65363ce5c53813c34a6edfc9db4725445bf4ffea605b7c81e0076fc03427`. It ran on
  2026-11-19 at 11:43 IST, with load averages of 9.93, 11.62 and 14.06 before
  the run and 9.70, 11.40 and 13.90 after it.
- The Day 77 report has the digest
  `130322a7b46db6bc7e9e9bf135a39225e2c139097cd167b74ceb22fd54b2e062`. It ran on
  2026-11-20 at 11:29 IST, with load averages of 19.21, 21.17 and 23.01 before
  the run and 16.63, 20.39 and 22.64 after it.

| M x N x K | Day 76 `matmul_tiled_32` | Day 77 `matmul_s3_block_4x4` | Day 77 `matmul_s7_interior` | Day 76 `torch.matmul` | Day 77 `torch.matmul` |
| --- | --- | --- | --- | --- | --- |
| 1024 x 1024 x 1024 | 338.5 (332.6 to 342.8) | 920.3 (571.7 to 1049.5) | 1312.0 (1145.2 to 1469.4) | 2294.1 (2002.0 to 2380.0) | 2246.2 (1840.6 to 2534.7) |
| 2048 x 2048 x 2048 | 339.6 (337.1 to 348.3) | 1177.9 (1116.9 to 1222.9) | 1774.7 (1668.4 to 1819.4) | 3608.8 (3464.3 to 3656.4) | 3390.3 (2710.8 to 3505.9) |
| 4096 x 4096 x 4096 | 353.5 (333.1 to 359.1) | 1160.2 (1153.2 to 1195.8) | 1825.8 (1815.1 to 1832.2) | 3940.1 (3703.5 to 4010.8) | 3521.4 (3250.9 to 3698.8) |
| 2000 x 3000 x 1000 | 342.3 (339.8 to 345.4) | 1334.0 (1293.8 to 1368.5) | 1665.8 (1505.8 to 1737.9) | 3448.9 (3269.8 to 3483.6) | 3042.8 (2966.5 to 3291.3) |

The table has these kernels:

- `matmul_tiled_32` has the tiling of the Day 76 CUDA text at tile 32.
- `matmul_s3_block_4x4` has the 64 x 64 x 16 tiles and the 4 x 4 outputs for
  each thread of the Day 77 CUDA text `blocked_matmul_m64_n64_k16_t4x4`.
- `matmul_s7_interior` was the fastest Metal kernel of the Day 77 run.
- `torch.matmul` ran on the MPS device. PyTorch does not report which kernel it
  ran.

| M x N x K | Triton on a GPU | Day 76 and Day 77 CUDA texts | cuBLAS |
| --- | --- | --- | --- |
| 1024 x 1024 x 1024 | not measured: no NVIDIA or AMD GPU on this host | not measured | not measured |
| 2048 x 2048 x 2048 | not measured: no NVIDIA or AMD GPU on this host | not measured | not measured |
| 4096 x 4096 x 4096 | not measured: no NVIDIA or AMD GPU on this host | not measured | not measured |
| 2000 x 3000 x 1000 | not measured: no NVIDIA or AMD GPU on this host | not measured | not measured |

## What a later GPU run must do

A later run on a GPU host must do these steps to give the missing numbers:

1. Use a Linux host with an NVIDIA or AMD GPU. Install triton 3.9.0 and a torch
   build for that GPU, leave `TRITON_INTERPRET` unset, and record the GPU name,
   the driver and the CUDA or ROCm version.
2. Check that `TRITON_SOURCE` still has its pinned SHA-256 digest. Run the
   correctness cases of `config/day-79-triton-matmul.json` on the GPU. Each
   normal output must keep the Day 74 bound, and each exact output must hold
   its exact bits.
3. Wrap `matmul_kernel` in `triton.autotune` with `AUTOTUNE_CONFIGS` and the
   key M, N, K. Record the config that the tuner picks at each shape and each
   config that fails to compile. The tuning times are not results.
4. Time the four shapes of the tables above with the Day 74 benchmark operands.
5. Give each side 2 calls that are not timed and then 11 timed samples, with
   CUDA events or `triton.testing.do_bench`. Move the first side by one at each
   repeat, as the Day 76 Metal lab does.
6. On an NVIDIA GPU, time the tuned Triton kernel and the Day 76 CUDA text at
   tile 32. Also time the five Day 77 CUDA texts and `sgemm_row_major` with
   cuBLAS. Give each side the same device operands, and check the Day 76 TF32
   assumption. On an AMD GPU, record that the CUDA texts and cuBLAS do not run
   there.
7. Record the load averages before and after the timing. Divide 2 * M * N * K
   by each median time to get GFLOP/s, and apply the Day 76 ratio rule. A ratio
   is faster only when it is 1.1 or more and the IQR ranges do not overlap.
8. Seal the times, the host records and the correctness results in a report
   that a verifier checks. Compare a GPU rate with a Metal rate in this guide
   only as a rate of other hardware.

## Reports and verification

A report is canonical JSON with sorted keys, compact separators and ASCII
escapes. It holds a SHA-256 digest over every other field. The digest shows
drift and tampering. It is not a signature, and it cannot prove that a run
occurred.

The report records these items:

- the plan, its digest, the environment without host names, the time of the
  check and the load averages before and after it
- the pinned text, its digest, the declared configs, the autotune key and the
  flag `tuned` with the value false
- the rules of the check, and each case with its operand digests, its bound and
  one record for each run
- the line count of each text, and the Metal rates with their report digests
- the unmeasured sides, the steps of a later GPU run and the PyPI facts
- a summary and the evidence boundary

Only a report from the interpreter backend gets the label
`triton_cpu_interpreter_check`. A report from another backend, such as the
NumPy stand-in of the tests, gets the label `synthetic_backend`. The verifier
keeps that label.

The verifier uses other code than the run where the run uses lab code. It
imports neither triton nor torch. It does these checks:

1. It draws the operands again from the seed. It calculates the reference and
   the bound with the code of the Day 74 Metal verifier.
2. It calculates the programs, the K blocks and the guard of each run by
   closed forms.
3. It builds each exact buffer again from the float64 reference and its
   sentinel tail, and it checks its digest bit for bit.
4. It checks the structure of each normal record. The record must hold a
   digest and an error inside the bound. Its ratio and its count must agree
   with its error and its digest.
5. It counts the lines of each text with a second counter. That counter strips
   `//` comments line by line for the C texts, and it uses `ast` spans for the
   Triton text.
6. It requires that the Metal rates, the later steps and the PyPI facts equal
   the pinned data. Then it calculates the summary again.

The tamper tests change one field, seal the report again, and require that the
verifier names the section of the change. The test module
`test_triton_matmul_recorded.py` holds the report of the container run, byte
for byte. A host without triton verifies that report and builds each exact
buffer again.

## Evidence boundary

The evidence boundary of an interpreter report holds true or false fields and
one statement. The fields `triton_interpreter_on_cpu` and
`metal_rates_copied_from_recorded_reports` are true. These fields are false:
`gpu_executed`, `triton_compiled_for_gpu`, `autotuned`, `cuda_compiled`,
`cublas_called`, `nvidia_or_amd_gpu_used` and `interpreter_time_reported`.

This lab does not claim these results:

- It ran no kernel on a GPU, and it used no NVIDIA or AMD GPU, CUDA toolkit,
  cuBLAS or hardware counter.
- It tuned no config. The interpreter ignored the warps and the stages.
- It states no Triton, CUDA or cuBLAS rate, no speedup and no fraction of a
  vendor peak.
- That a GPU keeps the bound is a prediction from the stated arithmetic. No GPU
  run observed it.
- The Metal rates are copies of two recorded runs on one Apple GPU. They do not
  predict Triton or CUDA rates on other hardware.
- The CPU times in this guide describe one shared host, and they are not GPU
  measurements.

## What Ajay must still do

- Ajay must still watch Lecture 14, "Practitioner's Guide to Triton", and
  Lecture 18, "Fused Kernels".
- Ajay must still check the tracker notes against the two lectures.
- Ajay must still run the steps of a later GPU run to get the missing numbers.

## Sources

- GPU MODE lectures, https://github.com/gpu-mode/lectures: Lecture 14,
  "Practitioner's Guide to Triton", and Lecture 18, "Fused Kernels".
- triton-lang/triton at tag v3.9.0: `python/tutorials/03-matrix-multiplication.py`
  and `docs/programming-guide/chapter-1/introduction.rst`.
- The Triton 3.9.0 wheel: the docstrings of `tl.dot`, `tl.make_block_ptr`,
  `tl.make_tensor_descriptor`, `triton.autotune` and `triton.Config`, and
  `triton/runtime/interpreter.py`.
- PyPI, https://pypi.org/pypi/triton/3.9.0/json, and the core metadata file of
  the x86_64 wheel.
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
  2002, section 3.1, for the bound `gamma_K`.
