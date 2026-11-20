# Tiled CUDA matmul contract

This lab is the CUDA and cuBLAS half of Day 76. It writes a tiled matmul
kernel in CUDA C++ for the tile widths 8, 16 and 32. It also writes one
row-major `cublasSgemm` call. Both texts are source for a later run on an
NVIDIA GPU.

This host has no NVIDIA GPU, no CUDA toolkit, no `nvcc` and no cuBLAS. Thus
the lab compiles, links and runs none of the texts. It checks the texts
without a compiler, and it runs the schedule of the kernel in a CPU
simulator. It also estimates a declared occupancy. Each run seals a report,
and a verifier checks the report again.

## Commands

Run the pinned plan, check the report, and write the source files:

```bash
PYTHONPATH=src python -m ai_journey.gpu_tiled_cli run \
  --output /tmp/day-76-gpu-tiled-cuda.json

PYTHONPATH=src python -m ai_journey.gpu_tiled_cli verify \
  --report /tmp/day-76-gpu-tiled-cuda.json

PYTHONPATH=src python -m ai_journey.gpu_tiled_cli sources \
  --output-dir /tmp/day-76-cuda-sources
```

Without `--config`, `run` uses the pinned Day 76 plan in
`gpu_tiled_config.PINNED_PLAN`. With `--config`, it uses a plan file. `run`
and `verify` print one line:

```text
gpu_tiled_report=2419f0bfedcffd5eaa363e07640023e4e81a4bcfc752a415fd8a30158f06ffae plan=day-76-gpu-tiled-cuda cases=9 tiles=3 replay_matches=27/27 traffic_matches=27/27 race_free_runs=27/27 cuda_checks=297/297 cublas_checks=63/63 occupancy_estimates=9 simulated_on_cpu=true gpu_executed=false cuda_compiled=false cublas_called=false
```

`verify` prints the same line with the prefix `verified_gpu_tiled_report=`.
With `--config`, `verify` also requires that the report embeds that plan.
`sources` writes four files and a `SHA256SUMS` file, and it prints one line
for each source file. The exit codes are:

- `0`: the command succeeded.
- `1`: the command read or built a report, but the report failed
  verification.
- `2`: the arguments are wrong, an input is invalid or unreadable, or the
  command refused the output path.

No command needs a GPU, so no exit code marks a missing device. `run`
refuses an output path that names the plan, also through a symlink or a
hard link. The package has no console script for this lab yet, so the
commands use `python -m`.

On the development host, `run` took 30 s and `verify` took 14 s of wall
time on the CPU. The 1-minute load averages were 11.51 before `run`, 17.52
after it and 15.58 after `verify`. These times describe one shared host and
no GPU.

## Tiled kernel

`gpu_tiled_cuda.tiled_cuda_source(tile=T)` writes one kernel and its host
launcher. One T x T block computes one T x T tile of `C = A @ B`. `A` is
`m x k`, `B` is `k x n` and `C` is `m x n`. All three are row-major float32
matrices.

The kernel runs `cdiv(k, T)` phases. In each phase, each thread does these
steps:

1. It loads one element of `A` into `As[ty][tx]` and one element of `B`
   into `Bs[ty][tx]`. An element outside its matrix gives 0, and the thread
   reads no global memory for it.
2. It waits at `__syncthreads()`, so every load is in shared memory before a
   thread reads a slot.
3. It runs `i` from 0 while `i < min(T, k - phase * T)`, and it applies
   `sum = __fmaf_rn(As[ty][i], Bs[i][tx], sum)` at each step.
4. It waits at `__syncthreads()` again, so no thread loads the next phase
   over a value that another thread still reads.

After the last phase, each thread inside `C` stores its sum. Only the store
has a guard. Every thread must reach each barrier, so no other statement
has a guard or an early return. This is the phase loop of the T = 16 text:

```c
    for (int phase = 0; phase * 16 < k; ++phase) {
        int a_col = phase * 16 + tx;
        int b_row = phase * 16 + ty;
        As[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < min(16, k - phase * 16); ++i) {
            sum = __fmaf_rn(As[ty][i], Bs[i][tx], sum);
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
```

`__fmaf_rn` rounds `As[ty][i] * Bs[i][tx] + sum` once to float32, with ties
to even. The loop adds the terms in increasing k from +0.0. That is the loop
of `matmul_metal_reference.fma_replay`, so each output must hold the bits of
that replay. The inner bound stops a partial last phase at `k`, so the loop
reads no padded slot.

| T | Threads per block | Static shared memory | SHA-256 of the text |
| --- | --- | --- | --- |
| 8 | 64 | 512 bytes | `3b72d60eb486fb5d1671efcf5428d3b6d93f4c61d0e6ec1b4c97815be31884fa` |
| 16 | 256 | 2,048 bytes | `a789a301b41dde5d8449860297a0c5340fabfbc9c8c39209fa15791b038fc867` |
| 32 | 1,024 | 8,192 bytes | `178e6d6ec2976d70ef02e0f9cc38eedbdacdd863588b3e441fecc393826b7d6e` |

Each block holds two float tiles, so it uses `2 * T * T * 4` bytes of static
shared memory. The launcher sets `dim3 block(T, T)` and
`dim3 grid(cdiv(n, T), cdiv(m, T))`. Each text defines its own `cdiv`, so a
later run compiles each file as its own translation unit.

## Checks without a compiler

`check_tiled_source` parses a text with the helpers of the Day 73
checker. It reads comments as a compiler reads them. It refuses a line
splice, a trigraph, and a space or a comment inside an operator such as
`++`. It accepts only the three pinned functions, because one extra
statement, overload or macro can change the kernel. Such text fails the
check of its part, or it fails every check. The 11 checks are these:

- `cdiv_definition` and `tiled_signature`: the pinned `cdiv` and the pinned
  kernel signature.
- `tiled_shared_memory`: two `__shared__ float` tiles of `[T][T]`. The check
  reads the extents from the text and requires `2 * T * T * 4` bytes.
- `tiled_index`: `row` is `blockIdx.y * T + ty`, and `col` is
  `blockIdx.x * T + tx`.
- `tiled_phase_loop`: `for (int phase = 0; phase * T < k; ++phase)`, with no
  statement between the loop and the store guard.
- `tiled_loads`: one element of `A` and one element of `B` for each thread in
  each phase, with zero fill.
- `tiled_barriers`: one barrier after the loads, one barrier after the inner
  loop, and no other barrier in the kernel.
- `tiled_inner_loop`: `for (int i = 0; i < min(T, k - phase * T); ++i)`.
- `tiled_fma_order`: the sum is `0.0f` before the first phase, and the
  inner loop holds only the `__fmaf_rn` statement.
- `tiled_store_guard`: one guard `row < m && col < n` around the store, and
  no other `if`, `return` or loop exit.
- `tiled_grid`: the host `dim3` expressions, with C unsigned division, give
  the Day 74 launch of a `(T, T)` block for the case shape.

The tests change one statement at a time. For example, a text without the
first barrier fails `tiled_barriers`. A text with `i < 16` fails
`tiled_inner_loop`. A text with `+ +i` fails every check. A compiler reads
two unary plus signs there and no increment, so the loop does not stop.

## Row-major cuBLAS call

`cublas_source()` writes one function, `sgemm_row_major`. cuBLAS reads
column-major matrices. The bytes of the row-major `m x n` matrix `C` are the
bytes of the column-major `n x m` matrix `C^T`. Thus the call computes
`C^T = B^T A^T`:

```c
    return cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta,
        c, n);
```

`B^T` is `n x k` with the leading dimension `n`. `A^T` is `k x m` with the
leading dimension `k`. `C^T` has the leading dimension `n`.

Before the call, the function sets the host pointer mode for `alpha = 1` and
`beta = 0`. It also sets `CUBLAS_DEFAULT_MATH`. It returns each status that is
not `CUBLAS_STATUS_SUCCESS`. The SHA-256 of the text is
`9a86b34a84f3550412c2971a11b3972f6d1595b67193812afdb9583e1c36f889`.

Assumption: with `CUBLAS_DEFAULT_MATH`, `cublasSgemm` uses no TF32 Tensor
Core math. This statement comes from the cuBLAS documentation of
`cublasSetMathMode`. This lab did not check it. A later run must check it on
the NVIDIA host. The text states the same assumption in a comment, and the
report holds it in the field `tf32_assumption`.

`check_cublas_source` has 7 checks. They cover the include line, the
signature, the scalars, the pointer mode, the math mode, the form of the
call and the row-major result. The last check evaluates the parsed call
under the column-major rules of cuBLAS, with exact integers, on two probe
shapes. The checked call pattern can name 78,732 calls. A test evaluates
all of them, and only the pinned call gives `C = A @ B` on both probes. The
check also requires that each leading dimension fits the buffers of the case
shape.

## CPU block simulator

`gpu_tiled_blocksim.run_tiled_blocks` holds all blocks of a launch as numpy
float32 arrays. It runs the steps of the CUDA text in order, phase by phase:
load, barrier, fma, barrier. A product of two float32 values is exact in
float64. The simulator rounds each exact sum once to float32 with the
rounding helper of `matmul_metal_reference`.

The simulator counts each element that a thread reads from `A` or `B`. It
also logs the shared memory accesses of one block between barriers. The
shared indices do not depend on `blockIdx`, so the accesses of one block
represent the accesses of every block. Two kinds of access are hazards:

- A read-after-write hazard is a read of a slot that another thread wrote
  in the same interval between barriers.
- A write-after-read hazard is a write of a slot that another thread read
  in the same interval.

The pinned plan runs the nine Day 74 correctness cases at each tile, with
the seed 74. These are the results:

- All 27 runs hold the bits of `fma_replay` for their case.
- In each run, the counted loads, the zero-filled slots and the stores equal
  the counts of `memory_access_tiling.TiledMatmulTraffic`.
- Each run stores each element of `C` once, and it logs no hazard.

For the case 1000 x 999 x 100 at T = 8, the grid has 125 block columns and
125 block rows. Each block column reads all of `A` once, and each block row
reads all of `B` once. Thus the run counts 12,500,000 loads of `A` and
12,487,500 loads of `B`.

The tests also remove each barrier. For a 37 x 45 x 29 case at T = 16, the
values stay the same. The simulator runs each step for all threads before
the next step. The hazard log shows each race:

| Phase steps | Read-after-write | Write-after-read |
| --- | --- | --- |
| load, barrier, fma, barrier | 0 | 0 |
| load, fma, barrier | 13,920 | 0 |
| load, barrier, fma | 0 | 7,680 |

CUDA does not specify an order for the warps of a block. Thus, without a
barrier, an output can depend on that order.

## Replay digests and error bounds

Case `i` draws its operands as the Day 74 Metal lab draws its case `i`, with
the seed 74. The table gives the replay digest of each case. It also gives
the largest relative error of the replay against the float64 reference, and
the error bound of the case.

| Case | m x n x k | Replay SHA-256 | Replay error | Error bound |
| --- | --- | --- | --- | --- |
| 0 | 1 x 1 x 1 | `c191379486fedbbc3670285734a47dca9ca4cf00935f00fe38571af72588e654` | 3.073e-08 | 5.960e-08 |
| 1 | 17 x 33 x 9 | `3b282a4121200a322b2e56328041b32cf50a6ffd2cefa1437e35ef6e8944795e` | 1.078e-07 | 5.364e-07 |
| 2 | 32 x 32 x 16 | `99c7d073bbbe251d71b0f79a86f532ec170e424667ee6e6a88c26b95a40a324c` | 1.635e-07 | 9.537e-07 |
| 3 | 31 x 47 x 129 | `df4e9195e86c3d69d512e764bec264c1fd10291179db24bbf6e0bc1db2b87356` | 1.494e-07 | 7.689e-06 |
| 4 | 129 x 65 x 1000 | `1099e8736feff06bb7dc6bfc2c2fee0ef5f77f12e5ae2a247ee79e37f2190e66` | 2.772e-07 | 5.961e-05 |
| 5 | 255 x 257 x 511 | `2c02aa5cbfd0c93fce859735369ded18cce0a55e3f96c61c135162b7e8f33735` | 2.621e-07 | 3.046e-05 |
| 6 | 64 x 48 x 4096 | `e52164d2122ed3281ad076a3d4fe09aad0949f28ce44a9a12163bdc12e65f0d0` | 2.295e-07 | 2.442e-04 |
| 7 | 3 x 1000 x 2048 | `9c0b2e517abfeef3b183b4dfb217bebb7b139b9759c8704cdb8a266e1e34dc13` | 2.359e-07 | 1.221e-04 |
| 8 | 1000 x 999 x 100 | `2a6ff03da26f480c69009b777d34f831008ca552e3e9d70738a65f6011936dcb` | 3.318e-07 | 5.961e-06 |

A recorded Day 74 report from the Apple GPU of the development host holds
cases 0 and 1 with the same seed, index and shape. Its replay digests for
these cases equal the digests in this table.

The relative error of an element is `|C - R| / (|A| @ |B|)`, where `R` is
the float64 reference. Its bound is `gamma_K(2**-24) + gamma_K(2**-53)`. The
bound holds for each order of the additions, so a float32 cuBLAS output must
also stay inside it.

## Declared occupancy

`gpu_tiled_occupancy` follows `matmul_model_occupancy`. It uses the SIMT
calculator rules and the compute capability 8.0 record. Each block declares
`T * T` threads and `2 * T * T * 4` bytes of static shared memory. The
register counts 32, 40 and 64 are assumptions, because this lab compiles no
kernel. Each cell gives the theoretical occupancy and the resident blocks
per SM:

| T | 32 registers | 40 registers | 64 registers |
| --- | --- | --- | --- |
| 8 | 1.000, 32 blocks | 0.750, 24 blocks | 0.500, 16 blocks |
| 16 | 1.000, 8 blocks | 0.750, 6 blocks | 0.500, 4 blocks |
| 32 | 1.000, 2 blocks | 0.500, 1 block | 0.500, 1 block |

With the 1 KB that the driver keeps for each block, one block allocates
1,536, 3,072 or 9,216 bytes. Then shared memory allows 109, 54 or 18 blocks
per SM, so it limits no estimate. The warp slots and the registers set each
value. At 40 registers, the registers allow 48 warps per SM. One 32 x 32
block needs 32 warps, so a second block does not fit. These values are
theoretical occupancy, not measured occupancy, and they predict no speed.

## Report and verification

The report holds the plan and the four source texts with their digests. It
also holds the checks of each case, the run of each tile and the occupancy
estimates. Its digest is the SHA-256 of its canonical JSON. The pinned report has the
digest `2419f0bfedcffd5eaa363e07640023e4e81a4bcfc752a415fd8a30158f06ffae`.

`verify` trusts no number in a report. It recomputes these values with
other code than the run used:

- The operands, the float64 reference, the error bound and the replay come
  from the code of the Day 74 Metal verifier. That replay rounds to odd in
  float64. The run's replay uses a TwoSum error term at each float32
  midpoint instead. Thus the two methods check each other.
- The grid, the phases, the loads and the zero-filled slots come from closed
  forms, with no call to the simulator or to `TiledMatmulTraffic`.
- The occupancy comes from the 8.0 values that the Day 74 model verifier
  pins.

Last, `verify` runs the simulator again. Each new run must equal the
recomputed run. A failure names the first field that differs.

## What a later NVIDIA run must do

This lab produces no GFLOP/s number. A later run on an NVIDIA GPU must do
these steps to produce the tiled CUDA number and the cuBLAS number:

1. Run `sources`, and copy the four files to the NVIDIA host. Check them
   with `sha256sum -c SHA256SUMS`.
2. Record the GPU name, its compute capability, the driver version, the CUDA
   toolkit version and the cuBLAS version.
3. Record the value of `NVIDIA_TF32_OVERRIDE`. This variable can change the
   TF32 behavior of NVIDIA libraries.
4. Compile each `.cu` file alone with `nvcc` for the compute capability of
   the GPU. Do not use `--use_fast_math` or `-ftz=true`. Both can flush
   subnormal values to zero.
5. Make the operands of each case with `matmul_metal_reference.matmul_inputs`
   from the seed and the case index. Copy them to the GPU.
6. Run each tiled kernel on each case. Each output must have the replay
   digest of its case, by `matmul_metal_reference.matrix_sha256`.
7. Run `sgemm_row_major` on each case. For each output, `max_relative_error`
   against the float64 reference must not exceed `error_bound(k)`.
8. Check the TF32 assumption on that host, for example from the names of the
   kernels that cuBLAS launches in a profiler trace. Until a run checks it, do
   not report the cuBLAS number as a float32 result.
9. Time each tiled kernel and each cuBLAS call after warm-up calls, for
   example with CUDA events. Use the benchmark shapes of
   `config/day-74-matmul-metal.json`.
10. Record the load averages of the host before and after the timing.
11. Divide `2 * m * n * k` by each median time to get GFLOP/s. Seal the
    times, the host records and the correctness results in a report that a
    verifier checks.

The outputs of steps 6 and 7 come from the GPU. The digests and bounds that
these steps use come from the report of this lab.

## What this lab does not claim

- The lab compiled, linked and ran no CUDA or cuBLAS text. It does not know
  that `nvcc` accepts the texts.
- The lab used no NVIDIA GPU, no CUDA toolkit, no cuBLAS, no Nsight and no
  hardware counter.
- The lab measured no GPU time. It states no GFLOP/s, no speedup, no vendor
  peak and no fraction of a peak.
- That an NVIDIA GPU gives the replay bits is a prediction from the stated
  rounding of `__fmaf_rn`. No GPU run observed it.
- The TF32 statement is an assumption from the cuBLAS documentation.
- The occupancy values come from declared register counts. They are not
  achieved occupancy.
- The CPU times are wall times on one shared host. They are not GPU
  measurements, and they say nothing about production or distributed
  performance.

## Sources

- NVIDIA, CUDA C++ Best Practices Guide v13.5, section 10.2.3.2, for the
  tiled matmul in shared memory.
- NVIDIA, CUDA Math API, for `__fmaf_rn`.
- NVIDIA, cuBLAS documentation, for `cublasSgemm`, `cublasSetMathMode` and
  `cublasSetPointerMode`.
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
  SIAM, 2002, section 3.1, for the bound `gamma_K`.
- S. Boldo and G. Melquiond, "Emulation of FMA and correctly rounded sums:
  proved algorithms using rounding to odd", IEEE Transactions on Computers
  57(4), 2008, for the rounding to odd.
