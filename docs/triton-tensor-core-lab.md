# Tensor Cores in the Day 82 Triton matmul

Day 82 of the curriculum follows GPU MODE Lecture 23, "Tensor Cores", and
Lecture 15, "CUTLASS". The tracker note is "Tensor Core MMA, data layouts,
swizzling." The hands-on task is "Benchmark a matmul with and without Tensor
Cores (fp32 vs bf16). Explain the speedup ratio." This lab is the NVIDIA half
of Day 82. The plan date was 2026-11-25, and the lab code dates from 2026-11-26.

This host has no NVIDIA GPU, so the lab cannot time a matmul on one. It does
these steps instead:

1. It pins one Triton matmul text and declares four variants of it: fp32 with
   IEEE math, fp32 with TF32, fp16 and bf16.
2. It compiles each variant offline for the declared targets sm_80 and sm_90,
   on a Linux host with triton 3.9.0. No GPU is necessary for this step.
3. It reads the PTX and the TTGIR of each compile with explicit rules and
   checks the instruction classes that each variant must use.
4. It models the fp32, tf32 and bf16 times of square matmuls on A100 and H100
   from declared datasheet inputs, and it explains the ratio.
5. It seals one report, and an independent verifier checks the report.

Nothing here ran on an NVIDIA GPU. The compiler output comes only from the
Linux CI job `triton-tc-contracts`, and that job had not run when this guide
got its text. The model times are model output from declared inputs. They
are not measurements.

Ajay must still watch Lecture 23 and Lecture 15. Ajay must still read the PTX
and TTGIR that the CI job keeps. This guide shows where to start. It does not
replace the lectures or the reading.

## What an MMA instruction computes

A Tensor Core runs one matrix multiply-accumulate (MMA), `D = A x B + C`, on
small tiles. One instruction does the work of many scalar FMAs. PTX names
the instruction in two forms:

- `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`: one warp of 32
  threads computes a 16 x 8 x 16 product. That is 2048 multiply-adds. The
  types are D, A, B and C in that order. This is the form of sm_80.
- `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`: the TF32 form. Its K
  is 8, so one instruction does 1024 multiply-adds.
- `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`: one warpgroup of 4
  warps computes a 64 x 128 x 16 product. A and B can come from shared memory
  through descriptors. The instruction runs asynchronously until a
  `wgmma.wait_group`. Only sm_90a has this form.

Without a Tensor Core, the same dot product runs on the FMA lanes. One
`fma.rn.f32` is one multiply-add of one thread.

## Fragments and data layouts

An MMA instruction reads its operands from registers in a fixed layout. Each
thread of the warp holds one part of each tile, a fragment. For
`m16n8k16` with bf16 inputs, the 16 x 16 A tile has 256 elements, so each of
the 32 threads holds 8 of them. The 16 x 8 B tile gives 4 elements to each
thread, and the 16 x 8 accumulator gives 4 float32 values to each thread.

TTGIR names these layouts as encodings:

- `#ttg.nvidia_mma<{versionMajor, versionMinor, warpsPerCTA, instrShape}>`
  is the accumulator layout. Version 2 is the `mma.sync` layout, and version
  3 is the `wgmma` layout. `instrShape` is the shape of one instruction.
- `#ttg.dot_op<{opIdx, parent, kWidth}>` is the register layout of operand A
  (`opIdx = 0`) or B (`opIdx = 1`). `kWidth` is the number of consecutive K
  elements that one thread holds.
- A `#ttg.blocked` parent means no Tensor Core. The recorded Day 80 TTGIR of
  the Day 79 kernel has this form: each thread runs scalar FMAs.

`ldmatrix.sync.aligned.m8n8.x4.shared.b16` loads four 8 x 8 matrices of
16-bit values from shared memory into this fragment layout. One instruction
feeds a whole warp.

## Swizzling as the TTGIR shared encoding shows it

Shared memory has 32 banks of 4 bytes. A warp that reads 8 rows of a tile at
the same column waits when the rows start in the same bank. That is a bank
conflict. A row of 64 fp16 values is 128 bytes, so column 0 of every row is
in bank 0.

`#ttg.swizzled_shared<{vec, perPhase, maxPhase, order}>` states the fix as a
rule. The rule moves each group of `vec` elements of a row to group
`group XOR ((row // perPhase) % maxPhase)`. An element keeps its place inside
its group. `swizzled_column` and `shared_bank` in
`ai_journey.triton_tc_ttgir` apply the rule.

This example uses parameters that the lab chose by hand, not compiler output:
`vec = 8`, `perPhase = 1`, `maxPhase = 8` on 64-column fp16 rows. Column 0 of
rows 0 to 7 then lands in banks 0, 4, 8, 12, 16, 20, 24 and 28. With
`maxPhase = 1`, all 8 rows land in bank 0. The recorded Day 80 TTGIR of the
Day 79 fp32 kernel holds `vec = 1, perPhase = 1, maxPhase = 1`: no swizzle.

On Hopper, `#ttg.nvmma_shared<{swizzlingByteWidth, transposed,
elementBitWidth}>` is the shared layout that a `wgmma` descriptor reads. It
names the swizzle by its byte width, for example 128. The analyzer records
both kinds. The values for the Day 82 compiles come from the CI dump. This guide
states none of them, because the CI job had not run.

## The four variants

The pinned text is `tc_matmul_kernel` from the Day 82 plan file, byte for
byte. Its SHA-256 is
`6833430d2fea2ac4557766b4fef78b335d34d97b513058c6ec1ad7c2ee84fd9a`. All four
variants compile it with one config: `BLOCK_M = 128`, `BLOCK_N = 128`,
`BLOCK_K = 32`, 4 warps and 3 stages. C is float32 in every variant.

| Variant | A and B | `input_precision` | Tensor Core | MMA operand type |
|---|---|---|---|---|
| `fp32-ieee` | fp32 | `ieee` | no | none |
| `fp32-tf32` | fp32 | `tf32` | yes | `tf32` |
| `fp16` | fp16 | `ieee` | yes | `f16` |
| `bf16` | bf16 | `ieee` | yes | `bf16` |

`input_precision` applies only to fp32 inputs. Triton ignores it for fp16 and
bf16, and it uses a Tensor Core for the 16-bit product with float32
accumulation.

## The analyzer rules

`ai_journey.triton_tc_ptx` splits a PTX text into statements at each `;`. It
removes comments, directives, labels and predicate guards, and it takes the
first word as the opcode. Thus it also counts an MMA inside inline asm, the
form that Triton writes.

Comments follow the C rules of PTX. A `/* */` comment can span lines, and a
comment mark inside a comment or a string starts no comment. The analyzer
refuses a text with an unpaired `/*` or `*/`. The verifier applies the same
rules with its own expressions.

The first rule that matches an opcode gives its class:

| Class | Rule |
|---|---|
| `mma_sync` | prefix `mma.sync.aligned.` |
| `wgmma_mma_async` | prefix `wgmma.mma_async.` |
| `wgmma_fence`, `wgmma_commit_group`, `wgmma_wait_group` | exact opcode |
| `ldmatrix`, `stmatrix` | prefix `ldmatrix.sync.aligned.`, `stmatrix.sync.aligned.` |
| `cp_async_bulk` | prefix `cp.async.bulk.` |
| `cp_async_commit_group`, `cp_async_wait_group` | exact opcode |
| `cp_async` | prefix `cp.async.ca.` or `cp.async.cg.` |
| `cvt_tf32` | a `cvt.` opcode that ends in `.tf32.f32` |
| `fma_rn_f32` | exact opcode `fma.rn.f32` |

An MMA opcode must name its `mNnNkN` shape and its operand types. If it does
not, the analysis fails. A rule that does not match the compiler output must
not count in silence. `fma.rn.ftz.f32` does not count as `fma_rn_f32`.

`static_tile_products` multiplies each count by the multiply-adds of its
instruction and by the threads, warps or warpgroups of one program. Then it
divides by one tile product, 128 x 128 x 32. A result of 1 says that the
static text holds exactly one tile product. The recorded Day 80 PTX of the
Day 79 kernel gives 1: 1024 `fma.rn.f32` per thread, 64 threads, and a
32 x 64 x 32 tile.

`ai_journey.triton_tc_ttgir` reads the encoding aliases, the inline
encodings, the module attributes, the dot ops and each `ttg.local_alloc`
buffer with its shape, element type and bytes.

The unit tests use hand-written SYNTHETIC PTX and TTGIR snippets. Their test
names and docstrings say so. They are not compiler output. Four tests also
read the recorded Day 80 texts of the Day 79 kernel, which are real triton
3.9.0 output. On those texts, the analyzer and the Day 80 reader count the
same 3097 instructions.

## The instruction-class rule

`ai_journey.triton_tc_rule` states the class that each compile must hold:

| Variant | sm_80 | sm_90 |
|---|---|---|
| `fp32-ieee` | `scalar_fma` | `scalar_fma` |
| `fp32-tf32`, `fp16`, `bf16` | `mma_sync` | `wgmma`, or `mma_sync` recorded |

The lab expects `wgmma` on sm_90, because Hopper has it and the 4-warp 128 x
128 tile fills one warpgroup. No compile checked this before this guide. A
Hopper GPU also runs `mma.sync`, so that choice belongs to the compiler. The
rule must match what Triton 3.9 really does. Thus the rule accepts `mma.sync`
on sm_90, adds a note to the row, and lists the variant in
`mma_sync_on_sm_90`. The test does not fail on it.

Each compile must also pass these checks:

1. The PTX target names the compile target, and the TTGIR names `cuda:80` or
   `cuda:90`.
2. Every MMA takes the operand type of the variant and accumulates in `f32`.
3. The TTGIR agrees with the PTX: `mma.sync` with an `nvidia_mma` version 2,
   `wgmma` with version 3, and scalar FMA with no `nvidia_mma`.
4. The static text holds a whole number of tile products of its class.
5. In a Tensor Core variant, scalar FMAs hold less than one tile product.
6. Each shared buffer of the TTGIR fits in the shared memory bytes of the
   compile metadata.

## Commands

The model and the verifier run on any host:

```bash
PYTHONPATH=src python -m ai_journey.triton_tc_cli model --config config/day-82-triton-tc.json --output /tmp/day-82-model.json
PYTHONPATH=src python -m ai_journey.triton_tc_cli verify --report /tmp/day-82-model.json --config config/day-82-triton-tc.json
```

The compile needs triton 3.9.0, so it runs only on Linux:

```bash
PYTHONPATH=src python -m ai_journey.triton_tc_cli compile --config config/day-82-triton-tc.json --output /tmp/day-82-compile.json --dump /tmp/day-82-dump
PYTHONPATH=src python -m ai_journey.triton_tc_cli verify --report /tmp/day-82-compile.json --config config/day-82-triton-tc.json --dump /tmp/day-82-dump
```

The parser uses the program name `ai-journey-triton-tc`. An installed wheel
gives the same commands under that name.

On a host without triton, `compile` exits 3 before it makes the output folder
or the dump folder. The exit codes are:

- 0: the command succeeded.
- 1: a report failed verification, or a compile broke the rule. A compile
  report is still written, so the cause can be read.
- 2: a usage error, an unreadable or invalid input, or a refused output.
- 3: this host cannot compile with triton, so `compile` wrote nothing.

The CI-only test `tests/test_triton_tc_compile_contract.py` compiles all
eight variants and targets with the real triton. On a host without triton,
it skips at import. When `TRITON_TC_DUMP_DIR` names a folder, it writes every
text, the index and `summary.json` there.

The CI job `triton-tc-contracts` runs the Day 82 tests on a Linux runner with
triton 3.9.0 and fails on a skipped test. It sets `TRITON_TC_DUMP_DIR` and
keeps that folder as an artifact. Then it runs the installed `compile` with a
dump, verifies the report with the dump, and runs `model`. That job did not
run yet.

## The declared inputs

`config/day-82-triton-tc.json` holds every device number with its source page,
the words that the page states, and the status `declared`. The pages were
read on 2026-11-26. The H100 boost clock is `not_stated`, because the cited
pages give no final H100 SXM clock.

| Input | A100 80GB SXM | H100 SXM | Source |
|---|---|---|---|
| SMs | 108 | 132 | Ampere and Hopper In-Depth blogs |
| FP32 FMA per clock per SM | 64 | 128 | Ampere and Hopper In-Depth blogs |
| TF32 dense FMA per clock per SM | 512 | 1024 | Ampere and Hopper In-Depth blogs |
| FP16 dense FMA per clock per SM | 1024 | 2048 | Ampere and Hopper In-Depth blogs |
| BF16 dense FMA per clock per SM | 1024 | 2048 | Ampere and Hopper In-Depth blogs |
| FP32 peak, TFLOPS | 19.5 | 67 | A100 and H100 product pages |
| TF32 Tensor Core peak, TFLOPS | 156 | 989 with sparsity | A100 and H100 product pages |
| FP16 and BF16 Tensor Core peak, TFLOPS | 312 | 1979 with sparsity | A100 and H100 product pages |
| Memory bandwidth, GB/s | 2039 | 3350 | A100 and H100 product pages |

The source URLs are:

- <https://www.nvidia.com/en-us/data-center/a100/>
- <https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/>
- <https://www.nvidia.com/en-us/data-center/h100/>
- <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>

The Hopper blog states that sparsity doubles the Tensor Core rate. Thus the
model halves each H100 figure "with sparsity": 494.5 TFLOPS for TF32 and
989.5 TFLOPS for FP16 and BF16. The A100 TF32 and BF16 rates per SM come
from the Ampere table. At one clock and one SM count, the TF32 peak is half
of the FP16 peak, and the BF16 peak is equal to it.

Two checks hold the inputs together. At the declared 1410 MHz, 108 SMs and
the per-SM rates reproduce each A100 peak within its rounding. The A100
record also agrees with the Day 71 pinned A100 record.

## The speedup model

`ai_journey.triton_tc_model` states its rule:

1. A square N x N x N matmul does 2 N^3 FLOPs.
2. The compute ceiling of a path is its listed dense peak.
3. The kernel reads A and B once and writes the float32 C once, so it moves
   N^2 x (2 x input bytes + 4) bytes. No kernel moves fewer bytes, so this
   gives the shortest memory time. A real kernel moves more.
4. The time is the larger of FLOPs over the compute ceiling and bytes over
   the bandwidth. A size is memory-bound when the memory time is larger.
5. The ratio divides each time by the bf16 time.

This is a model, not a measurement. The times are in microseconds.

| Device | N | fp32 us | tf32 us | bf16 us | Bounds fp32, tf32, bf16 | fp32 : tf32 : bf16 |
|---|---|---|---|---|---|---|
| A100 | 512 | 13.766 | 1.721 | 1.029 | compute, compute, memory | 13.38 : 1.67 : 1 |
| A100 | 1024 | 110.127 | 13.766 | 6.883 | compute, compute, compute | 16.00 : 2.00 : 1 |
| A100 | 2048 | 881.019 | 110.127 | 55.064 | compute, compute, compute | 16.00 : 2.00 : 1 |
| A100 | 4096 | 7048.151 | 881.019 | 440.509 | compute, compute, compute | 16.00 : 2.00 : 1 |
| A100 | 8192 | 56385.212 | 7048.151 | 3524.076 | compute, compute, compute | 16.00 : 2.00 : 1 |
| H100 | 512 | 4.006 | 0.939 | 0.626 | compute, memory, memory | 6.40 : 1.50 : 1 |
| H100 | 1024 | 32.052 | 4.343 | 2.504 | compute, compute, memory | 12.80 : 1.73 : 1 |
| H100 | 2048 | 256.416 | 34.742 | 17.362 | compute, compute, compute | 14.77 : 2.00 : 1 |
| H100 | 4096 | 2051.328 | 277.935 | 138.897 | compute, compute, compute | 14.77 : 2.00 : 1 |
| H100 | 8192 | 16410.621 | 2223.482 | 1111.179 | compute, compute, compute | 14.77 : 2.00 : 1 |

The fp16 path has the bf16 time in this model, because both have one peak and
one element size.

## Why the speedup ratio has these values

Per clock, the Tensor Cores of one SM do 16 times more bf16 FMAs than its
FMA lanes do FP32 FMAs. This is true on A100 and on H100. TF32 gives 8
times more. So a
compute-bound matmul has the time ratio fp32 : tf32 : bf16 = 16 : 2 : 1 when
all three paths run at one clock.

A100 shows that ratio from N = 1024. At 1410 MHz, its listed peaks reproduce
from the per-SM rates, so the peaks keep the per-clock ratio. At N = 512, the
bf16 time is a memory time. Its intensity is N / 4 = 128 FLOP per byte, below
the bf16 ridge of 153 FLOP per byte. Thus the ratio falls to 13.38 : 1.67 : 1.

H100 shows 14.77 : 2.00 : 1 at large N, not 16 : 2 : 1. With 132 SMs and the
per-SM rates, the FP32 peak of 67 TFLOPS implies about 1983 MHz. The Tensor
Core peaks imply about 1830 MHz. The model takes the listed peaks and chooses
no clock, so the clock difference stays in the ratio.

H100 is memory-bound at more sizes than A100. Its bf16 peak is 3.17 times
the A100 peak, but its bandwidth is only 1.64 times larger. So its bf16 ridge
moves from 153 to 295 FLOP per byte, and N = 512 and N = 1024 stay below it.

The model omits the kernel start overhead, caches, the clock under load, and
the tile count. At N = 512, the 128 x 128 tile gives only 16 programs for 108
or 132 SMs. So a real run at small N is far from these times, and a real ratio
can be smaller. Later, Ajay can measure a ratio on an NVIDIA GPU. Then the
model gives the first question: is the bf16 run compute-bound or memory-bound
at that N?

## CUTLASS reading notes

These are reading notes only. The lab runs no CUTLASS code, and Ajay must
still watch Lecture 15.

- CUTLASS splits a GEMM into a threadblock tile, a warp tile and an
  instruction tile. In this lab, `BLOCK_M x BLOCK_N x BLOCK_K` is the
  threadblock tile, `warpsPerCTA` places the warps, and `instrShape` is the
  instruction tile.
- CuTe, the layout library of CUTLASS 3, writes a layout as a shape and a
  stride. A TTGIR encoding states the same mapping from an element to a
  thread and a register.
- CuTe writes a swizzle as `Swizzle<B, M, S>`. It takes B bits of an offset
  that start S bits above its M low bits. It XORs them into the B bits just
  above the M low bits. With 16-bit elements, `Swizzle<3, 3, 3>` moves groups
  of 8 elements by a phase of 3 bits. That is the `vec = 8`, `maxPhase = 8`
  case of TTGIR on 64-element rows.
- CUTLASS names an MMA atom by its instruction, for example an SM80
  16 x 8 x 16 atom with bf16 inputs and float32 accumulation. That atom wraps
  the `mma.sync` form that this lab counts.

## The report and the verifier

`build_report` holds the pinned text, the declared config, its SHA-256 and
the model. With a backend, it also holds the rule row of each compile with
the digest and size of each text. `seal_report` adds a SHA-256 over the
canonical JSON. The seal detects drift. It is not a signature.

The label states where the compile section came from:

- `declared_model`: no compile, only the model of declared inputs.
- `triton_compiler_output`: triton compiled the text on this host.
- `synthetic_compile`: another backend gave the texts, for example the
  synthetic backend of the tests.

`ai_journey.triton_tc_verify` rebuilds every section with other code than
the lab's. It reads the config as raw JSON and recomputes the whole model in
exact fractions, with its own half-to-even rounding. It recounts the verdict
of each compile row from the counts that the row records, with its own rule
table and opcode grammar. It refuses a row with a missing or an unknown
field, and it rebuilds each MMA form from its opcode.

With a dump folder, the verifier also compares the dump index with the
report. The index must name the kernel digest, the config, the environment,
the warps and the stages of the report. The verifier checks the digest, the
line count and the size of each text. Then it rebuilds the whole PTX and
TTGIR record of each row from the texts. It uses its own expressions and its
own parser of the encoding parameters.

Without a dump folder, the verifier cannot read the texts. So only a run
with the dump finds a changed text field, for example a dot precision or a
buffer encoding. The tamper tests change one field of a sealed report, seal
it again, and expect the verifier to refuse it.

## The honest boundary

- This host has no NVIDIA GPU, no CUDA, no `nvcc` and no `ptxas`. No NVIDIA
  GPU ran the kernel, and nothing executed a PTX text or a cubin.
- The compiler output comes only from the Linux CI job `triton-tc-contracts`
  with triton 3.9.0. That job had not run when this guide got its text. So
  this guide states no instruction count of a Day 82 compile.
- The lab measured no NVIDIA time. The model times are model output from
  declared datasheet inputs.
- Each datasheet value is a declared input with a source. No value is a
  measurement, and the lab states no occupancy or register use.
- The synthetic PTX and TTGIR snippets are test inputs written by hand. They
  are not compiler output.
- Ajay must still watch Lecture 23 and Lecture 15. Ajay must still read the
  CI dump. Ajay must still read the log of the first run of the CI job
  `triton-tc-contracts`. Ajay must still measure the fp32 and bf16 ratio on
  an NVIDIA GPU when one is available.
