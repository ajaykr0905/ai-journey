# What Triton writes for the Day 79 matmul

Day 80 of the curriculum follows GPU MODE Lecture 29, "Triton Internals", and
Lecture 28, "Liger Kernel: Efficient Triton Kernels for LLM Training". The
tracker note is "Triton's compiler pipeline: Python AST to MLIR to PTX.
Understand what it does for you." The hands-on task is "Dump the PTX for your
Triton kernel and read it." This lab is the Triton internals half of Day 80.

The lab does these steps:

1. It models the recorded stage dump of the Day 79 kernel. The dump has 6
   configs and 2 declared NVIDIA targets, with 4 texts and 1 cubin each.
2. It checks the 3 committed texts and the kernel source against their
   SHA-256 digests.
3. It reads the PTX, the TTIR and the TTGIR with no triton import.
4. It maps each line of the kernel to the ops and instructions that carry it.
5. It compares the 12 compiles from the manifest.
6. It seals one report, and an independent verifier checks the report.

Nothing here ran on an NVIDIA GPU. The texts are compiler output for declared
targets. No GPU executed a PTX text or a cubin. No `ptxas` register report exists on
this host, so the lab states no register use and no occupancy.

Ajay must still watch Lecture 29 and Lecture 28. Ajay must still read the
committed PTX himself. This guide shows where to start. It does not replace
the reading. This lab does not cover the Liger kernels of Lecture 28.

## The recorded input

The session owner compiled `ai_journey.triton_matmul_kernel.TRITON_SOURCE`
with triton 3.9.0. The run used a Linux aarch64 `python:3.11-slim` container,
Python 3.11.18 and `TRITON_DEFAULT_FP_FUSION=1`. The dump script compiled each
of the 6 `AUTOTUNE_CONFIGS` for `sm_80` and for `sm_90`. It wrote these files
for each compile:

- `ttir.mlir`: the Triton IR.
- `ttgir.mlir`: the Triton GPU IR.
- `llir.ll`: the LLVM IR.
- `ptx.ptx`: the PTX.

It also wrote `index.json` with each digest and line count, the shared memory
bytes, the warps, the stages and the cubin digest and size.

The folder `tests/fixtures/day80_triton_ir/` holds `index.json` and 3 texts:
the TTIR, TTGIR and PTX of `sm_80` with the config `m32_n64_k32_g8_w2_s5`.
This is the smallest config. The other 11 compiles and all LLVM IR stay
digest only.

The manifest records the source digest
`9eb35e273b0fd229e83e223861ef7ce13c4f15ea46d2694baf13920851a31d56`. This is
the digest of the Day 79 kernel text at main. The digest of `index.json` is
`95c187abaae39edda3262eec53588855360b4d9b933f7cbe4e3536b3696b6473`.

## Commands

```bash
PYTHONPATH=src python -m ai_journey.triton_ir_cli summarize \
  --fixtures tests/fixtures/day80_triton_ir \
  --output /tmp/day-80-triton-ir.json
PYTHONPATH=src python -m ai_journey.triton_ir_cli verify \
  --report /tmp/day-80-triton-ir.json \
  --fixtures tests/fixtures/day80_triton_ir
PYTHONPATH=src python -m ai_journey.triton_ir_cli stage-map \
  --fixtures tests/fixtures/day80_triton_ir
PYTHONPATH=src python -m ai_journey.triton_ir_cli check \
  --fixtures tests/fixtures/day80_triton_ir \
  --output /tmp/day-80-triton-ir-check.json
```

- `summarize` checks the fixtures and prints the counts of the committed
  texts and 1 line for each of the 12 compiles. With `--output`, it writes
  the sealed report. It needs no triton.
- `verify` recomputes a sealed report from the fixtures. It needs no triton.
- `stage-map` prints 1 line for each kernel statement.
- `check` compiles every config again with triton and compares each digest.
  It writes a sealed report.

The parser uses the program name `ai-journey-triton-ir`. An installed wheel
gives the same commands under that name. The fixtures stay in `tests/`, so
each command needs `--fixtures` with the path of a checkout.

The exit codes are the codes of the Day 73 to 79 labs:

- 0: the command succeeded.
- 1: a report failed verification, or a regenerated text or metadata record
  differs from the dump. `check` writes its report in this case, so Ajay can
  read the difference.
- 2: a usage error, an unreadable or invalid input, or a refused output.
- 3: this host cannot compile with triton. Then `check` writes nothing.

Triton has no macOS wheel. On the development host, `check` exits 3 before
it makes the output folder. The `check` tests that need triton skip on macOS.

The texts also name the install path of triton. The TTIR, the TTGIR and the
PTX locations of the inlined `standard.py` helper hold the path
`/usr/local/lib/python3.11/site-packages/triton/language/standard.py`. A
triton at another path, for example in a virtual environment, gives other
digests for every stage. Then `check` exits 1, although the compiler output
can be the same in all other bytes.

## The pipeline

Triton lowers one `@triton.jit` function in these steps:

1. Python AST to TTIR. TTIR is an MLIR module of `tt.*` ops on whole tiles.
   It has no GPU layout. One `tl.load` gives one `tt.load`, and one `tl.dot`
   gives one `tt.dot`.
2. TTIR to TTGIR. Triton gives each tensor a layout encoding. It also
   pipelines the K loop into asynchronous copies to shared memory.
3. TTGIR to LLVM IR. Each tile op becomes scalar code for one thread.
4. LLVM IR to PTX. The LLVM NVPTX back end writes the PTX. The first lines of
   the committed PTX name it.
5. PTX to cubin. `ptxas` assembles the machine code for one GPU. The lab has
   only the cubin digest and size.

Each op keeps the location of its kernel line. TTIR and TTGIR use
`loc(#locN)`. PTX uses `.loc` lines. The file name of each location holds the
source digest, so a location names one exact kernel text.

## The counts of the committed texts

The TTIR holds 97 ops. It has 2 `tt.load` ops, and each load has a mask and
an `other` value. It has 1 `tt.dot`, 1 `scf.for`, 1 masked `tt.store`, 19
`tt.splat` ops and 2 `tt.make_range` ops, for 0 to 32 and 0 to 64.

The TTGIR holds 220 ops. Its module names `ttg.num-warps = 2`,
`ttg.threads-per-warp = 32` and `ttg.target = "cuda:80"`. It defines these
encodings:

- 3 `#blocked` layouts. `#blocked` has `sizePerThread = [4, 4]`,
  `threadsPerWarp = [2, 16]` and `warpsPerCTA = [2, 1]`.
- 1 `#shared` layout, of the kind `ttg.swizzled_shared`.
- 1 `#smem` memory space.

The TTGIR has no `#mma` encoding. Both `tt.dot` operands use `#ttg.dot_op`
with the parent `#blocked`. The TTGIR has 0 `tt.load` ops. It has 10
`ttg.async_copy_global_to_local` ops, 10 `ttg.async_commit_group` ops and 2
`ttg.async_wait` ops, with `num = 6` and `num = 0`.

The PTX starts with `.version 8.8`, `.target sm_80` and `.address_size 64`.
The entry `matmul_kernel` has 14 parameters: 3 matrix pointers, 9 int32
values and 2 more pointers. The kernel text declares only the first 12, so
Triton adds the last 2. `.reqntid 64` asks for 64 threads: 2 warps of 32.

The PTX declares `%p<251>`, `%r<2296>` and `%rd<480>`. These are virtual
registers. Only `ptxas` decides the real register use, and no `ptxas` report
exists here. The PTX declares one `.extern .shared` array, `global_smem`,
with no size. The compile metadata gives the size: 49152 bytes.

The entry holds 3097 instructions. These are the main families:

| Family | Count |
| --- | ---: |
| `fma.rn.f32` | 1024 |
| `cp.async` copies | 240 |
| `cp.async.commit_group` | 10 |
| `cp.async.wait_group` | 2 |
| `ld.shared` | 96 |
| `st.shared` | 8 |
| `ldmatrix` | 8 |
| `bar.sync` | 18 |
| `st.global` | 32 |
| `ld.global` | 0 |
| `mma` | 0 |
| `setp` | 218 |
| `selp` | 377 |

Thus the PTX holds 252 instructions with the prefix `cp.async`. It loads no
global memory with `ld.global`: every load of A and B is a `cp.async` copy.

## The stage map

`stage-map` counts, for each kernel statement, the ops and the PTX families
that name its line. These rows come from the committed texts:

| Line | Statement | TTIR | TTGIR | PTX in the whole entry |
| ---: | --- | --- | --- | --- |
| 36 | `for k0 in range(...)` | `scf.for` | `scf.for`, `ttg.async_wait` | loop control, 3 branches |
| 38 | `a_mask = ...` | 2 `arith.cmpi` | 6 `arith.cmpi` | 21 `setp` |
| 39 | `b_mask = ...` | 2 `arith.cmpi` | 6 `arith.cmpi` | 64 `setp` |
| 40 | `a = tl.load(...)` | 1 `tt.load` | 5 async copies | 80 `cp.async`, 64 `ld.shared` |
| 41 | `b = tl.load(...)` | 1 `tt.load` | 5 async copies | 160 `cp.async`, 32 `ld.shared` |
| 42 | `acc = tl.dot(...)` | 1 `tt.dot` | 1 `tt.dot` | 1024 `fma.rn.f32` |
| 46 | `c_mask = ...` | 2 `arith.cmpi` | 1 `arith.cmpi` | 32 `setp` |
| 47 | `tl.store(...)` | 1 `tt.store` | `tt.store`, `ttg.convert_layout` | 32 `st.global`, 8 `st.shared`, 8 `ldmatrix` |

The TTGIR has 5 copies for each load. Four copies are in the prologue before
the loop, and one copy is in the loop. The TTIR has 5 ops with no kernel
line, the TTGIR has 16 and the PTX has 38.

The store needs a layout change. The accumulator has the `#blocked` layout,
and the store uses another `#blocked` layout. Triton moves the tile through
shared memory with `st.shared` and `ldmatrix`. Here `ldmatrix` is a
warp-wide shared memory load, not Tensor Core math.

## Read one K loop iteration of the PTX

The K loop starts at the label `$L__BB0_3` on line 2216 of `ptx.ptx`. It ends
with the branch back on line 3869. It holds 1450 instructions. Read it in this
order:

1. Line 2217 sets a predicate that tells if a later tile is still inside K.
   Lines 2218 to 2220 compute the next ring slot. The slot counts up and
   returns to 0 after 3, because the ring has 4 slots.
2. Line 2222 is `cp.async.wait_group 6`. It waits until at most 6 copy groups
   are in flight. Then the oldest stage is complete in shared memory.
3. Line 2223 is `bar.sync 0`. All 64 threads wait here, so no thread reads a
   slot before all copies to it are visible.
4. Lines 2226 to 2324 hold the 96 `ld.shared.v4.b32` loads. Each load moves 4
   floats of A or B from the slot into registers.
5. Lines 2326 to 3349 hold the 1024 `fma.rn.f32` instructions. This is one
   thread's part of one 32 x 64 x 32 tile product: 32 x 64 x 32 / 64 = 1024.
6. Line 3441 is a second `bar.sync 0`. Then no thread writes a slot while
   another thread still reads it.
7. Lines 3449 to 3859 hold 48 `cp.async.ca.shared.global` copies of 4 bytes.
   They fetch the tile of the iteration 4 steps ahead. A `selp.b32` before
   each copy picks a source size of 4 or 0. A source size of 0 writes zeros,
   so the mask becomes a zero fill.
8. Two `cp.async.commit_group` instructions close the A group and the B
   group.
9. Lines 3863 to 3869 add 32 to `k0`, move the A and B pointers, compare and
   branch back.

The loop body has no `ld.global` and no `mma`. One pass holds 48 copies,
which equals 48 tile floats for each thread: (32 x 32 + 32 x 64) / 64.

## What Triton does for you

The kernel text names tiles, masks and one `tl.dot`. The compiler output
shows the work that Triton did from that text:

- Masking. Each mask became `setp` compares and `selp` selects. The load
  masks became the source size operand of `cp.async`. The store mask became a
  predicate on each `st.global`.
- Pipelining. `num_stages = 5` gave a ring of 4 shared buffers for each
  operand, a prologue of 8 commit groups and a wait that keeps 3 stages in
  flight. The kernel text holds no shared memory and no wait.
- Layouts. Triton chose which thread holds which elements. With
  `#blocked`, each of the 64 threads holds 32 of the 2048 accumulator values.
  Triton also inserted the layout change for the store.
- Register allocation. The kernel text names no register. The PTX names
  virtual registers. `ptxas` maps them to machine registers, and this lab has
  no report of that step.

## Why there is no mma

The Day 79 kernel calls `tl.dot` with `input_precision="ieee"`. That asks for
float32 products, and not for TF32 Tensor Core math. Thus Triton took the
FMA path. The TTGIR has no `#mma` encoding, the dot operands have the parent
`#blocked`, and the PTX has 0 `mma` instructions and 1024 `fma.rn.f32`
instructions. The derived count fits the tile: 32 x 64 x 32 multiply-adds over
64 threads give 1024 for each thread.

## num_stages and the cp.async ring

The manifest shows the ring rule for all 12 compiles. The shared memory bytes
equal `(num_stages - 1)` times the bytes of one A tile and one B tile:

| Config | num_stages | Tile bytes | Shared bytes | Buffers |
| --- | ---: | ---: | ---: | ---: |
| `m128_n128_k32_g8_w4_s4` | 4 | 32768 | 98304 | 3 |
| `m128_n64_k32_g8_w4_s4` | 4 | 24576 | 73728 | 3 |
| `m64_n128_k32_g8_w4_s4` | 4 | 24576 | 73728 | 3 |
| `m128_n32_k32_g8_w4_s4` | 4 | 20480 | 61440 | 3 |
| `m64_n32_k32_g8_w2_s5` | 5 | 12288 | 49152 | 4 |
| `m32_n64_k32_g8_w2_s5` | 5 | 12288 | 49152 | 4 |

The committed TTGIR shows the rule for its config. It allocates
`4x32x32` floats for A and `4x32x64` floats for B. The prologue copies the
tiles at K offsets 0, 32, 64 and 96. The loop then copies the tile 128
elements ahead. Only the committed config has its TTGIR in the repository, so
the lab derives the ring of the other configs from the shared bytes.

## sm_80 and sm_90

The manifest shows these differences for each of the 6 configs:

- The TTIR is identical for both targets. TTIR has no target.
- The TTGIR, LLVM IR, PTX and cubin differ.
- The shared memory bytes are equal.
- The `sm_90` PTX has 101 to 389 more lines.
- The `sm_90` LLVM IR has 1363 to 4939 fewer lines.

For the committed config, the `sm_90` TTGIR differs only in its target. The
digest of the committed `sm_80` TTGIR, with `"cuda:80"` changed to
`"cuda:90"`, equals the recorded `sm_90` digest. The repository cannot show
why the PTX and LLVM IR differ, because those texts are digest only.

These numbers need the full dumps:

- the instruction families, registers and loops of the 11 PTX texts that are
  not committed,
- the layouts, rings and waits of the 11 TTGIR texts that are not committed,
- every LLVM IR text,
- the register use and spills of any cubin, which only a `ptxas` report
  gives.

## Determinism

The session owner compiled every config a second time in the same container.
The owner reports these results:

- The TTIR, TTGIR, LLVM IR and PTX were byte-identical for every config and
  both targets.
- The `sm_80` cubins differed between the two runs.
- The `sm_90` cubins were equal.

The repository holds one run only, so the lab cannot check these results
itself. For this reason `check` requires each text and each metadata record
to match, and it records each cubin match without a rule. A Linux host can
run `check` and test the text result again. It needs Python 3.11 and the
triton 3.9.0 wheel at the install path of the dump.

The CI job `triton-ir-contracts` does this test on an x86_64 Linux runner. It
uses the `python:3.11-slim` image of the dump, so triton has the same install
path. The installed `check` compares the x86_64 texts with the aarch64 dump,
and the job prints each digest pair as data. Then the tests with the real
triton run with no skip. They require all 48 texts and all 12 metadata
records to match. Thus a difference between the two architectures fails the
job on purpose. No result of this job exists yet.

## The report and the verifier

The report holds the manifest, the source digest and the committed files.
It also holds the 3 stage summaries, the stage map, the pipeline facts, the
comparison and an optional regeneration. A SHA-256 seal covers all other fields. The seal
detects drift. It is not a signature.

The label names the source of the texts:

- `recorded_compiler_output`: the host read the recorded texts only.
- `triton_regeneration`: the triton compiler on this host compiled the text
  again. This report holds the host clock and load averages only.
- `synthetic_regeneration`: a stand-in backend gave the texts. Tests use it.

The verifier rebuilds each section with other code than the lab. It parses
the raw `index.json` with plain `json`. It counts the PTX with whole-text
patterns, and it resolves the MLIR locations with a fixpoint pass. It also
checks the manifest digest against a digest that it pins itself. The tamper
tests change one field, seal the report again and expect a rejection.

## The honest boundary

- The texts are compiler output for the declared targets `sm_80` and
  `sm_90`. Nothing ran on an NVIDIA GPU.
- No GPU executed a PTX text or a cubin. No number here is a time or a rate.
- No `ptxas` register report exists on this host. The `.reg` counts are
  virtual registers.
- The determinism results come from the session owner. The repository holds
  one run.
- Ajay must still watch Lecture 29 and Lecture 28, and Ajay must still read
  the committed PTX himself.
