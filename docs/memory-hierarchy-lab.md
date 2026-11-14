# CPU memory-hierarchy measurement lab

Day 71 of the curriculum, "How GPU Computing Works", argues that a GPU is not
a fast CPU: it hides memory latency with massive parallelism, and memory
bandwidth, not arithmetic, is usually the constraint. The SIMT and
memory-access labs model those ideas analytically. This lab measures the
underlying effects on the CPU of the host that runs it: how bandwidth falls
as the working set outgrows the caches, how much the order of reads costs,
where a measured roofline puts streaming kernels and a matmul, and what a
stated latency implies is in flight. Every number is a wall-clock
measurement of NumPy kernels on that CPU host. Nothing here runs on,
measures, or predicts a GPU.

## Commands

```bash
ai-journey-memory-hierarchy run \
  --config config/day-71-memory-hierarchy.json \
  --output /tmp/day-71-memory-hierarchy.json
ai-journey-memory-hierarchy verify \
  --report /tmp/day-71-memory-hierarchy.json \
  --config config/day-71-memory-hierarchy.json
```

`python -m ai_journey.memory_hierarchy_cli` accepts the same arguments.
`run` measures, verifies its own report, and publishes it; it prints one
line per kernel curve, access pattern and the matmul, labeled
`cpu_host_measurement`, then `derived` roofline and Little's-law lines and a
summary line with the report digest. `verify` recomputes a report and prints
the same summary prefixed with `verified_`. `--config` on `verify` is
optional; with it, the report must have been measured from that exact
config.

Exit code 0 means success, 1 a readable report that failed verification, and
2 usage errors, unreadable or invalid inputs, refused outputs, and runs that
failed a kernel check or exceeded their time budget. On the development host
the pinned run takes about 5 seconds, peaks at 208 MiB of lab arrays, and
writes a report of about 46 KB.

Configs are read from regular non-symlink files of at most 64 KiB, reports of
at most 1 MiB. Duplicate JSON keys, `NaN` and `Infinity`, unknown or missing
fields, and booleans or floats posing as integers are rejected. Reports are
canonical JSON (sorted keys, compact separators, ASCII escapes) with a
SHA-256 digest over every other field. The digest detects drift and
tampering; it is not a signature and cannot prove that a timing was
observed. The writer verifies a report before publishing it through a
fsynced temporary file and `os.replace`, and refuses an output that aliases
the config through a resolved path, symlink, or hardlink.

## Controls

| Field | Pinned | Allowed |
| --- | --- | --- |
| `dtype` | `float64` | `float32` or `float64`, shared by every section |
| `seed` | 71 | 0 to 2^32 - 1 |
| `timing.warmup` | 2 | 0 to 10 untimed batches before each measurement |
| `timing.repeats` | 11 | 3 to 50 timed samples per measurement |
| `timing.target_bytes_per_sample` | 33554432 | 1 KiB to 1 GiB of accounted bytes, at most 65536 calls per sample |
| `timing.max_runtime_seconds` | 20 | 1 to 300 |
| `stream.kernels` | copy, scale, add, triad | STREAM order; triad is required |
| `stream.elements` | 12 points, 1024 to 8388608 | 1 to 16 strictly increasing points of 64 to 2^24 elements per array |
| `access.elements` | 8388608 | 2^10 to 2^24, a multiple of the stride with at least two reads per pass |
| `access.stride_elements` | 16 | 2 to 4096 |
| `matmul.m`, `n`, `k` | 1024 | 16 to 2048 |
| `littles_law.latency_ns` | 100 | 1 to 100000; a stated figure, never measured |
| `littles_law.latency_note` | where the figure comes from | single-spaced ASCII, at most 400 characters |

Before any array is allocated, the whole config must fit a 256 MiB peak
allocation (the largest section's arrays plus check temporaries, or for the
access section the strided order's pass offsets when a small stride makes
them larger), and the run must fit 2^38 accounted bytes, 2^22 kernel calls
and 2^38 FLOPs, counted with the batch sizes the run will use. The report
records these counts as `planned_work`. The time budget is read from the
run's clock after every warmup and timed batch and once at the end, never
inside a timed region, so a run stops at the first check after its budget
runs out.

## STREAM kernels and byte accounting

The kernels follow McCalpin's STREAM with `q = 3`. Each is a fixed list of
NumPy calls writing through an output operand, so no call allocates a
temporary array (a tracemalloc test pins this) and the bytes per element
follow from the operands. Bytes are counted the STREAM way: one element per
array read and one per array written.

| Kernel | NumPy calls | Reads | Writes | FLOPs | float64 bytes per element |
| --- | --- | --- | --- | --- | --- |
| copy | `numpy.copyto(c, a)` | 1 | 1 | 0 | 16 |
| scale | `numpy.multiply(q, a, out=c)` | 1 | 1 | 1 | 16 |
| add | `numpy.add(a, b, out=c)` | 2 | 1 | 1 | 24 |
| triad | `numpy.multiply(q, b, out=c)`, then `numpy.add(c, a, out=c)` | 3 | 2 | 2 | 40 |

NumPy has no fused multiply-add ufunc, so triad makes two passes and moves
five elements per element where STREAM's single-loop triad counts three. The
accounting is operand traffic only: write-allocate reads, cache hits, and
prefetching are not observed, so the memory system can move more or fewer
bytes than stated. Before each measurement the output is filled with NaN;
after the clock stops it must equal STREAM's definition (`c = a`, `c = q a`,
`c = a + b`, `c = a + q b`) exactly, or the run fails.

## Working-set sweep

One seeded allocation sized for the largest point serves the sweep, and each
point binds the kernels to the first `elements` entries of `a`, `b` and `c`.
A kernel's working set is the bytes of the distinct arrays it touches: two
arrays for copy and scale, three for add and triad. Each sample repeats a
call until it moves at least `target_bytes_per_sample` accounted bytes, so a
cache-sized call of about a microsecond is timed over a long enough interval.
The smallest points are still dominated by NumPy's per-call overhead
(`median_ns_per_call` shows it; triad at 1024 elements takes about 750 ns
per call on the development host, for two ufunc calls), so their bandwidth
understates what the cache delivers. The report gives one curve per kernel
and, for each, the peak point and the largest point. It names no cache
level and detects no cache size: where the curve falls is read from the data.

## Access-pattern contrast

Four patterns read the same 64 MiB float64 source into the same output:

- `copy`: `numpy.copyto(output, source)`, the streaming reference;
- `sequential`: `source.take(order, out=output, mode='clip')` with order
  0, 1, 2, ...;
- `strided`: the same call visiting elements j, j + S, j + 2S, ... for j = 0
  .. S - 1, with S = 16 elements (128 bytes);
- `random`: the same call with `numpy.random.default_rng((seed, 1))
  .permutation(n)`.

Every gather reads each source element exactly once through an 8-byte index,
so the three gathers move identical accounted bytes (24 per element for
float64) and differ only in the order of their source reads. A slower gather
pays for its order alone: the stride gives up spatial locality, and the
permutation also gives up predictability, so its misses overlap only as far
as the core keeps independent loads in flight. `take` runs with
`mode="clip"` because with the default `"raise"` and an `out` argument NumPy
buffers the whole output in a temporary copy; every index is in range, so
clipping never changes a value. Each order is built just before its
measurement and checked against fancy indexing after it. The report adds
time-per-element ratios between the patterns.

## Measured roofline

The compute ceiling is the matmul's median FLOP rate, with FLOPs counted as
2 M N K through `roofline.matmul_workload`. The placement bandwidth ceiling
is triad's median accounted bandwidth at the largest sweep point; a second
ceiling records triad at the sweep's peak point, giving the ridge an
in-cache working set would see. `roofline.RooflineHardware` computes each
ridge point and `roofline.analyze_roofline` places scale, add and triad at
the largest point, and the matmul, at their arithmetic intensities, using
medians on both axes. Triad and the matmul sit on the ceilings they define,
so their `fraction_of_attainable` is 1 by construction, up to floating-point
rounding (triad's can read 1.0000000000000002); another kernel can exceed 1
when it streams faster than triad did. Copy and the gathers perform no FLOPs
and have no roofline position.

The matmul runs on NumPy's BLAS, which may use several threads, while the
STREAM kernels and gathers run on one; the ridge describes these NumPy
workloads on this host, not the processor's peak. Every element of the final
product is checked against a float64 reference within
(gamma_K(u) + gamma_K(2^-53)) (|A| |B|), the classical inner-product rounding
bound, where u is the dtype's unit roundoff. For float64 the reference uses
the same BLAS, so the recorded `max_bound_ratio` is usually exactly 0; the
check still rejects a non-finite or unwritten product and any element off by
more than the bound, though not a smaller error (with the lab's
standard-normal operands the float32 bound at K = 1024 is about 0.04).

## Little's law

Little's law says requests in flight equal their completion rate times how
long each takes. The report multiplies triad's median bandwidth at the
largest working set, and each access pattern's median element rate, by the
config's stated latency. Fields keep three kinds of number apart:
`measured_` rates come from this host's samples, `stated_latency_ns` is the
config input, and `derived_` values are their product, an implication rather
than an observation. The section records `latency_measured: false`; nothing
counts the requests that were actually outstanding. The pinned 100 ns is a
round planning figure for a load that misses every cache level, not a
datasheet value for any host.

## Statistics

Warmup batches run but are never timed. Each timed sample reads an integer
nanosecond clock (`time.perf_counter_ns`) before and after one batch, and
output checks run after the clock stops. Percentiles use the nearest-rank
method of `memory_access_benchmark`: sorted samples[ceil(p n / 100) - 1] for
p in 25, 50 and 75, median = p50 and IQR = p75 - p25, with no interpolation.
Each measurement keeps its raw samples, min, quartiles, median, max and IQR,
and its median and best (min-time) rates. Ceilings, placements and Little's
law use medians.

In Python, `main`, `measure_report` and every measuring function accept an
injected clock so tests can script durations and drive the budget. A report
timed that way records timer `injected`, `measured: false`, scope
`synthetic_injected_clock`, and a boundary with
`cpu_measurement_this_host: false`, and its lines print as
`synthetic_timing`. Its kernels and checks still ran. `verify` accepts it
only as synthetic timing and rejects any report whose timer, `measured`
flag, scope and boundary disagree.

## Verification

`verify` never calls the measuring modules to recompute a number; the config
parser it shares only validates bounds. It parses the embedded config
strictly and checks its digest, rebuilds the list of measurements, each
kernel's and pattern's byte and FLOP accounting, every batch size and the
planned peak allocation from its own tables, and recomputes from the raw
samples the statistics and rates, the curves and their summaries, the access
ratios, both ceilings, the ridge points and placements, the Little's-law
products, the planned byte, call and FLOP totals, the run totals (timed
nanoseconds must fit inside the elapsed run, which must fit the budget) and
the summary. Each section must match as canonical JSON, so `1`, `1.0` and
`true` stay distinct. It cannot re-observe a timing.

## One run on the development host

Measured on 2026-11-14 with the pinned config on a shared Darwin arm64 host
with 11 logical CPUs, CPython 3.11.14 and NumPy 1.26.4 built against OpenBLAS
0.3.23.dev, while other jobs were running. These are one run's medians from
one host, not portable figures; rerun the command to measure yours.

- Triad: 116 GB/s at a 96 KiB working set, 120 to 126 GB/s from 384 KiB to
  36 MiB, 105 GB/s at 48 MiB, 102 GB/s at 96 MiB and 80 GB/s at 192 MiB.
  Copy peaked at 178 GB/s at 256 KiB and ran at 105 GB/s at 128 MiB.
- Access patterns over the 64 MiB source: copy 0.165 ns per element,
  sequential take 1.87 ns, strided take 6.68 ns, random take 11.13 ns. The
  random gather took 5.96 times as long as the sequential one and 1.67 times
  as long as the strided one for the same bytes.
- The 1024^3 float64 matmul ran at 129 GFLOP/s, giving a ridge of 1.61 FLOPs
  per byte against the 192 MiB triad and 1.03 against the sweep's peak
  triad. Scale, add and triad sit near 0.04 to 0.06 FLOPs per byte, far left
  of the ridge; the matmul's compulsory intensity is 85.3.
- At the stated 100 ns, triad's bandwidth implies about 8,000 bytes in
  flight, and the random gather's rate about 9 elements in flight (15 for
  strided, 54 for sequential).

## Evidence boundary and exclusions

The report's boundary states: "Wall-clock timings of NumPy kernels on the CPU
host that built this report. Bandwidths count the operand bytes each NumPy
call reads and writes, not hardware traffic, and include NumPy call
overhead. The matmul runs on NumPy's BLAS, which may use several threads;
the other kernels use one. Little's-law rows multiply measured rates by a
stated latency that was not measured. No cache size is detected or claimed,
and nothing here measures or predicts GPU, CUDA, distributed, or production
performance."

The lab reads no hardware counters and does not observe cache or line sizes,
TLB behavior, prefetchers, write-allocate traffic, clock frequency or
thermal state, thread placement, or the BLAS thread count. Results vary
with load and builds. The report is CPU evidence from one host only and is
not evidence of GPU performance, lecture viewing, completed exercises, or
learner understanding.

## Sources

- J. D. McCalpin, "Memory Bandwidth and Machine Balance in Current High
  Performance Computers", IEEE Computer Society Technical Committee on
  Computer Architecture Newsletter, December 1995; STREAM kernels and byte
  counting: https://www.cs.virginia.edu/stream/
- S. Williams, A. Waterman, and D. Patterson, "Roofline: An Insightful Visual
  Performance Model for Multicore Architectures", Communications of the ACM
  52(4), 2009.
- J. D. C. Little, "A Proof for the Queuing Formula: L = lambda W",
  Operations Research 9(3), 1961.
- N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
  SIAM, 2002, section 3.1 (inner-product rounding bound).
- NumPy reference, `numpy.take`: "out is always buffered if mode='raise'":
  https://numpy.org/doc/stable/reference/generated/numpy.take.html
