# Simulator profiling lab

Day 75 asks for a profiling loop: profile, name the top bottleneck, fix it,
and measure again. This lab applies that loop to the pure-Python CUDA launch
simulator of the [kernel launch lab](kernel-launch-lab.md). The
[matmul model lab](matmul-model-lab.md) uses the same simulator for its
naive matmul.

All numbers in this guide are host-specific CPU measurements. They come from
one Apple arm64 laptop with macOS and CPython 3.11.14. No GPU ran. The lab
compiled no CUDA source and used no NVIDIA GPU. A "hotspot" here is an
inference from profiler self time and wall-clock timings. No number comes
from hardware counters. The Day 74 ceilings are GPU ceilings, so they do not
apply to this CPU loop.

## Commands

```bash
python -m ai_journey.sim_profile_cli run \
  --config config/day-75-sim-profile.json \
  --stage round-4 --output /tmp/day-75-round-4.json
python -m ai_journey.sim_profile_cli rounds \
  --config config/day-75-sim-profile.json \
  --snapshot /tmp/day-75-baseline.json --snapshot /tmp/day-75-round-1.json \
  --snapshot /tmp/day-75-round-2.json --snapshot /tmp/day-75-round-3.json \
  --snapshot /tmp/day-75-round-4.json --output /tmp/day-75-rounds.json
python -m ai_journey.sim_profile_cli verify \
  --report /tmp/day-75-rounds.json --config config/day-75-sim-profile.json
```

The parser uses the program name `ai-journey-sim-profile`. The commands
import neither torch nor NumPy.

The `run` command profiles and times the current tree at one stage. A stage
is `baseline` or `round-N`, and the config must list round N. The command
writes a sealed snapshot. It prints the load averages, the ranked hotspots,
the median and IQR of each workload, and one summary line.

The `rounds` command reads one snapshot for each stage, in stage order. It
verifies each snapshot and then seals the comparison of each stage with the
stage before it. The `verify` command checks a snapshot or a rounds report.
With `--config`, it also binds the report to that config file.

The exit codes are:

- 0: the command succeeded.
- 1: the command read or built a report, but the report failed its checks.
  A tree whose misuse records changed also gives code 1.
- 2: a usage error, an unreadable or invalid input, or a refused output.

## Workloads

The pinned config names three workloads. Each one has a ragged extent, so
the guards of each kernel run:

| Workload | Extent | Block | Threads |
| --- | --- | --- | ---: |
| Day 73 vector add | 8000 elements | 256 | 8192 |
| Day 73 grayscale | 90 x 70 image | 16 x 16 | 7680 |
| Day 74 naive matmul, `x_columns` | m = 40, n = 45, k = 32 | 16 x 16 | 2304 |

The inputs come from the Day 73 seeded streams, with seed 75. The lab builds
the inputs before the profiler or the timer runs. Each pass runs the Day 73 or
Day 74 run helper without a change. Each output must have the SHA-256 digest
of the pure-Python reference. The reference code does not use the simulator.
Thus a faster simulator cannot change an output bit without a failure.

## Semantic guard

The pinned workloads have no races and no out-of-bounds accesses. Thus they
cannot show a lost record. The lab also runs four misuse cases, each in the
canonical order and in one seeded permutation:

- the unguarded vector add on 1000 elements
- the unguarded grayscale kernel on a 40 x 30 image
- a three-point stencil that writes its own input
- a guarded naive matmul with the `x_rows` mapping

The lab puts each record of the eight access logs into one canonical JSON
list. The lab took the SHA-256 of that list on the simulator before round 1.
The pinned digest is
`7b6d751094724ef06dca5adbeed4a4a0c38fbbf8a4b0a0cd7f497cc498456a24`.
Each snapshot repeats the cases and refuses a tree with another digest. The
logs hold 826 out-of-bounds reads, 308 out-of-bounds writes, 464 write
conflicts and 140 read-write conflicts. The pinned report digests of the Day
73 and Day 74 labs also stayed the same, and their test suites pass without
a change.

## Harness

The harness runs one checked pass of each workload under `cProfile`. It
reads the result with `pstats`. Each function gets a stable name of the form
`module:qualname`, for example
`ai_journey.kernel_launch_buffer:DeviceBuffer.__getitem__`. The qualname
comes from the syntax tree of the module, because Python 3.11 profiles keep
only the bare name and the first line.

The snapshot keeps these values for each function: the calls, the primitive
calls, the self time and the cumulative time. It also merges the three
workloads and ranks the functions by self time. Call counts are exact and
repeat from run to run.

The profiler adds a cost to each call. That cost makes small functions that
run often look larger. Thus the profiler times only rank the functions of
one profile. The speedups come from timed passes with no profiler.

## Timing

Each workload runs 2 untimed passes and then 31 timed passes. The timer is
`time.perf_counter_ns`, through the Day 74 `time_repeats` helper. Each pass
must give the reference digest, and that check runs after the clock stops.
The statistics are nearest-rank values with no interpolation. The median is
p50, and the IQR is p75 minus p25.

Each snapshot records the 1, 5 and 15 minute load averages before and after
it runs. This host has 11 logical CPUs. Its 1-minute load stayed between 9
and 19 during the lab, because other work ran at the same time. That load
makes a pure-Python loop slower and less regular. Thus small speedups are
inside the noise.

## The profiling loop

Each round used these steps:

1. Profile the current tree and read the ranked hotspots.
2. Name the top hotspot, and select one target from the ranked list.
3. Make one focused change to the simulator or its buffers, in one commit.
4. Run the semantic guard and the Day 73 and Day 74 test suites.
5. Take snapshots of the old and new trees in turn, three times, and compare
   the medians.

The baseline profile ranked these functions by self time:

| Rank | Function | Calls | Share of self time |
| --- | --- | ---: | ---: |
| 1 | `DeviceBuffer.__getitem__` | 150100 | 11.5% |
| 2 | `AccessRecorder._record_read` | 150100 | 11.4% |
| 3 | `to_float32` | 171320 | 10.9% |
| 4 | `_guarded_dot` | 2304 | 5.4% |
| 5 | `Index3.__post_init__` | 33878 | 5.2% |

The first two functions are almost equal. In one of the two sealed sessions,
`_record_read` ranked first.

### Round 1: buffer reads

The top hotspot was `DeviceBuffer.__getitem__`, with `_record_read` next to
it. Each read called three helpers: `_recorder_for`, `_record_read` and
`_running`. The change records the read inside `__getitem__` itself. The
checks and the records stay the same, in the same order. The profiled calls
fell from 2785398 to 2335098.

### Round 2: float32 rounding

After round 1, `__getitem__` stayed first. Its remaining self time is the
read record itself, which the race detection needs. A trial change cut one
dictionary lookup from each read, but its gain stayed inside the noise. The
lab did not keep it. The second hotspot was
`to_float32`. It called the module-level `struct.pack` and `struct.unpack`,
which find the format again on each call. The change rounds through one
precompiled `struct.Struct("<f")`. The call counts stay the same.

### Round 3: thread references

After round 2, ranks 1 to 3 were the reads, the float32 rounding and
`_guarded_dot`. The last one is the loop of the matmul kernel itself, which
is the work under simulation. The next hotspot was `Index3.__post_init__`.
Before round 3, `AccessRecorder.finish` built two `Index3` values for each
written element, also when no race used them. The change builds a
`ThreadRef` only for an element in a write or read-write conflict. The
`Index3` calls fell from 33878 to 1678. The profiled calls fell from 2335098
to 1803798.

### Round 4: float32 store checks

After round 3, the next hotspot below the reads, the rounding and the kernel
loop was `_check_value`. Each float32 store and each input element calls it.
It called `to_float32`, which checked the type and the finiteness a second
time. The change tests the exact value with one `Struct` pack and unpack.
The `to_float32` calls fell from 171320 to 123200. The profiled calls fell
from 1803798 to 1707558.

## Measured speedups

The table gives the speedup of the sum of the three medians in each round.
The paired columns come from three alternating runs of the old and the new
tree. The sealed columns come from two later sessions that ran all five
stages one after the other.

| Round | Paired runs | Sealed session 1 | Sealed session 2 |
| --- | --- | ---: | ---: |
| 1 | 1.079, 1.087 (two pairs) | 1.223 | 1.079 |
| 2 | 1.060, 1.097, 0.992 | 1.205 | 1.082 |
| 3 | 1.402, 1.311, 1.347 | 1.291 | 1.334 |
| 4 | 1.329, 1.028, 1.224 | 1.093 | 0.967 |
| All | | 2.079 | 1.508 |

Sealed session 2 measured these medians, with the IQR in parentheses:

| Workload | Baseline | After round 4 |
| --- | ---: | ---: |
| vector add | 43.5 ms (2.7) | 25.4 ms (5.1) |
| grayscale | 35.1 ms (4.1) | 23.6 ms (10.5) |
| naive matmul | 56.5 ms (14.8) | 40.6 ms (4.0) |
| sum of medians | 135.0 ms | 89.6 ms |

The 1-minute load was 17.2 to 18.4 in session 2 and 14.7 to 17.4 in
session 1.

The session 1 baseline median, 174.8 ms, is the highest of the lab. The
other baseline runs gave medians of 122 to 135 ms in total. Thus the overall speedup of session 1 probably
includes host noise. The rounds report digests are:

- session 1: `2f5e1e4dc739784bf6421ee4136a86a29760ce683e6e24bad991c404165430a9`
- session 2: `d458263e3239c23b565e7febcc084ca6f2cb3b3984e7037188c2765657e4cb22`

The reports are not in the repository, because each one holds raw timings
of one host at one time. The `run` and `rounds` commands make new reports.

These results follow from the measurements:

- Round 3 gave the only large and repeatable gain: 1.29 to 1.40 in each run.
  In session 2 it cut the vector add and grayscale medians by 34% to 41%.
- Rounds 1 and 2 helped the naive matmul most, because it makes 115200 reads
  and 115200 roundings. The total gain was about 1.06 to 1.10 in most runs.
- The gain of round 4 is smaller than the noise at this load. One sealed
  session shows 0.967.
- The rank of a function did not predict its gain. Round 3 targeted rank 4
  and gave the most. The profiler cost on each call inflates small functions
  that run often.

## Report and verification

A snapshot holds these items:

- the config without its rounds, and the stage
- the host environment, the clock and the load averages
- the result of the semantic guard
- every profiled function of each workload, and the merged profile
- the ranked hotspots
- the raw timed samples with their statistics, and a summary
 The environment
names the Python version, the platform and the machine type only. It names
no host and no user.

A rounds report holds the full config, one snapshot for each stage, and one
entry for each round. A round entry names the top hotspot before and after,
the target with its rank and calls, and the medians, IQRs and speedups. Each
speedup is the before median divided by the after median, to 3 decimals.

The verifiers do these checks:

- They check each field set, the labels, the evidence boundary and the
  SHA-256 seal.
- They rebuild the launch geometry and the reference digests from the
  embedded config, without the simulator.
- They redo the profile sums, the hotspot ranking, the shares, the
  statistics and each speedup with their own code.
- They require the same clock and environment in all snapshots of a rounds
  report.

The verifiers check the timings for structure and arithmetic only. They
cannot prove that a host made a timing. The seal detects changes, and it is
not a signature.

## Boundaries

The timings are wall-clock CPU measurements of one host. They depend on
that host, its Python version and its load. They do not predict the speed of
a CUDA kernel, an NVIDIA GPU, or a production or distributed system. A
report from an injected clock says `synthetic_timing` and measures no host.
