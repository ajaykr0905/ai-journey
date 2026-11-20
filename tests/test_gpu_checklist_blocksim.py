"""CPU tests of the Day 77 block simulator. No test compiles or runs CUDA."""

import functools

import numpy
import pytest

from ai_journey.gpu_checklist_blocksim import (
    ChecklistTraffic,
    _stage,
    require_aligned,
    run_blocked_blocks,
    run_blocks,
    run_coarsened_blocks,
    vector_flags,
)
from ai_journey.gpu_checklist_cuda import (
    PINNED_BLOCKED,
    PINNED_COARSENED,
    PINNED_KERNELS,
    BlockedKernel,
    CoarsenedKernel,
    GpuChecklistError,
)
from ai_journey.gpu_tiled_blocksim import PHASE_STEPS, SharedHazards
from ai_journey.gpu_tiled_config import PINNED_PLAN
from ai_journey.gpu_tiled_cuda import GpuTiledError
from ai_journey.matmul_metal_mapping import MatmulMetalError
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
)
from ai_journey.memory_access_tiling import MatmulShape, TiledMatmulTraffic

RAGGED = MatmulShape(37, 45, 29)
ALIGNED = MatmulShape(40, 52, 36)
COARSENED, BLOCKED = PINNED_COARSENED[1], PINNED_BLOCKED[0]
DROPPED = (
    ("load", "fma", "barrier"),
    ("load", "barrier", "fma"),
    ("load", "fma"),
    ("barrier", "load", "barrier", "fma"),
)


@functools.cache
def day_74_case(index):
    shape = PINNED_PLAN.shapes[index]
    left, right = matmul_inputs(PINNED_PLAN.seed, CORRECTNESS_STREAM, index, shape)
    return left, right, fma_replay(left, right)


def operands(shape, seed=0):
    generator = numpy.random.default_rng(seed)
    left = generator.standard_normal((shape.m, shape.k), dtype=numpy.float32)
    right = generator.standard_normal((shape.k, shape.n), dtype=numpy.float32)
    return left, right


def same_bits(output, replay):
    return numpy.array_equal(output.view(numpy.uint32), replay.view(numpy.uint32))


@pytest.mark.parametrize("kernel", PINNED_KERNELS, ids=lambda item: item.name)
@pytest.mark.parametrize("index", range(len(PINNED_PLAN.shapes)))
def test_each_day_74_case_matches_the_fma_replay_bit_for_bit(index, kernel):
    left, right, replay = day_74_case(index)
    shape = PINNED_PLAN.shapes[index]

    run = run_blocks(left, right, kernel=kernel)

    assert run.output.dtype == numpy.float32 and same_bits(run.output, replay)
    assert run.output_sha256 == matrix_sha256(replay)
    assert run.matches_traffic and run.unsafe_vector_loads == 0
    assert run.hazards.race_free and run.stored_once and run.one_writer_per_slot
    assert run.launch == kernel.launch(shape)
    assert run.inner_steps == shape.k
    assert run.hazards.barriers == run.traffic.barriers_per_block
    assert (run.a_vector, run.b_vector) == vector_flags(kernel, shape, (True, True))


def test_the_day_74_cases_reach_both_float4_paths():
    flags = {
        index: vector_flags(BLOCKED, shape, (True, True))
        for index, shape in enumerate(PINNED_PLAN.shapes)
    }

    # k and n are multiples of 4 in cases 2, 6 and 7; only k in 4 and 8.
    assert [index for index, flag in flags.items() if flag == (True, True)] == [2, 6, 7]
    assert [index for index, flag in flags.items() if flag == (True, False)] == [4, 8]
    assert all(
        flag == (False, False) for index, flag in flags.items() if index in (0, 1, 3, 5)
    )
    assert vector_flags(COARSENED, ALIGNED, (True, True)) == (False, False)


def test_coarsening_cuts_the_a_loads_of_the_day_76_tile():
    left, right = operands(RAGGED)
    run = run_coarsened_blocks(left, right, kernel=COARSENED)
    day_76 = TiledMatmulTraffic(RAGGED, tile=16)

    # One block column of width 64 reads A once; the Day 76 tile has 3.
    assert (run.a_loads, day_76.useful_a_loads) == (37 * 29, 3 * 37 * 29)
    assert run.b_loads == day_76.useful_b_loads == 3 * 29 * 45
    assert run.to_mapping()["global_loads"] == {
        "a": 1073,
        "b": 3915,
        "a_float4": 0,
        "b_float4": 0,
        "unsafe_float4": 0,
        "zero_filled_slots": 7680 - 1073 - 3915,
        "tile_slot_writes": 3 * 2 * (1 + 4) * 256,
    }
    assert run.shared_reads == 2 * run.fma_steps == 2 * 3 * 256 * 4 * 29
    assert run.stores == 37 * 45 and run.hazards == SharedHazards(16, 0, 0)


def test_the_blocked_counts_follow_the_declared_formula():
    left, right = operands(ALIGNED)
    run = run_blocked_blocks(left, right, kernel=BLOCKED)
    model = ChecklistTraffic(BLOCKED, ALIGNED, True, True)

    assert run.traffic == model and run.matches_traffic
    assert model.to_mapping() == {
        "kernel": "blocked_matmul_m64_n64_k16_t4x4",
        "shape": {"m": 40, "n": 52, "k": 36},
        "vector": {"a": True, "b": True},
        "phases": 3,
        "a_loads": 40 * 36,
        "b_loads": 36 * 52,
        "a_float4_loads": 360,
        "b_float4_loads": 468,
        "tile_slot_writes": 3 * 16 * 128,
        "zero_filled_slots": 6144 - 1440 - 1872,
        "barriers_per_block": 6,
        "fma_steps": 256 * 16 * 36,
        "shared_reads": 256 * 8 * 36,
        "stores": 40 * 52,
    }
    # A thread reads 8 shared values for 16 fma operations in each step.
    assert run.shared_reads * 2 == run.fma_steps


def test_unaligned_bases_take_the_scalar_loads_with_the_same_bits():
    left, right = operands(ALIGNED)
    vector = run_blocked_blocks(left, right, kernel=BLOCKED)
    scalar = run_blocked_blocks(left, right, kernel=BLOCKED, aligned=(False, False))
    mixed = run_blocked_blocks(left, right, kernel=BLOCKED, aligned=(True, False))

    assert (vector.a_vector_loads, vector.b_vector_loads) == (360, 468)
    assert (scalar.a_vector_loads, scalar.b_vector_loads) == (0, 0)
    assert (mixed.a_vector_loads, mixed.b_vector_loads) == (360, 0)
    for run in (vector, scalar, mixed):
        assert run.a_loads == 1440 and run.b_loads == 1872 and run.matches_traffic
        assert run.output_sha256 == matrix_sha256(fma_replay(left, right))


def test_a_float4_that_leaves_its_row_or_its_alignment_is_unsafe():
    source = numpy.arange(12, dtype=numpy.float32).reshape(2, 6)
    lines, starts = numpy.array([0, 0, 0, 1]), numpy.array([0, 4, 1, 2])

    values, vector, scalar, unsafe = _stage(source, lines, starts, (2, 6), True)
    # The group at column 4 leaves the row of 6, and the one at column 1
    # starts at an offset that 4 does not divide. Offset 1 * 6 + 2 = 8 is safe.
    assert (vector, scalar, unsafe) == (4, 0, 2)
    values, vector, scalar, unsafe = _stage(source, lines, starts, (2, 6), False)
    assert (vector, scalar, unsafe) == (0, 14, 0)
    assert values[1].tolist() == [4.0, 5.0, 0.0, 0.0]
    assert values[3].tolist() == [8.0, 9.0, 10.0, 11.0]


def brute_hazards(kernel, shape, steps):
    """Count the hazards again with Python sets, from the index text of the kernel."""

    if type(kernel) is CoarsenedKernel:
        t, threads = kernel.tile, kernel.tile * kernel.tile
        cells = [divmod(thread, t) for thread in range(threads)]
        write_a = [(ty * t + tx, ("A", ty * t + tx)) for ty, tx in cells]
        write_b = [(ty * t + tx, ("B", ty * t + tx)) for ty, tx in cells]

        def reads(count):
            return {
                (ty * t + tx, slot)
                for ty, tx in cells
                for i in range(count)
                for slot in (("A", ty * t + i), ("B", i * t + tx))
            }

        rounds, step_k = kernel.coarsening, t
    else:
        x, _ = kernel.block
        bm, bn, bk = kernel.block_m, kernel.block_n, kernel.block_k
        ga, gb = bk // 4, bn // 4
        threads = kernel.threads_per_block
        write_a = [
            (t, ("A", (t % ga * 4 + q) * bm + t // ga))
            for t in range(threads)
            for q in range(4)
        ] + [
            (t, ("B", t // gb * bn + t % gb * 4 + q))
            for t in range(threads)
            for q in range(4)
        ]
        write_b = []

        def reads(count):
            return {
                (t, slot)
                for t in range(threads)
                for i in range(count)
                for slot in (
                    *(
                        ("A", i * bm + t // x * kernel.thread_m + mi)
                        for mi in range(kernel.thread_m)
                    ),
                    *(
                        ("B", i * bn + t % x * kernel.thread_n + ni)
                        for ni in range(kernel.thread_n)
                    ),
                )
            }

        rounds, step_k = 1, bk
    raw = war = barriers = 0
    writer, readers = {}, {}

    def write(pairs):
        nonlocal war
        for thread, slot in pairs:
            war += len(readers.get(slot, set()) - {thread})
            writer[slot] = thread

    for phase in range(-(-shape.k // step_k)):
        count = min(step_k, shape.k - phase * step_k)
        if rounds > 1:
            write(write_a)
        for _ in range(rounds):
            for step in steps:
                if step == "barrier":
                    barriers += 1
                    writer.clear()
                    readers.clear()
                elif step == "load":
                    write(write_b if rounds > 1 else write_a)
                else:
                    for thread, slot in reads(count):
                        raw += slot in writer and writer[slot] != thread
                        readers.setdefault(slot, set()).add(thread)
    return SharedHazards(barriers, raw, war)


@pytest.mark.parametrize("kernel", (COARSENED, BLOCKED), ids=lambda item: item.name)
def test_a_dropped_barrier_is_a_counted_hazard_with_the_same_values(kernel):
    left, right = operands(RAGGED)
    pinned = run_blocks(left, right, kernel=kernel)
    runs = {
        steps: run_blocks(left, right, kernel=kernel, steps=steps) for steps in DROPPED
    }

    assert pinned.hazards.race_free
    assert pinned.hazards == brute_hazards(kernel, RAGGED, PHASE_STEPS)
    for steps, run in runs.items():
        assert run.hazards == brute_hazards(kernel, RAGGED, steps), steps
        assert run.output_sha256 == pinned.output_sha256
    no_first, no_second, none, leading = (runs[steps].hazards for steps in DROPPED)
    # Without the first barrier, each read meets the write of its phase.
    assert no_first.read_after_write > 0 and no_first.write_after_read == 0
    # Without the second, the next write meets the reads before it.
    assert no_second.read_after_write == 0 and no_second.write_after_read > 0
    # With no barrier, the A writes of a coarsened phase stay new for all of
    # its B tiles, so at least as many reads race.
    assert none.read_after_write >= no_first.read_after_write
    if type(kernel) is CoarsenedKernel:
        # A barrier before each B load does not cover the A load that opens
        # the next phase: it meets the reads of the last round, 16 readers
        # of each of the 256 slots of As.
        assert leading == SharedHazards(16, 0, 256 * 15)
    else:
        assert leading == SharedHazards(4, 0, 0)


def test_the_hazard_counts_of_the_coarsened_kernel_have_closed_forms():
    left, right = operands(RAGGED)
    hazards = {
        steps: run_coarsened_blocks(left, right, kernel=COARSENED, steps=steps).hazards
        for steps in DROPPED[:2]
    }

    # Phase 0 runs 16 steps and phase 1 runs 13. In each of the 4 rounds, a
    # thread reads 16 or 13 slots of each tile, and 1 of them is its own
    # while its tx or ty is below the count. The A writes are new only in
    # the first round of a phase, so that round races on both tiles.
    phase_0, phase_1 = 256 * 15, 16 * (13 * 12 + 3 * 13)
    raw = 5 * phase_0 + 5 * phase_1
    # Rounds 1 to 3 write Bs over the 16 readers of each slot that the last
    # round read, and phase 1 also writes As over those of phase 0.
    war = 3 * phase_0 + 2 * phase_0 + 3 * phase_1
    assert (
        hazards[DROPPED[0]] == SharedHazards(8, raw, 0) == SharedHazards(8, 34_800, 0)
    )
    assert (
        hazards[DROPPED[1]] == SharedHazards(8, 0, war) == SharedHazards(8, 0, 28_560)
    )


def test_the_hazard_counts_of_the_blocked_kernel_have_closed_forms():
    left, right = operands(RAGGED)
    hazards = {
        steps: run_blocked_blocks(left, right, kernel=BLOCKED, steps=steps).hazards
        for steps in DROPPED[:3]
    }

    # Phase 0 reads 16 steps: each thread reads 64 slots of each tile, and 4
    # of each are its own. Phase 1 reads 13 steps: 12,480 reads of each tile
    # come from other threads.
    raw = 256 * 2 * 60 + 2 * 12_480
    # The second phase writes each of the 2 * 1024 slots that 16 threads
    # read, the writer among them.
    war = 2 * 1024 * 15
    assert hazards[DROPPED[0]] == SharedHazards(2, raw, 0)
    assert hazards[DROPPED[1]] == SharedHazards(2, 0, war)
    assert hazards[DROPPED[2]] == SharedHazards(0, raw, war)


def test_the_traffic_of_the_benchmark_points_comes_from_closed_forms():
    shape = MatmulShape(4096, 4096, 4096)
    cube = shape.m * shape.n * shape.k
    loads = {
        kernel.name: ChecklistTraffic(
            kernel, shape, *vector_flags(kernel, shape, (True, True))
        )
        for kernel in PINNED_KERNELS
    }

    assert {name: item.a_loads for name, item in loads.items()} == {
        "coarsened_matmul_t16_c2": cube // 32,
        "coarsened_matmul_t16_c4": cube // 64,
        "coarsened_matmul_t32_c2": cube // 64,
        "blocked_matmul_m64_n64_k16_t4x4": cube // 64,
        "blocked_matmul_m128_n128_k8_t8x8": cube // 128,
    }
    assert loads["coarsened_matmul_t16_c4"].b_loads == (
        TiledMatmulTraffic(shape, tile=16).useful_b_loads
    )
    assert loads["blocked_matmul_m128_n128_k8_t8x8"].a_vector_loads == cube // 512
    for item in loads.values():
        reads_per_fma = item.shared_reads / item.fma_steps
        assert (
            reads_per_fma
            == {2: 2.0, 4: 2.0, 16: 0.5, 64: 0.25}[item.kernel.outputs_per_thread]
        )


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"kernel": COARSENED, "shape": ALIGNED, "a_vector": True}, GpuChecklistError, "no float4"),
        ({"kernel": BLOCKED, "shape": RAGGED, "a_vector": True}, GpuChecklistError, "4 divides"),
        ({"kernel": BLOCKED, "shape": ALIGNED, "b_vector": 1}, TypeError, "booleans"),
        ({"kernel": (64, 64), "shape": ALIGNED}, TypeError, "CoarsenedKernel or"),
    ],
)  # fmt: skip
def test_the_traffic_refuses_flags_that_the_text_never_sets(arguments, error, message):
    with pytest.raises(error, match=message):
        ChecklistTraffic(**arguments)


def test_runs_refuse_other_kernels_operands_schedules_and_flags():
    left, right = operands(RAGGED)

    with pytest.raises(TypeError, match="CoarsenedKernel"):
        run_coarsened_blocks(left, right, kernel=BLOCKED)
    with pytest.raises(TypeError, match="BlockedKernel"):
        run_blocked_blocks(left, right, kernel=COARSENED)
    with pytest.raises(GpuTiledError, match="exactly one"):
        run_blocks(left, right, kernel=BLOCKED, steps=("load", "fma", "fma"))
    with pytest.raises(GpuTiledError, match="follow the load"):
        run_blocks(left, right, kernel=COARSENED, steps=("fma", "barrier", "load"))
    with pytest.raises(MatmulMetalError, match="conformable 2D float32"):
        run_blocks(left.astype(numpy.float64), right, kernel=COARSENED)
    for aligned in ((True,), [True, True], (1, True)):
        with pytest.raises(GpuChecklistError, match="two booleans"):
            require_aligned(aligned)
    with pytest.raises(GpuChecklistError, match="two booleans"):
        run_blocks(left, right, kernel=COARSENED, aligned=(True, None))
    assert BlockedKernel(64, 64, 16, 4, 4) == BLOCKED
