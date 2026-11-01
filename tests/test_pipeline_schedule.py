"""GPipe schedules expose deterministic dependency-valid execution slots."""

from dataclasses import FrozenInstanceError, replace

import pytest

from ai_journey.hybrid_topology import HybridParallelTopology
from ai_journey.pipeline_partition import partition_transformer_layers
from ai_journey.pipeline_schedule import (
    PipelineSlot,
    generate_1f1b_schedule,
    generate_gpipe_schedule,
    validate_pipeline_schedule,
)


def test_one_f_one_b_schedule_exposes_warmup_steady_and_cooldown_phases():
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=4)

    schedule = generate_1f1b_schedule(stage_plan, microbatch_count=2)

    assert schedule.to_mapping() == {
        "policy": "1f1b",
        "stage_count": 2,
        "microbatch_count": 2,
        "slots": [
            {
                "index": 0,
                "operations": [
                    {
                        "identity": "forward:0:0",
                        "stage_index": 0,
                        "microbatch_index": 0,
                        "direction": "forward",
                        "phase": "warmup",
                        "dependencies": [],
                    }
                ],
            },
            {
                "index": 1,
                "operations": [
                    {
                        "identity": "forward:0:1",
                        "stage_index": 0,
                        "microbatch_index": 1,
                        "direction": "forward",
                        "phase": "steady",
                        "dependencies": [],
                    },
                    {
                        "identity": "forward:1:0",
                        "stage_index": 1,
                        "microbatch_index": 0,
                        "direction": "forward",
                        "phase": "steady",
                        "dependencies": ["forward:0:0"],
                    },
                ],
            },
            {
                "index": 2,
                "operations": [
                    {
                        "identity": "backward:1:0",
                        "stage_index": 1,
                        "microbatch_index": 0,
                        "direction": "backward",
                        "phase": "steady",
                        "dependencies": ["forward:1:0"],
                    }
                ],
            },
            {
                "index": 3,
                "operations": [
                    {
                        "identity": "backward:0:0",
                        "stage_index": 0,
                        "microbatch_index": 0,
                        "direction": "backward",
                        "phase": "steady",
                        "dependencies": ["forward:0:0", "backward:1:0"],
                    },
                    {
                        "identity": "forward:1:1",
                        "stage_index": 1,
                        "microbatch_index": 1,
                        "direction": "forward",
                        "phase": "steady",
                        "dependencies": ["forward:0:1"],
                    },
                ],
            },
            {
                "index": 4,
                "operations": [
                    {
                        "identity": "backward:1:1",
                        "stage_index": 1,
                        "microbatch_index": 1,
                        "direction": "backward",
                        "phase": "steady",
                        "dependencies": ["forward:1:1"],
                    }
                ],
            },
            {
                "index": 5,
                "operations": [
                    {
                        "identity": "backward:0:1",
                        "stage_index": 0,
                        "microbatch_index": 1,
                        "direction": "backward",
                        "phase": "cooldown",
                        "dependencies": ["forward:0:1", "backward:1:1"],
                    }
                ],
            },
        ],
    }
    validate_pipeline_schedule(schedule)


def test_one_f_one_b_validation_rejects_a_phase_that_misstates_the_policy():
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=4)
    schedule = generate_1f1b_schedule(stage_plan, microbatch_count=2)
    operation = replace(schedule.slots[0].operations[0], phase="steady")
    slot = replace(schedule.slots[0], operations=(operation,))

    with pytest.raises(ValueError, match="phase does not match"):
        validate_pipeline_schedule(replace(schedule, slots=(slot, *schedule.slots[1:])))


def test_one_f_one_b_validation_rejects_nonalternating_stage_work():
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=4)
    gpipe = generate_gpipe_schedule(stage_plan, microbatch_count=2)
    phase_by_identity = {
        operation.identity: operation.phase
        for slot in generate_1f1b_schedule(stage_plan, microbatch_count=2).slots
        for operation in slot.operations
    }
    relabeled_slots = tuple(
        replace(
            slot,
            operations=tuple(
                replace(operation, phase=phase_by_identity[operation.identity])
                for operation in slot.operations
            ),
        )
        for slot in gpipe.slots
    )

    with pytest.raises(ValueError, match="operation order does not match"):
        validate_pipeline_schedule(replace(gpipe, policy="1f1b", slots=relabeled_slots))


@pytest.mark.parametrize(
    "stage_count,microbatch_count,expected_phases",
    [
        (1, 3, (("steady",) * 6,)),
        (
            3,
            1,
            (
                ("warmup", "cooldown"),
                ("warmup", "cooldown"),
                ("steady", "steady"),
            ),
        ),
        (
            4,
            2,
            (
                ("warmup", "warmup", "cooldown", "cooldown"),
                ("warmup", "warmup", "cooldown", "cooldown"),
                ("warmup", "steady", "steady", "cooldown"),
                ("steady", "steady", "steady", "steady"),
            ),
        ),
    ],
)
def test_one_f_one_b_phase_boundaries_cover_degenerate_and_uneven_pipelines(
    stage_count, microbatch_count, expected_phases
):
    topology = HybridParallelTopology(stage_count, 1, 1, stage_count, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=stage_count)

    schedule = generate_1f1b_schedule(stage_plan, microbatch_count=microbatch_count)
    phases_by_stage = tuple(
        tuple(
            operation.phase
            for slot in schedule.slots
            for operation in slot.operations
            if operation.stage_index == stage_index
        )
        for stage_index in range(stage_count)
    )

    assert phases_by_stage == expected_phases
    validate_pipeline_schedule(schedule)


def test_one_f_one_b_emits_every_required_operation_exactly_once():
    topology = HybridParallelTopology(3, 1, 1, 3, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=6)

    schedule = generate_1f1b_schedule(stage_plan, microbatch_count=5)
    identities = tuple(
        operation.identity for slot in schedule.slots for operation in slot.operations
    )

    assert len(identities) == 30
    assert len(set(identities)) == 30
    validate_pipeline_schedule(schedule)


def test_one_f_one_b_reduces_the_longest_analytical_bubble_without_runtime_claims():
    topology = HybridParallelTopology(4, 1, 1, 4, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=8)
    gpipe = generate_gpipe_schedule(stage_plan, microbatch_count=5)
    one_f_one_b = generate_1f1b_schedule(stage_plan, microbatch_count=5)

    def longest_contiguous_bubble(schedule, stage_index):
        longest = 0
        current = 0
        for slot in schedule.slots:
            if any(
                operation.stage_index == stage_index for operation in slot.operations
            ):
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return longest

    assert tuple(
        longest_contiguous_bubble(gpipe, stage_index) for stage_index in range(4)
    ) == (6, 4, 2, 3)
    assert tuple(
        longest_contiguous_bubble(one_f_one_b, stage_index) for stage_index in range(4)
    ) == (3, 2, 2, 3)
    assert len(one_f_one_b.slots) == len(gpipe.slots)


def test_one_f_one_b_generation_requires_an_admitted_plan_and_microbatches():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=1)

    with pytest.raises(TypeError, match="stage_plan must be a PipelinePartitionPlan"):
        generate_1f1b_schedule(object(), microbatch_count=1)
    with pytest.raises(TypeError, match="microbatch_count must be an integer"):
        generate_1f1b_schedule(stage_plan, microbatch_count=True)
    with pytest.raises(ValueError, match="microbatch_count must be positive"):
        generate_1f1b_schedule(stage_plan, microbatch_count=0)


def test_gpipe_schedule_exposes_canonical_forward_and_backward_slots():
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=4)

    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=2)

    assert schedule.to_mapping() == {
        "policy": "gpipe",
        "stage_count": 2,
        "microbatch_count": 2,
        "slots": [
            {
                "index": 0,
                "operations": [
                    {
                        "identity": "forward:0:0",
                        "stage_index": 0,
                        "microbatch_index": 0,
                        "direction": "forward",
                        "phase": "forward",
                        "dependencies": [],
                    }
                ],
            },
            {
                "index": 1,
                "operations": [
                    {
                        "identity": "forward:0:1",
                        "stage_index": 0,
                        "microbatch_index": 1,
                        "direction": "forward",
                        "phase": "forward",
                        "dependencies": [],
                    },
                    {
                        "identity": "forward:1:0",
                        "stage_index": 1,
                        "microbatch_index": 0,
                        "direction": "forward",
                        "phase": "forward",
                        "dependencies": ["forward:0:0"],
                    },
                ],
            },
            {
                "index": 2,
                "operations": [
                    {
                        "identity": "forward:1:1",
                        "stage_index": 1,
                        "microbatch_index": 1,
                        "direction": "forward",
                        "phase": "forward",
                        "dependencies": ["forward:0:1"],
                    }
                ],
            },
            {
                "index": 3,
                "operations": [
                    {
                        "identity": "backward:1:1",
                        "stage_index": 1,
                        "microbatch_index": 1,
                        "direction": "backward",
                        "phase": "backward",
                        "dependencies": ["forward:1:1"],
                    }
                ],
            },
            {
                "index": 4,
                "operations": [
                    {
                        "identity": "backward:0:1",
                        "stage_index": 0,
                        "microbatch_index": 1,
                        "direction": "backward",
                        "phase": "backward",
                        "dependencies": ["forward:0:1", "backward:1:1"],
                    },
                    {
                        "identity": "backward:1:0",
                        "stage_index": 1,
                        "microbatch_index": 0,
                        "direction": "backward",
                        "phase": "backward",
                        "dependencies": ["forward:1:0"],
                    },
                ],
            },
            {
                "index": 5,
                "operations": [
                    {
                        "identity": "backward:0:0",
                        "stage_index": 0,
                        "microbatch_index": 0,
                        "direction": "backward",
                        "phase": "backward",
                        "dependencies": ["forward:0:0", "backward:1:0"],
                    }
                ],
            },
        ],
    }


def test_gpipe_schedule_emits_every_required_operation_exactly_once():
    topology = HybridParallelTopology(3, 1, 1, 3, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=6)

    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=4)
    identities = tuple(
        operation.identity for slot in schedule.slots for operation in slot.operations
    )

    assert len(identities) == 24
    assert set(identities) == {
        f"{direction}:{stage_index}:{microbatch_index}"
        for direction in ("forward", "backward")
        for stage_index in range(3)
        for microbatch_index in range(4)
    }
    validate_pipeline_schedule(schedule)


def test_each_microbatch_visits_forward_then_reverse_backward_stage_order():
    topology = HybridParallelTopology(3, 1, 1, 3, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=6)

    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=3)
    slot_by_identity = {
        operation.identity: slot.index
        for slot in schedule.slots
        for operation in slot.operations
    }

    assert tuple(
        tuple(slot_by_identity[f"forward:{stage}:{microbatch}"] for stage in range(3))
        for microbatch in range(3)
    ) == ((0, 1, 2), (1, 2, 3), (2, 3, 4))
    assert tuple(
        tuple(
            slot_by_identity[f"backward:{stage}:{microbatch}"]
            for stage in reversed(range(3))
        )
        for microbatch in range(3)
    ) == ((7, 8, 9), (6, 7, 8), (5, 6, 7))


def test_pipeline_schedule_slots_and_operations_are_immutable():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=1)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=1)

    with pytest.raises(FrozenInstanceError):
        schedule.microbatch_count = 2
    with pytest.raises(FrozenInstanceError):
        schedule.slots[0].index = 3
    with pytest.raises(FrozenInstanceError):
        schedule.slots[0].operations[0].phase = "steady"


@pytest.mark.parametrize(
    "slot_index,operation_index,dependencies",
    [
        (1, 1, ()),
        (4, 0, ("forward:0:1",)),
    ],
)
def test_schedule_validation_rejects_missing_required_dependencies(
    slot_index, operation_index, dependencies
):
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=4)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=2)
    original_slot = schedule.slots[slot_index]
    invalid_operation = replace(
        original_slot.operations[operation_index], dependencies=dependencies
    )
    operations = list(original_slot.operations)
    operations[operation_index] = invalid_operation
    invalid_slot = replace(
        original_slot,
        operations=tuple(operations),
    )
    slots = list(schedule.slots)
    slots[slot_index] = invalid_slot
    invalid_schedule = replace(
        schedule,
        slots=tuple(slots),
    )

    with pytest.raises(ValueError, match="dependencies do not match"):
        validate_pipeline_schedule(invalid_schedule)


@pytest.mark.parametrize("first,second", [(0, 1), (2, 3)])
def test_schedule_validation_rejects_dependencies_in_later_slots(first, second):
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=4)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=1)
    slots = list(schedule.slots)
    slots[first], slots[second] = (
        replace(slots[second], index=first),
        replace(slots[first], index=second),
    )

    with pytest.raises(ValueError, match="dependency must precede"):
        validate_pipeline_schedule(replace(schedule, slots=tuple(slots)))


def test_gpipe_generation_requires_an_admitted_plan_and_positive_microbatches():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=1)

    with pytest.raises(TypeError, match="stage_plan must be a PipelinePartitionPlan"):
        generate_gpipe_schedule(object(), microbatch_count=1)
    with pytest.raises(TypeError, match="microbatch_count must be an integer"):
        generate_gpipe_schedule(stage_plan, microbatch_count=True)
    with pytest.raises(ValueError, match="microbatch_count must be positive"):
        generate_gpipe_schedule(stage_plan, microbatch_count=0)


def test_gpipe_validation_rejects_a_phase_that_misstates_the_operation():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=1)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=1)
    operation = replace(schedule.slots[0].operations[0], phase="steady")
    slot = replace(schedule.slots[0], operations=(operation,))
    invalid_schedule = replace(schedule, slots=(slot, *schedule.slots[1:]))

    with pytest.raises(ValueError, match="phase does not match"):
        validate_pipeline_schedule(invalid_schedule)


def test_schedule_validation_rejects_missing_and_duplicate_operations():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=1)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=1)

    with pytest.raises(ValueError, match="missing or unexpected operations"):
        validate_pipeline_schedule(replace(schedule, slots=schedule.slots[:-1]))

    duplicate_slot = replace(
        schedule.slots[1],
        operations=(schedule.slots[0].operations[0], schedule.slots[1].operations[0]),
    )
    with pytest.raises(ValueError, match="duplicate operations"):
        validate_pipeline_schedule(
            replace(schedule, slots=(schedule.slots[0], duplicate_slot))
        )


def test_schedule_validation_rejects_noncanonical_slot_indices():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=1)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=1)
    invalid_slot = replace(schedule.slots[0], index=7)

    with pytest.raises(ValueError, match="slot indices must be consecutive"):
        validate_pipeline_schedule(
            replace(schedule, slots=(invalid_slot, *schedule.slots[1:]))
        )


def test_gpipe_validation_rejects_dependency_valid_forward_interleaving():
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=4)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=2)
    operations = {
        operation.identity: operation
        for slot in schedule.slots
        for operation in slot.operations
    }
    order = (
        ("forward:0:0",),
        ("forward:0:1", "forward:1:0"),
        ("backward:1:0",),
        ("forward:1:1",),
        ("backward:1:1",),
        ("backward:0:1",),
        ("backward:0:0",),
    )
    interleaved = replace(
        schedule,
        slots=tuple(
            PipelineSlot(index, tuple(operations[identity] for identity in identities))
            for index, identities in enumerate(order)
        ),
    )

    with pytest.raises(ValueError, match="GPipe forwards must finish"):
        validate_pipeline_schedule(interleaved)


def test_schedule_validation_rejects_two_operations_on_one_stage_in_a_slot():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=1)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=2)
    operations = tuple(
        operation for slot in schedule.slots for operation in slot.operations
    )
    oversubscribed = replace(
        schedule,
        slots=(
            PipelineSlot(0, operations[:2]),
            PipelineSlot(1, (operations[2],)),
            PipelineSlot(2, (operations[3],)),
        ),
    )

    with pytest.raises(ValueError, match="one operation per stage"):
        validate_pipeline_schedule(oversubscribed)


def test_schedule_validation_rejects_unknown_schedule_policies():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=1)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=1)

    with pytest.raises(ValueError, match="policy must be gpipe or 1f1b"):
        validate_pipeline_schedule(replace(schedule, policy="interleaved"))
