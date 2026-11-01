"""Pipeline bubble evidence is derived from validated schedule slots."""

from dataclasses import replace

import pytest

from ai_journey.hybrid_topology import HybridParallelTopology
from ai_journey.pipeline_metrics import measure_pipeline_bubbles
from ai_journey.pipeline_partition import partition_transformer_layers
from ai_journey.pipeline_schedule import generate_gpipe_schedule
from ai_journey.pipeline_schedule import generate_1f1b_schedule


def test_gpipe_metrics_expose_busy_idle_denominators_and_longest_bubbles():
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=4)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=2)

    metrics = measure_pipeline_bubbles(schedule)

    assert metrics.to_mapping() == {
        "policy": "gpipe",
        "stage_count": 2,
        "microbatch_count": 2,
        "schedule_slot_count": 6,
        "busy_stage_slots": 8,
        "idle_stage_slots": 4,
        "stage_slot_denominator": 12,
        "bubble_fraction": {
            "numerator": 4,
            "denominator": 12,
            "value": pytest.approx(1 / 3),
        },
        "utilization": {
            "numerator": 8,
            "denominator": 12,
            "value": pytest.approx(2 / 3),
        },
        "stages": [
            {
                "stage_index": 0,
                "busy_slots": 4,
                "idle_slots": 2,
                "longest_idle_run": 2,
            },
            {
                "stage_index": 1,
                "busy_slots": 4,
                "idle_slots": 2,
                "longest_idle_run": 1,
            },
        ],
        "evidence_boundary": (
            "Analytical schedule-slot evidence only; no execution timing, "
            "hardware utilization, throughput, or production behavior was measured."
        ),
    }


@pytest.mark.parametrize("schedule_policy", ["gpipe", "1f1b"])
def test_one_stage_one_microbatch_has_no_pipeline_bubble(schedule_policy):
    topology = HybridParallelTopology(1, 1, 1, 1, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=1)
    generator = (
        generate_gpipe_schedule
        if schedule_policy == "gpipe"
        else generate_1f1b_schedule
    )

    metrics = measure_pipeline_bubbles(generator(stage_plan, microbatch_count=1))

    assert metrics.busy_stage_slots == 2
    assert metrics.idle_stage_slots == 0
    assert metrics.stage_slot_denominator == 2
    assert metrics.bubble_fraction == 0.0
    assert metrics.utilization == 1.0
    assert tuple(stage.longest_idle_run for stage in metrics.stages) == (0,)


def test_one_f_one_b_metrics_cover_uneven_warmup_and_cooldown_lengths():
    topology = HybridParallelTopology(4, 1, 1, 4, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=8)

    metrics = measure_pipeline_bubbles(
        generate_1f1b_schedule(stage_plan, microbatch_count=2)
    )

    assert metrics.schedule_slot_count == 10
    assert metrics.busy_stage_slots == 16
    assert metrics.idle_stage_slots == 24
    assert metrics.stage_slot_denominator == 40
    assert metrics.bubble_fraction == 0.6
    assert metrics.utilization == 0.4
    assert tuple(stage.busy_slots for stage in metrics.stages) == (4, 4, 4, 4)
    assert tuple(stage.idle_slots for stage in metrics.stages) == (6, 6, 6, 6)
    assert tuple(stage.longest_idle_run for stage in metrics.stages) == (5, 3, 2, 3)


def test_metrics_reject_an_incomplete_schedule_before_deriving_evidence():
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=2)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=1)
    incomplete_slot = replace(schedule.slots[0], operations=())
    incomplete = replace(schedule, slots=(incomplete_slot, *schedule.slots[1:]))

    with pytest.raises(ValueError, match="missing or unexpected operations"):
        measure_pipeline_bubbles(incomplete)


def test_metrics_reject_a_schedule_that_has_not_satisfied_its_policy():
    topology = HybridParallelTopology(2, 1, 1, 2, 1)
    stage_plan = partition_transformer_layers(topology, layer_count=2)
    schedule = generate_1f1b_schedule(stage_plan, microbatch_count=1)
    invalid_operation = replace(schedule.slots[0].operations[0], phase="steady")
    invalid_slot = replace(schedule.slots[0], operations=(invalid_operation,))
    invalid = replace(schedule, slots=(invalid_slot, *schedule.slots[1:]))

    with pytest.raises(ValueError, match="phase does not match"):
        measure_pipeline_bubbles(invalid)


def test_metrics_require_a_pipeline_schedule():
    with pytest.raises(TypeError, match="schedule must be a PipelineSchedule"):
        measure_pipeline_bubbles(object())
