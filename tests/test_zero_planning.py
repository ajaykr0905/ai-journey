from ai_journey.distributed_capacity import ClusterTopology, LinkProfile
from ai_journey.resource_accounting import DecoderSpec
from ai_journey.zero_planning import (
    estimate_zero_collectives,
    zero_collective_schedule,
    zero_stage_memory_peaks,
    zero_stage_state,
)


def _model() -> DecoderSpec:
    return DecoderSpec(32, 2, 16, 4, 2, 32)


def test_zero_stages_shard_only_the_state_families_they_own():
    expected_sharded = {
        0: frozenset(),
        1: frozenset({"optimizer", "master_parameters"}),
        2: frozenset({"gradients", "optimizer", "master_parameters"}),
        3: frozenset({"parameters", "gradients", "optimizer", "master_parameters"}),
    }
    model = _model()

    for stage, sharded_families in expected_sharded.items():
        states = zero_stage_state(
            model,
            stage=stage,
            world_size=3,
            parameter_dtype="bfloat16",
            gradient_dtype="float32",
            optimizer_dtype="float32",
            optimizer_slots=2,
            master_parameter_dtype="float32",
        )

        assert [state.rank for state in states] == [0, 1, 2]
        assert all(state.stage == stage for state in states)
        for family in (
            "parameters",
            "gradients",
            "optimizer",
            "master_parameters",
        ):
            values = [getattr(state, f"{family}_bytes") for state in states]
            if family in sharded_families:
                assert (
                    max(values) - min(values)
                    <= {
                        "parameters": 2,
                        "gradients": 4,
                        "optimizer": 8,
                        "master_parameters": 4,
                    }[family]
                )
            else:
                assert len(set(values)) == 1


def test_zero_stage_state_preserves_every_sharded_byte_without_padding():
    model = _model()
    states = zero_stage_state(
        model,
        stage=3,
        world_size=3,
        parameter_dtype="bfloat16",
        gradient_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=2,
        master_parameter_dtype="float32",
    )

    assert sum(state.parameters_bytes for state in states) == model.parameter_count * 2
    assert sum(state.gradients_bytes for state in states) == model.parameter_count * 4
    assert sum(state.optimizer_bytes for state in states) == model.parameter_count * 8
    assert sum(state.master_parameters_bytes for state in states) == (
        model.parameter_count * 4
    )


def test_zero_stages_expose_their_distinct_collective_sequences():
    model = _model()
    expected = {
        0: [("gradient_sync", "all_reduce", "gradients")],
        1: [
            ("gradient_sync", "all_reduce", "gradients"),
            ("parameter_sync", "all_gather", "parameters"),
        ],
        2: [
            ("gradient_partition", "reduce_scatter", "gradients"),
            ("parameter_sync", "all_gather", "parameters"),
        ],
        3: [
            ("parameter_materialization", "all_gather", "parameters"),
            ("gradient_partition", "reduce_scatter", "gradients"),
        ],
    }

    for stage, expected_steps in expected.items():
        schedule = zero_collective_schedule(
            model,
            stage=stage,
            world_size=3,
            parameter_dtype="bfloat16",
            gradient_dtype="float32",
        )
        assert [
            (step.phase, step.operation, step.tensor_family) for step in schedule
        ] == expected_steps


def test_zero_collective_schedule_preserves_uneven_logical_payloads():
    model = _model()
    parameter_bytes = model.parameter_count * 2
    gradient_bytes = model.parameter_count * 4

    stage_two = zero_collective_schedule(
        model,
        stage=2,
        world_size=3,
        parameter_dtype="bfloat16",
        gradient_dtype="float32",
    )
    gradient_partition, parameter_sync = stage_two
    assert gradient_partition.input_bytes_by_rank == (gradient_bytes,) * 3
    assert sum(gradient_partition.output_bytes_by_rank) == gradient_bytes
    assert sum(parameter_sync.input_bytes_by_rank) == parameter_bytes
    assert parameter_sync.output_bytes_by_rank == (parameter_bytes,) * 3


def test_zero_memory_peaks_separate_persistent_and_transient_buffers():
    model = _model()
    peaks = zero_stage_memory_peaks(
        model,
        stage=3,
        world_size=3,
        parameter_dtype="bfloat16",
        gradient_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=2,
        master_parameter_dtype="float32",
        activation_bytes=1_000,
        gradient_bucket_bytes=4_096,
    )

    assert len(peaks) == 3
    assert peaks[0].persistent_bytes == 34_272
    assert peaks[0].activation_bytes == 1_000
    assert peaks[0].parameter_materialization_bytes == 7_616
    assert peaks[0].gradient_partition_buffer_bytes == 4_096
    assert peaks[0].peak_bytes == 42_888
    assert peaks[0].dominant_transient == "parameter_materialization"


def test_zero_stage_memory_reduces_peak_as_state_is_progressively_sharded():
    model = _model()
    stage_peaks = {
        stage: zero_stage_memory_peaks(
            model,
            stage=stage,
            world_size=3,
            parameter_dtype="bfloat16",
            gradient_dtype="float32",
            optimizer_dtype="float32",
            optimizer_slots=2,
            master_parameter_dtype="float32",
            activation_bytes=1_000,
            gradient_bucket_bytes=4_096,
        )[0].peak_bytes
        for stage in range(4)
    }

    assert stage_peaks == {0: 103_816, 1: 58_120, 2: 46_984, 3: 42_888}


def test_zero_collective_estimates_expose_padding_and_wire_costs():
    topology = ClusterTopology(
        world_size=5,
        workers_per_node=5,
        intra_node=LinkProfile(1_000, 0.1),
        inter_node=LinkProfile(100, 1.0),
    )
    estimates = estimate_zero_collectives(
        _model(),
        stage=2,
        topology=topology,
        parameter_dtype="bfloat16",
        gradient_dtype="float32",
    )

    gradient_partition, parameter_sync = estimates
    assert gradient_partition.operation == "reduce_scatter"
    assert gradient_partition.logical_payload_bytes == 22_848
    assert gradient_partition.padded_payload_bytes == 22_850
    assert gradient_partition.padding_bytes == 2
    assert gradient_partition.wire_bytes_per_rank == 18_280
    assert gradient_partition.link_scope == "intra_node"

    assert parameter_sync.operation == "all_gather"
    assert parameter_sync.logical_payload_bytes == 11_424
    assert parameter_sync.padded_payload_bytes == 11_430
    assert parameter_sync.padding_bytes == 6
    assert parameter_sync.wire_bytes_per_rank == 9_144
