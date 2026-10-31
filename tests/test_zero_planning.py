from ai_journey.resource_accounting import DecoderSpec
from ai_journey.zero_planning import zero_stage_state


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
