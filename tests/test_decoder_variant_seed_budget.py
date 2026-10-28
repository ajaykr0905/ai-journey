from dataclasses import replace

import pytest

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.decoder_variant_comparison import run_decoder_document_comparison
from ai_journey.decoder_variant_budget import require_comparison_budget
from ai_journey.decoder_variant_plan import DecoderExperimentPlan
from ai_journey.gpt2_config import GPT2Config


def plan(*, seeds, steps=20, evaluation_batch_size=16):
    baseline = DecoderVariantConfig(
        GPT2Config(
            vocab_size=256,
            context_length=16,
            embedding_dim=32,
            head_count=4,
            layer_count=2,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
        )
    )
    return DecoderExperimentPlan(
        variants=(
            ("baseline", baseline),
            ("rmsnorm", replace(baseline, norm_kind="rmsnorm")),
        ),
        declared_changes=("norm_kind",),
        seeds=seeds,
        steps=steps,
        batch_size=4,
        evaluation_batch_size=evaluation_batch_size,
    )


def test_plan_charges_every_declared_seed_to_total_comparison_work():
    # One seed costs 540,672 proxy units; four exceed the 1,000,000 job limit.
    assert plan(seeds=(1,)).seeds == (1,)
    with pytest.raises(ValueError, match="total comparison work"):
        plan(seeds=(1, 2, 3, 4))


def test_document_comparison_charges_actual_evaluation_across_all_seeds():
    # The small nominal plan fits. Its actual held-out evaluation adds 200
    # forwards per arm: 450,560 units per seed, 1,802,240 across all four.
    declared = plan(seeds=(1, 2, 3, 4), steps=1, evaluation_batch_size=1)
    with pytest.raises(ValueError, match="total comparison work"):
        run_decoder_document_comparison(declared, b"abcd" * 10, b"h" * 1600)


def test_sequential_seeds_do_not_multiply_peak_attention():
    # One trial needs 4,194,304 attention elements, below the 8,000,000 limit.
    # Repeating it sequentially four times does not quadruple resident storage.
    base = GPT2Config(
        vocab_size=256,
        context_length=2048,
        embedding_dim=4,
        head_count=1,
        layer_count=1,
    )
    require_comparison_budget(
        base,
        steps=1,
        batch_size=1,
        evaluation_windows=1,
        variant_count=1,
        seed_count=4,
    )


@pytest.mark.parametrize("seed_count", [True, False, 0, -1, 1.5, "2"])
def test_seed_count_requires_a_positive_integer(seed_count):
    with pytest.raises(ValueError, match="positive integers"):
        require_comparison_budget(
            GPT2Config(
                vocab_size=256,
                context_length=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
            ),
            steps=1,
            batch_size=1,
            evaluation_windows=1,
            seed_count=seed_count,
        )
