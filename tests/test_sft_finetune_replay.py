from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_experiment import run_finetune_experiment
from ai_journey.sft_finetune_replay import replay_finetune_report
from ai_journey.sft_finetune_report import build_finetune_report
from tests.test_sft_finetune_experiment import dataset


def test_replay_rebuilds_the_exact_sealed_experiment():
    source = dataset()
    config = SFTFineTuneConfig(
        block_size=8,
        batch_size=2,
        steps=1,
        evaluation_examples=1,
        generation_tokens=1,
        checkpoint_interval=1,
    )
    report = build_finetune_report(run_finetune_experiment(source, config))
    result = replay_finetune_report(report, source)
    assert result["report_sha256"] == report["sha256"]
    assert result["replayed_steps"] == 1
