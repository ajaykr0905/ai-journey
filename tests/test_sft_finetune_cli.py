import json

from ai_journey.sft_finetune_cli import main
from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_io import load_finetune_report


def test_build_command_runs_and_seals_a_bounded_experiment(tmp_path):
    dataset = tmp_path / "chatml.jsonl"
    rows = [
        {"assistant_spans": [[7, 13]], "example_id": f"train-{index}", "split": "train", "text": "prompt answer"}
        for index in range(3)
    ] + [
        {"assistant_spans": [[7, 13]], "example_id": "held", "split": "validation", "text": "prompt answer"}
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            SFTFineTuneConfig(
                block_size=8, batch_size=2, steps=1, evaluation_examples=1,
                generation_tokens=1, checkpoint_interval=1,
            ).as_mapping()
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    assert main(["build", "--dataset", str(dataset), "--config", str(config), "--output", str(output)]) == 0
    assert load_finetune_report(output)["result"]["training"]["optimizer_steps"] == 1
