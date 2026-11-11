from ai_journey.sft_finetune_io import load_finetune_report, write_finetune_report
from ai_journey.sft_finetune_report import build_finetune_report
from tests.test_sft_finetune_report import result


def test_report_storage_round_trips_sealed_json_atomically(tmp_path):
    path = tmp_path / "nested" / "report.json"
    report = build_finetune_report(result())
    write_finetune_report(path, report)
    assert load_finetune_report(path) == report
    assert path.read_bytes().endswith(b"\n")
