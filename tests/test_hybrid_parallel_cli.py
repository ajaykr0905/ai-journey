from __future__ import annotations

from pathlib import Path

from ai_journey.hybrid_parallel_cli import main
from ai_journey.hybrid_parallel_io import load_hybrid_parallel_report
from ai_journey.hybrid_parallel_verify import main as verify_main


def _arguments(output: Path) -> list[str]:
    return [
        "--world-size",
        "8",
        "--data-parallel-size",
        "1",
        "--tensor-parallel-size",
        "2",
        "--pipeline-parallel-size",
        "2",
        "--sequence-parallel-size",
        "2",
        "--tensor-shape",
        "2,3",
        "--tensor-values",
        "0,1,2,3,4,5",
        "--token-ids",
        "11,12,13,14,15",
        "--layer-count",
        "5",
        "--schedule-policy",
        "1f1b",
        "--microbatch-count",
        "2",
        "--element-size-bytes",
        "4",
        "--pipeline-payload-bytes",
        "16",
        "--source-rank",
        "0",
        "--destination-rank",
        "2",
        "--activation-shape",
        "4",
        "--activation-dtype",
        "float32",
        "--payload-hex",
        bytes(range(16)).hex(),
        "--output",
        str(output),
    ]


def test_generate_executes_cpu_gloo_and_offline_verifier_replays_report(
    tmp_path, capsys
):
    output = tmp_path / "nested" / "hybrid-parallel.json"

    assert main(_arguments(output)) == 0
    generated = capsys.readouterr().out
    assert "backend=gloo" in generated
    assert "device=cpu" in generated
    assert "performance=false" in generated
    before = output.read_bytes()
    report = load_hybrid_parallel_report(output)
    assert report["inputs"]["topology"]["world_size"] == 8
    assert report["analytical"]["schedule"]["policy"] == "1f1b"
    assert report["executed_observations"]["pipeline_handoff"]["outcome"] == ("success")

    assert verify_main(["--report", str(output)]) == 0
    verified = capsys.readouterr().out
    assert "verified_hybrid_parallel=" in verified
    assert "executed=cpu_gloo" in verified
    assert "performance=false" in verified
    assert output.read_bytes() == before


def test_generate_rejects_unbounded_or_inconsistent_explicit_inputs(tmp_path):
    output = tmp_path / "report.json"
    too_large = _arguments(output)
    too_large[too_large.index("--world-size") + 1] = "4097"
    assert main(too_large) == 2
    assert not output.exists()

    wrong_payload = _arguments(output)
    wrong_payload[wrong_payload.index("--payload-hex") + 1] = "00"
    assert main(wrong_payload) == 2
    assert not output.exists()

    malformed_list = _arguments(output)
    malformed_list[malformed_list.index("--token-ids") + 1] = "1,,2"
    assert main(malformed_list) == 2
    assert not output.exists()

    alias = _arguments(output)
    alias[alias.index("--output") + 1] = str(output)
    output.write_text("protected", encoding="utf-8")
    assert main(alias) == 2
    assert output.read_text(encoding="utf-8") == "protected"
