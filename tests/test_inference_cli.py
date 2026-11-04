import json
from pathlib import Path

from ai_journey.inference_cli import main


def protocol_mapping():
    gib = 1024**3
    return {
        "name": "cli-inference",
        "parameter_count": 7_000_000_000,
        "layers": 32,
        "query_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "context_tokens": 32_768,
        "request_lengths": [32_768] * 16,
        "bytes_per_element": 2,
        "total_device_bytes": 80 * gib,
        "model_bytes": 14 * gib,
        "runtime_reserve_bytes": 6 * gib,
        "block_tokens": 16,
        "speculative": {
            "draft_tokens": 4,
            "acceptance_numerator": 4,
            "acceptance_denominator": 5,
        },
    }


def test_cli_builds_and_verifies_the_same_sealed_report(tmp_path, capsys):
    protocol = tmp_path / "protocol.json"
    report = tmp_path / "report.json"
    protocol.write_text(json.dumps(protocol_mapping()))

    assert main(["build", "--protocol", str(protocol), "--output", str(report)]) == 0
    built_digest = capsys.readouterr().out.strip()
    assert main(["verify", "--report", str(report)]) == 0

    assert capsys.readouterr().out.strip() == built_digest
    assert len(built_digest) == 64


def test_project_installs_the_inference_cli_entry_point():
    package = Path("pyproject.toml").read_text()

    assert 'ai-journey-inference = "ai_journey.inference_cli:main"' in package
