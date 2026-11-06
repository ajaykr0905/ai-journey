from hashlib import sha256
import json
from pathlib import Path

from ai_journey.evaluation_suite_cli import main
from ai_journey.evaluation_suite_evidence import read_suite
from ai_journey.hellaswag_cli import main as hellaswag_main
from ai_journey.lambada_cli import main as lambada_main
from test_hellaswag import example_payload, protocol_payload as hellaswag_protocol
from test_hellaswag import server_payload as hellaswag_server
from test_lambada import protocol_payload as lambada_protocol
from test_lambada import server_payload as lambada_server


def benchmark_reports(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    source = root / "data/day-19-demo-names.txt"
    spec = root / "config/gpt2-training.json"

    hellaswag_response = json.dumps(
        hellaswag_server(example_payload(label="0")), separators=(",", ":")
    ).encode()
    hellaswag_data = tmp_path / "hellaswag-data.json"
    hellaswag_data.write_bytes(hellaswag_response)
    hellaswag_config = hellaswag_protocol()
    hellaswag_config.update(count=1, world_size=1, bootstrap_samples=100)
    hellaswag_config["source"].update(
        sample_sha256=sha256(hellaswag_response).hexdigest(),
        sample_bytes=len(hellaswag_response),
    )
    hellaswag_config_path = tmp_path / "hellaswag-protocol.json"
    hellaswag_config_path.write_text(json.dumps(hellaswag_config))
    hellaswag_report = tmp_path / "hellaswag-report.json"
    assert (
        hellaswag_main(
            [
                "evaluate",
                "--dataset-response",
                str(hellaswag_data),
                "--protocol",
                str(hellaswag_config_path),
                "--source",
                str(source),
                "--spec",
                str(spec),
                "--output",
                str(hellaswag_report),
            ]
        )
        == 0
    )

    lambada_response = json.dumps(
        lambada_server(), sort_keys=True, separators=(",", ":")
    ).encode()
    lambada_data = tmp_path / "lambada-data.json"
    lambada_data.write_bytes(lambada_response)
    lambada_config_path = tmp_path / "lambada-protocol.json"
    lambada_config_path.write_text(json.dumps(lambada_protocol(lambada_response)))
    lambada_report = tmp_path / "lambada-report.json"
    assert (
        lambada_main(
            [
                "evaluate",
                "--dataset-response",
                str(lambada_data),
                "--protocol",
                str(lambada_config_path),
                "--source",
                str(source),
                "--spec",
                str(spec),
                "--output",
                str(lambada_report),
            ]
        )
        == 0
    )
    capsys.readouterr()
    return source, spec, hellaswag_report, lambada_report


def test_cli_builds_and_verifies_three_axis_suite(tmp_path, capsys):
    source, spec, hellaswag, lambada = benchmark_reports(tmp_path, capsys)
    heldout = tmp_path / "heldout.txt"
    heldout.write_bytes(b"public heldout alpha\npublic heldout beta\n")
    output = tmp_path / "suite.json"
    arguments = [
        "build",
        "--source",
        str(source),
        "--spec",
        str(spec),
        "--heldout-data",
        str(heldout),
        "--hellaswag-report",
        str(hellaswag),
        "--lambada-report",
        str(lambada),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    receipt = json.loads(capsys.readouterr().out)
    suite = read_suite(output)
    assert receipt["evidence_sha256"] == suite["evidence_sha256"]
    assert receipt["claim_status"] == "complete_bounded_three_axis"
    assert set(receipt["evaluations"]) == {
        "heldout_likelihood",
        "hellaswag_validation",
        "lambada_last_word",
    }
    assert main(["verify", "--report", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == receipt
