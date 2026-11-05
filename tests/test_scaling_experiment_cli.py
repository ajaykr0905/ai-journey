import json

import pytest

from ai_journey.scaling_experiment_cli import main
from ai_journey.scaling_laws import ScalingLawError
from test_scaling_experiment_verify import report


def protocol_payload():
    source = report()
    return {
        "protocol": source["protocol"],
        "inputs": source["inputs"],
        "observations": source["observations"],
        "target_compute": source["target"]["compute"],
        "policy": source["policy"],
    }


def test_cli_builds_and_independently_verifies_a_report(tmp_path, capsys):
    protocol = tmp_path / "protocol.json"
    output = tmp_path / "report.json"
    protocol.write_text(json.dumps(protocol_payload()))

    assert main(["build", "--protocol", str(protocol), "--output", str(output)]) == 0
    build_output = capsys.readouterr().out
    assert "execution_authenticated=false" in build_output
    assert main(["verify", "--report", str(output)]) == 0
    verify_output = capsys.readouterr().out
    assert "verified_scaling_experiment=" in verify_output
    assert "runs=12" in verify_output


def test_cli_protects_protocols_from_output_aliasing(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_payload()))

    with pytest.raises(ScalingLawError, match="must not alias"):
        main(["build", "--protocol", str(protocol), "--output", str(protocol)])
