import json

import pytest

from ai_journey.scaling_cli import main
from ai_journey.scaling_io import load_scaling_report
from ai_journey.scaling_laws import ScalingLawError
from test_scaling_protocol import payload


def test_scaling_cli_builds_and_verifies_a_sealed_report(tmp_path, capsys):
    protocol = tmp_path / "protocol.json"
    report_path = tmp_path / "report.json"
    protocol.write_text(json.dumps(payload()))

    assert (
        main(["build", "--protocol", str(protocol), "--output", str(report_path)]) == 0
    )
    built = capsys.readouterr().out
    report = load_scaling_report(report_path)
    assert report["integrity"]["digest"] in built
    assert "parameters=100000000" in built
    assert "tokens=2000000000" in built
    assert "corpus_admitted=false" in built
    assert "measured=false" in built

    assert main(["verify", "--report", str(report_path)]) == 0
    verified = capsys.readouterr().out
    assert report["integrity"]["digest"] in verified
    assert "corpus_admitted=false" in verified


def test_scaling_cli_preserves_protocol_when_output_aliases_it(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload()))
    before = protocol.read_bytes()

    with pytest.raises(ScalingLawError, match="must not alias"):
        main(["build", "--protocol", str(protocol), "--output", str(protocol)])

    assert protocol.read_bytes() == before


def test_scaling_cli_fails_before_replacing_output_on_invalid_protocol(tmp_path):
    protocol = tmp_path / "protocol.json"
    output = tmp_path / "report.json"
    protocol.write_text("{}")
    output.write_bytes(b"protected\n")

    with pytest.raises(ScalingLawError, match="fields"):
        main(["build", "--protocol", str(protocol), "--output", str(output)])

    assert output.read_bytes() == b"protected\n"
