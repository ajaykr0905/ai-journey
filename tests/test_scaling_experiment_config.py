import json
from pathlib import Path

from ai_journey.scaling_experiment_cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_checked_reference_fixture_builds_without_measurement_claims(tmp_path):
    protocol = ROOT / "config/day-62-scaling-experiment-fixture.json"
    output = tmp_path / "report.json"

    assert main(["build", "--protocol", str(protocol), "--output", str(output)]) == 0

    report = json.loads(output.read_text())
    assert report["evidence_kind"] == "deterministic_reference_fixture"
    assert report["evidence_boundaries"]["observations_measured"] is False
    assert report["evidence_boundaries"]["execution_authenticated"] is False
    assert report["acceptance"]["admitted"] is True
