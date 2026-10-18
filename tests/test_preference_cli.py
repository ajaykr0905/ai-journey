import json

import pytest

from ai_journey.preference_cli import CONFIG_BYTES, main, read_json, verify_main
from ai_journey.preference_data import PreferenceDataError
from ai_journey.preference_experiment import (
    PreferenceComparisonConfig,
    preference_config_dict,
)


def write_config(path):
    config = PreferenceComparisonConfig(fold_count=2, sft_steps=1, preference_steps=1)
    path.write_text(json.dumps(preference_config_dict(config)))


def test_cli_writes_and_exactly_replays_a_report(tmp_path, capsys):
    config = tmp_path / "config.json"
    report = tmp_path / "nested" / "report.json"
    write_config(config)

    assert main(["--config", str(config), "--output", str(report)]) == 0
    assert report.is_file()
    assert "4 seed-fold observations" in capsys.readouterr().out

    assert verify_main(["--report", str(report)]) == 0
    assert "16 trials" in capsys.readouterr().out


def test_cli_preserves_its_input_config(tmp_path, capsys):
    config = tmp_path / "config.json"
    write_config(config)
    before = config.read_bytes()

    assert main(["--config", str(config), "--output", str(config)]) == 2

    assert config.read_bytes() == before
    assert "must preserve" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    [
        '{"seeds":[44,45],"seeds":[46,47]}',
        '{"beta":NaN}',
        "\xff",
    ],
)
def test_bounded_json_reader_rejects_ambiguous_input(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_bytes(content.encode("latin-1"))

    with pytest.raises(PreferenceDataError):
        read_json(path, limit=CONFIG_BYTES)


def test_bounded_json_reader_rejects_oversized_input(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (CONFIG_BYTES + 1))

    with pytest.raises(PreferenceDataError, match="byte budget"):
        read_json(path, limit=CONFIG_BYTES)
