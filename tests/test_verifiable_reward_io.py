import os

import pytest

from ai_journey.verifiable_reward_contract import VerifiableRewardError
from ai_journey.verifiable_reward_io import load_reward_report, write_reward_report


def test_report_write_is_canonical_and_round_trips(tmp_path):
    path = tmp_path / "nested" / "report.json"
    write_reward_report(path, {"b": 2, "a": 1})
    assert path.read_bytes() == b'{"a":1,"b":2}\n'
    assert load_reward_report(path) == {"a": 1, "b": 2}


def test_loader_rejects_duplicate_fields_and_symlinks(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"x":1,"x":2}')
    with pytest.raises(VerifiableRewardError, match="duplicate"):
        load_reward_report(path)
    link = tmp_path / "link.json"
    os.symlink(path, link)
    with pytest.raises(VerifiableRewardError, match="non-symlink"):
        load_reward_report(link)
