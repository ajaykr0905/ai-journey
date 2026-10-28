"""Public MoE plans must fit the bounded canonical operator wire format."""

import json
from pathlib import Path

import pytest

from ai_journey.moe_ddp_plan import (
    public_moe_ddp_plan_from_mapping,
    read_public_moe_ddp_plan,
)


def _mapping():
    return public_moe_ddp_plan_from_mapping(
        json.loads(Path("config/moe-ddp-cpu.json").read_text())
    ).to_mapping()


def _wire_bytes(mapping):
    return json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _mapping_at_size(size):
    mapping = _mapping()
    mapping["source"]["name"] = "🙂" * 4096
    mapping["source"]["revision"] = ""
    # A supplementary-plane character occupies twelve ASCII JSON escape bytes.
    escaped, ascii_padding = divmod(size - len(_wire_bytes(mapping)), 12)
    mapping["source"]["revision"] = "🙂" * escaped + "a" * ascii_padding
    assert 0 < len(mapping["source"]["revision"]) <= 4096
    assert len(_wire_bytes(mapping)) == size
    return mapping


def test_plan_rejects_escape_expansion_beyond_its_wire_budget():
    mapping = _mapping()
    mapping["source"]["name"] = "🙂" * 4096
    mapping["source"]["revision"] = "🙂" * 4096

    with pytest.raises(ValueError):
        public_moe_ddp_plan_from_mapping(mapping)


@pytest.mark.parametrize(
    "metadata",
    [{}, {"name": "Public café 数据 🙂", "revision": "révision-α"}],
)
def test_admitted_ascii_and_unicode_plans_roundtrip_through_reader(tmp_path, metadata):
    mapping = _mapping()
    mapping["source"].update(metadata)
    plan = public_moe_ddp_plan_from_mapping(mapping)
    config = tmp_path / "plan.json"
    config.write_bytes(_wire_bytes(plan.to_mapping()))

    actual, source = read_public_moe_ddp_plan(
        config, Path("data/day-19-demo-names.txt")
    )

    assert actual.to_mapping() == mapping
    assert actual.fingerprint() == plan.fingerprint()
    assert source == Path("data/day-19-demo-names.txt").read_bytes()


@pytest.mark.parametrize("size", [65535, 65536])
def test_canonical_plans_at_wire_limit_roundtrip(tmp_path, size):
    mapping = _mapping_at_size(size)
    plan = public_moe_ddp_plan_from_mapping(mapping)
    config = tmp_path / "plan.json"
    config.write_bytes(_wire_bytes(plan.to_mapping()))

    actual, _ = read_public_moe_ddp_plan(config, Path("data/day-19-demo-names.txt"))

    assert actual.to_mapping() == mapping
    assert actual.fingerprint() == plan.fingerprint()


def test_plan_one_canonical_byte_above_wire_limit_is_rejected():
    with pytest.raises(ValueError):
        public_moe_ddp_plan_from_mapping(_mapping_at_size(65537))


def test_reader_bounds_supplied_formatting_before_reading_source(tmp_path):
    plan = public_moe_ddp_plan_from_mapping(_mapping())
    config = tmp_path / "plan.json"
    config.write_bytes(b" " * 65537 + _wire_bytes(plan.to_mapping()))

    with pytest.raises(ValueError):
        read_public_moe_ddp_plan(config, tmp_path / "nonexistent-source.txt")
