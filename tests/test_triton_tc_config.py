"""Tests of the declared A100 and H100 inputs and of their strict loader.

The values are datasheet and architecture-page inputs, never measurements.
Each test changes one field of the committed config and expects the loader
to refuse it.
"""

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from ai_journey.triton_tc_config import (
    PATHS,
    config_sha256,
    load_config,
    parse_config,
    read_config,
)
from ai_journey.triton_tc_kernel import TritonTcError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "day-82-triton-tc.json"


def raw():
    return json.loads(CONFIG.read_text(encoding="ascii"))


def a100(value):
    return value["devices"][0]


def h100(value):
    return value["devices"][1]


def test_the_committed_config_declares_every_input_with_a_source():
    payload, config = read_config(CONFIG)

    assert config.to_mapping() == json.loads(payload)
    assert config_sha256(payload) == config_sha256(CONFIG.read_bytes())
    assert config.sizes == (512, 1024, 2048, 4096, 8192)
    assert [device.target for device in config.devices] == ["sm_80", "sm_90"]
    urls = {source.key: source.url for source in config.sources}
    assert urls == {
        "a100-product": "https://www.nvidia.com/en-us/data-center/a100/",
        "ampere-blog": (
            "https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/"
        ),
        "h100-product": "https://www.nvidia.com/en-us/data-center/h100/",
        "hopper-blog": (
            "https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/"
        ),
    }
    statuses = [
        item.status
        for device in config.devices
        for item in (
            device.sm_count,
            device.memory_bandwidth,
            *(path.fma_per_sm_per_clock for path in device.paths),
            *(path.listed_tflops for path in device.paths),
        )
    ]
    assert set(statuses) == {"declared"}
    assert config.device("sm_90").boost_clock_mhz.status == "not_stated"
    assert config.device("sm_80").boost_clock_mhz.value == 1410


def test_the_per_sm_rates_and_dense_peaks_are_the_declared_values():
    config = load_config(CONFIG)
    rows = {
        device.key: [(path.name, path.rate, path.dense_tflops) for path in device.paths]
        for device in config.devices
    }
    assert rows == {
        "a100-sxm4-80gb": [
            ("fp32", 64, Fraction("19.5")),
            ("tf32", 512, Fraction(156)),
            ("fp16", 1024, Fraction(312)),
            ("bf16", 1024, Fraction(312)),
        ],
        "h100-sxm5-80gb": [
            ("fp32", 128, Fraction(67)),
            ("tf32", 1024, Fraction("494.5")),
            ("fp16", 2048, Fraction("989.5")),
            ("bf16", 2048, Fraction("989.5")),
        ],
    }


def test_the_a100_clock_reproduces_every_listed_peak_and_h100_has_no_clock():
    config = load_config(CONFIG)
    device = config.device("sm_80")
    for name in PATHS:
        check = device.reproduction(name)
        assert check["within_rounding"], name
        assert check["derived_flops_per_second"] == (
            2 * device.path(name).rate * 108 * 1410 * 10**6
        )
    assert config.device("sm_90").reproduction("bf16") is None
    clocks = {
        name: round(float(config.device("sm_90").implied_clock_mhz(name)), 1)
        for name in PATHS
    }
    # The H100 FP32 peak and its Tensor Core peaks imply different clocks.
    assert clocks == {"fp32": 1982.7, "tf32": 1829.2, "fp16": 1830.1, "bf16": 1830.1}


def edits():
    def drop(value):
        del a100(value)["paths"]["bf16"]

    def extra(value):
        h100(value)["sm_count"]["measured"] = True

    def status(value):
        h100(value)["memory_bandwidth_gb_per_second"]["status"] = "measured"

    def unstated(value):
        a100(value)["sm_count"]["status"] = "not_stated"

    return [
        (drop, "paths fields mismatch"),
        (extra, "fields mismatch"),
        (status, "status must be one of"),
        (unstated, "only a not_stated one has none"),
        (lambda v: a100(v)["sm_count"].update(value=True), "sm_count must be"),
        (
            lambda v: a100(v)["memory_bandwidth_gb_per_second"].update(value="1935"),
            "Day 71 pinned device",
        ),
        (
            lambda v: a100(v)["paths"]["fp16"]["fma_per_sm_per_clock"].update(
                value=1000
            ),
            "do not reproduce the listed peak",
        ),
        (
            lambda v: h100(v)["paths"]["fp32"]["listed_tflops"].update(
                with_sparsity=True
            ),
            "FP32 FMA figure has no sparsity",
        ),
        (lambda v: h100(v).update(source="h100"), "fields mismatch"),
        (
            lambda v: h100(v)["sm_count"].update(source="wikipedia"),
            "sources that the table lacks",
        ),
        (
            lambda v: v["sources"]["a100-product"].update(url="http://nvidia.com/"),
            "url must be https",
        ),
        (lambda v: v.update(devices=v["devices"][::-1]), "in order"),
        (lambda v: v.update(output_bytes=2), "output_bytes must be 4"),
        (lambda v: v.update(sizes=[1024, 512]), "increasing integers"),
        (lambda v: v["sparsity"].update(factor=4), "sparsity factor"),
        (lambda v: v.update(schema_version=2), "schema_version"),
        (
            lambda v: a100(v)["memory_bandwidth_gb_per_second"].update(value="2,039"),
            "canonical decimal",
        ),
    ]


@pytest.mark.parametrize(("edit", "message"), edits())
def test_the_loader_refuses_a_config_that_breaks_one_rule(edit, message, tmp_path):
    value = copy.deepcopy(raw())
    edit(value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TritonTcError, match=message):
        load_config(path)


def test_the_loader_refuses_duplicate_keys_and_non_objects(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"name": "a", "name": "b"}', encoding="ascii")
    with pytest.raises(TritonTcError, match="duplicate JSON key"):
        load_config(path)
    with pytest.raises(TritonTcError, match="must be a JSON object"):
        parse_config([])
