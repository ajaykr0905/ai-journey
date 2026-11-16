"""Benchmark files must be bounded, unambiguous, and protected from output aliases."""

import math
import os

import pytest

from ai_journey.hellaswag import HellaSwagError
from ai_journey.hellaswag_cli import main
from ai_journey.hellaswag_io import read_evaluation_bytes, parse_evaluation_json
from test_hellaswag_modern import evaluation_files


@pytest.mark.parametrize(
    "raw", [b'{"a":1,"a":2}', b'{"value":NaN}', b'{"value":Infinity}']
)
def test_ambiguous_or_nonfinite_json_is_rejected(raw):
    with pytest.raises(HellaSwagError, match="duplicate|finite"):
        parse_evaluation_json(raw, name="fixture")


@pytest.mark.parametrize(
    ("raw", "literal"),
    [
        (b'{"value": 1e400}', "1e400"),
        (b'{"value": -1e400}', "-1e400"),
        (b'{"value": [1, [2.5, 1e400]]}', "1e400"),
        (b'{"value": {"inner": {"deep": -1e400}}}', "-1e400"),
        (b'{"value": NaN}', "NaN"),
        (b'{"value": Infinity}', "Infinity"),
        (b'{"value": -Infinity}', "-Infinity"),
    ],
)
def test_overflowing_or_nonfinite_numbers_are_rejected_by_literal(raw, literal):
    with pytest.raises(HellaSwagError, match="finite JSON numbers") as error:
        parse_evaluation_json(raw, name="fixture")
    assert str(error.value).endswith(f": {literal}")


def test_finite_edge_numbers_parse_unchanged():
    payload = parse_evaluation_json(
        b'{"tiny": 5e-324, "huge": 1.7976931348623157e308, "zero": -0.0}',
        name="fixture",
    )
    assert payload == {"tiny": 5e-324, "huge": 1.7976931348623157e308, "zero": 0.0}
    assert math.copysign(1.0, payload["zero"]) == -1.0


def test_reader_rejects_oversized_symlink_and_fifo_inputs(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b"1234")
    with pytest.raises(HellaSwagError, match="constraints"):
        read_evaluation_bytes(path, maximum=3, name="fixture")
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(HellaSwagError, match="constraints"):
        read_evaluation_bytes(link, maximum=8, name="fixture")
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(HellaSwagError, match="constraints"):
        read_evaluation_bytes(fifo, maximum=8, name="fixture")


@pytest.mark.parametrize(
    "target", ["source", "spec", "response", "protocol", "hardlink"]
)
def test_cli_refuses_to_overwrite_any_input_before_model_loading(
    tmp_path, monkeypatch, target
):
    _, _, source, spec, response, protocol = evaluation_files(tmp_path)
    outputs = {
        "source": source,
        "spec": spec,
        "response": response,
        "protocol": protocol,
    }
    output = outputs.get(target, tmp_path / "alias.json")
    if target == "hardlink":
        os.link(source, output)
    originals = {path: path.read_bytes() for path in (source, spec, response, protocol)}

    def unexpected(*args, **kwargs):
        raise AssertionError("model construction must not occur")

    monkeypatch.setattr("ai_journey.hellaswag_cli._load_model", unexpected)
    with pytest.raises(SystemExit):
        main(
            [
                "evaluate",
                "--source",
                str(source),
                "--spec",
                str(spec),
                "--dataset-response",
                str(response),
                "--protocol",
                str(protocol),
                "--output",
                str(output),
            ]
        )
    assert all(path.read_bytes() == content for path, content in originals.items())
