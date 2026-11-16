import pytest

from ai_journey.telemetry_metrics_model import TelemetryMetricsError
from ai_journey.telemetry_metrics_yaml import (
    MAX_DEPTH,
    MAX_YAML_BYTES,
    emit_yaml,
    read_yaml,
)

DOCUMENT = {
    "groups": [
        {
            "name": "lab_slo",
            "rules": [
                {
                    "record": "scope:lab:ratio",
                    "expr": 'sum by (scope) (x{outcome="ok"}) / sum by (scope) (x)',
                },
                {
                    "alert": "LabSlow",
                    "expr": "lab_seconds > 0.25",
                    "labels": {"scope": "local_cpu_lab", "slo": "lab-latency"},
                    "annotations": {"summary": "It's slow: 'quoted' # not a comment"},
                },
            ],
        },
        {"name": "other", "rules": [{"record": "a:b:c", "expr": "on"}]},
    ]
}

GOLDEN = """\
# Generated for a test.
#
groups:
  - name: 'lab_slo'
    rules:
      - record: 'scope:lab:ratio'
        expr: 'sum by (scope) (x{outcome="ok"}) / sum by (scope) (x)'
      - alert: 'LabSlow'
        expr: 'lab_seconds > 0.25'
        labels:
          scope: 'local_cpu_lab'
          slo: 'lab-latency'
        annotations:
          summary: 'It''s slow: ''quoted'' # not a comment'
  - name: 'other'
    rules:
      - record: 'a:b:c'
        expr: 'on'
"""


def test_emitter_writes_the_quoted_two_space_golden_document():
    assert emit_yaml(DOCUMENT, comments=("Generated for a test.", "")) == GOLDEN


def test_reader_returns_the_same_strings_in_the_same_order():
    document = read_yaml(GOLDEN)

    assert document == DOCUMENT
    rule = document["groups"][0]["rules"][1]
    assert list(rule) == ["alert", "expr", "labels", "annotations"]
    assert rule["annotations"]["summary"] == "It's slow: 'quoted' # not a comment"
    assert document["groups"][1]["rules"][0]["expr"] == "on"
    without_comments = GOLDEN.split("\n", 2)[2]
    assert emit_yaml(read_yaml(without_comments)) == without_comments


def test_scalars_keep_every_printable_character_and_the_empty_string():
    for value in ("", "'", "''", "a: b", "- x", "{[&*!|>%@`]}", " pad ", "naïve"):
        assert read_yaml(emit_yaml({"key": value})) == {"key": value}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("key: 'x'", "end with a line feed"),
        ("", "end with a line feed"),
        ("# only a comment\n", "no document"),
        ("key:\t'x'\n", "tabs"),
        ("key: 'x'\r\n", "CR"),
        ("\ufeffkey: 'x'\n", "BOM"),
        ("a: 'x'\n\nb: 'y'\n", "blank lines"),
        ("key: 'x' \n", "trailing spaces"),
        ("a:\n   b: 'x'\n", "odd indents"),
        ("a: 'x'\n  b: 'y'\n", "unexpected indentation"),
        ("a:\n    b: 'x'\n", "indented by two spaces"),
        ("a:\n- b: 'x'\n", "indented by two spaces"),
        ("a:\n", "needs a nested block"),
        ("a:\nb: 'x'\n", "indented by two spaces"),
        ("key: x\n", "single-quoted"),
        ('key: "x"\n', "single-quoted"),
        ("key: 'x' # note\n", "single-quoted"),
        ("key: 'it's'\n", "single-quoted"),
        ("key: 'x\n", "single-quoted"),
        ("key: '\n", "single-quoted"),
        ("key: {a: 'b'}\n", "single-quoted"),
        ("key: ['b']\n", "single-quoted"),
        ("key: &anchor 'b'\n", "single-quoted"),
        ("key: *anchor\n", "single-quoted"),
        ("key: !!str 'b'\n", "single-quoted"),
        ("key: |\n  text\n", "single-quoted"),
        ("key: '\x07'\n", "printable"),
        ("key:'x'\n", "one space"),
        ("---\nkey: 'x'\n", "plain key"),
        ("key: 'x'\n...\n", "plain key"),
        ("'key': 'x'\n", "plain key"),
        ("a-b: 'x'\n", "plain key"),
        ("a:\n  - 'x'\n", "plain key"),
        ("- a: 'x'\n", "block mapping"),
        ("a: 'x'\na: 'y'\n", "duplicate key"),
        ("a:\n  - b: 'x'\n    b: 'y'\n", "duplicate key"),
        ("a: 'x'\n# late\n", "precede the document"),
        ("#bad\na: 'x'\n", "precede the document"),
        ("a:\n  - b: 'x'\n- c: 'y'\n", "unexpected indentation"),
        ("a:\n  b: 'x'\n  - c: 'y'\n", "unexpected indentation"),
    ],
)
def test_reader_rejects_everything_outside_the_subset(text, message):
    with pytest.raises(TelemetryMetricsError, match=message):
        read_yaml(text)


def test_reader_and_emitter_bound_size_and_depth():
    deep = {"k": "v"}
    for _ in range(MAX_DEPTH + 1):
        deep = {"k": deep}
    with pytest.raises(TelemetryMetricsError, match="deeper"):
        emit_yaml(deep)
    text = "".join("  " * level + "k:\n" for level in range(MAX_DEPTH + 2))
    text += "  " * (MAX_DEPTH + 2) + "k: 'v'\n"
    with pytest.raises(TelemetryMetricsError, match="nested block"):
        read_yaml(text)
    with pytest.raises(TelemetryMetricsError, match="exceeds"):
        read_yaml("k: '" + "x" * MAX_YAML_BYTES + "'\n")
    with pytest.raises(TypeError):
        read_yaml(GOLDEN.encode())


@pytest.mark.parametrize(
    ("document", "error", "message"),
    [
        ({}, TelemetryMetricsError, "non-empty dicts"),
        ([], TelemetryMetricsError, "non-empty dicts"),
        ({"a": {}}, TelemetryMetricsError, "non-empty dicts"),
        ({"a": []}, TelemetryMetricsError, "must not be empty"),
        ({"a": ["x"]}, TelemetryMetricsError, "non-empty dicts"),
        ({"a-b": "x"}, TelemetryMetricsError, "plain identifiers"),
        ({1: "x"}, TelemetryMetricsError, "plain identifiers"),
        ({"a": "line\nbreak"}, TelemetryMetricsError, "printable"),
        ({"a": 1}, TypeError, "str, dict or list"),
        ({"a": None}, TypeError, "str, dict or list"),
    ],
)
def test_emitter_refuses_values_the_subset_cannot_carry(document, error, message):
    with pytest.raises(error, match=message):
        emit_yaml(document)
    with pytest.raises(TelemetryMetricsError, match="comments"):
        emit_yaml({"a": "x"}, comments=("bad\ncomment",))
