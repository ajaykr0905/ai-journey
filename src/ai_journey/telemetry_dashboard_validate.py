"""Strict validation of generated Grafana dashboard JSON.

The validator accepts the dashboard-JSON subset ``build_dashboard`` emits and
reads it without trusting the generator: exactly the known top-level fields,
the stated ``schemaVersion``, a null ``id`` for import and a Grafana-safe
``uid``; unique positive panel IDs and unique titles; known panel types with
their exact fields; ``gridPos`` rectangles inside the 24-column grid that do
not overlap; every query panel and target naming the Prometheus datasource
through a declared ``datasource`` template variable; unique uppercase
``refId`` letters per panel; and PromQL that passes ``check_promql`` against
the metric catalog's series. The dashboard must be tagged and titled as local
CPU lab telemetry and carry a text panel that states it.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from .telemetry_dashboard_grafana import DASHBOARD_SCHEMA_VERSION, GRID_COLUMNS
from .telemetry_dashboard_metrics import SERIES
from .telemetry_dashboard_promql import PromqlSyntaxError, check_promql
from .telemetry_traces_io import TelemetryVerificationError

QUERY_PANEL_TYPES = frozenset({"stat", "bargauge", "timeseries"})
UNITS = frozenset({"s", "ms", "ns", "short", "none", "percentunit"})
LOCAL_CPU_LAB_TAG = "local-cpu-lab"
MAX_PANELS = 64
_TOP_LEVEL = frozenset(
    {
        "annotations",
        "description",
        "editable",
        "graphTooltip",
        "id",
        "links",
        "panels",
        "refresh",
        "schemaVersion",
        "tags",
        "templating",
        "time",
        "timepicker",
        "timezone",
        "title",
        "uid",
        "version",
    }
)
_PANEL_BASE = frozenset({"id", "type", "title", "description", "gridPos", "options"})
_QUERY_PANEL = _PANEL_BASE | {"datasource", "fieldConfig", "targets"}
_TARGET = frozenset({"refId", "expr", "legendFormat", "datasource"})
_UID = re.compile(r"[A-Za-z0-9_-]{1,40}")
_TITLE = re.compile(r"[!-~](?:[ -~]{0,126}[!-~])?")
_VARIABLE_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REF_ID = re.compile(r"[A-Z]")


class DashboardValidationError(TelemetryVerificationError):
    """Raised when dashboard JSON breaks the generated-dashboard contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DashboardValidationError(message)


def _integer(value: object, minimum: int, maximum: int, where: str) -> int:
    _require(
        type(value) is int and minimum <= value <= maximum,
        f"{where} must be an integer in [{minimum}, {maximum}]",
    )
    return value


@dataclass(frozen=True)
class DashboardSummary:
    """What a validated dashboard holds, for a sealed report to bind."""

    uid: str
    schema_version: int
    panels: int
    targets: int
    series: tuple[str, ...]
    functions: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "schema_version": self.schema_version,
            "panels": self.panels,
            "targets": self.targets,
            "series": list(self.series),
            "functions": list(self.functions),
        }


def _datasources(templating: object) -> dict[str, str]:
    _require(
        type(templating) is dict and set(templating) == {"list"},
        "templating must hold exactly a variable list",
    )
    variables = templating["list"]
    _require(type(variables) is list, "templating.list must be a list")
    plugins: dict[str, str] = {}
    for index, variable in enumerate(variables):
        where = f"templating.list[{index}]"
        _require(type(variable) is dict, f"{where} must be an object")
        name = variable.get("name")
        _require(
            type(name) is str
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None,
            f"{where} needs a variable name",
        )
        _require(name not in plugins, f"{where} repeats variable {name}")
        _require(
            variable.get("type") == "datasource",
            f"{where} must be a datasource variable",
        )
        plugin = variable.get("query")
        _require(type(plugin) is str and bool(plugin), f"{where} needs a plugin query")
        plugins[name] = plugin
    return plugins


def _datasource(value: object, plugins: dict[str, str], where: str) -> dict[str, str]:
    _require(
        type(value) is dict and set(value) == {"type", "uid"},
        f"{where} must be a datasource reference",
    )
    match = (
        _VARIABLE_REFERENCE.fullmatch(value["uid"])
        if type(value["uid"]) is str
        else None
    )
    _require(
        match is not None and match.group(1) in plugins,
        f"{where} must reference a declared datasource variable",
    )
    _require(
        value["type"] == "prometheus" and plugins[match.group(1)] == "prometheus",
        f"{where} must be a Prometheus datasource",
    )
    return value


def _grid(value: object, where: str) -> tuple[int, int, int, int]:
    _require(
        type(value) is dict and set(value) == {"x", "y", "w", "h"},
        f"{where} must hold exactly x, y, w and h",
    )
    x = _integer(value["x"], 0, GRID_COLUMNS - 1, f"{where}.x")
    w = _integer(value["w"], 1, GRID_COLUMNS, f"{where}.w")
    _require(x + w <= GRID_COLUMNS, f"{where} extends past {GRID_COLUMNS} columns")
    y = _integer(value["y"], 0, 10_000, f"{where}.y")
    h = _integer(value["h"], 1, 100, f"{where}.h")
    return x, y, w, h


def _overlaps(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    (ax, ay, aw, ah), (bx, by, bw, bh) = first, second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def validate_dashboard(
    dashboard: object, *, allowed_series: Collection[str] = SERIES
) -> DashboardSummary:
    """Check one dashboard object and summarize the queries it runs."""

    _require(type(dashboard) is dict, "dashboard must be a JSON object")
    missing = sorted(_TOP_LEVEL - set(dashboard))
    unknown = sorted(set(dashboard) - _TOP_LEVEL)
    _require(not missing, f"dashboard lacks {', '.join(missing)}")
    _require(not unknown, f"dashboard has unknown field {', '.join(unknown)}")
    _require(
        dashboard["schemaVersion"] == DASHBOARD_SCHEMA_VERSION
        and type(dashboard["schemaVersion"]) is int,
        f"schemaVersion must be {DASHBOARD_SCHEMA_VERSION}",
    )
    _require(dashboard["id"] is None, "a dashboard for import has a null id")
    uid = dashboard["uid"]
    _require(type(uid) is str and _UID.fullmatch(uid) is not None, "uid is invalid")
    title = dashboard["title"]
    _require(
        type(title) is str
        and _TITLE.fullmatch(title) is not None
        and "local CPU lab" in title,
        "title must name the dashboard as local CPU lab telemetry",
    )
    tags = dashboard["tags"]
    _require(
        type(tags) is list
        and all(type(tag) is str for tag in tags)
        and len(set(tags)) == len(tags)
        and LOCAL_CPU_LAB_TAG in tags,
        f"tags must be unique strings including {LOCAL_CPU_LAB_TAG}",
    )
    _integer(dashboard["version"], 1, 2**31 - 1, "version")
    _integer(dashboard["graphTooltip"], 0, 2, "graphTooltip")
    _require(type(dashboard["editable"]) is bool, "editable must be a boolean")
    _require(
        dashboard["annotations"] == {"list": []} and dashboard["links"] == [],
        "annotations and links must be empty",
    )
    time = dashboard["time"]
    _require(
        type(time) is dict
        and set(time) == {"from", "to"}
        and all(type(value) is str and value for value in time.values()),
        "time must hold from and to strings",
    )
    plugins = _datasources(dashboard["templating"])
    panels = dashboard["panels"]
    _require(
        type(panels) is list and 1 <= len(panels) <= MAX_PANELS,
        f"panels must be a list of 1 to {MAX_PANELS} panels",
    )
    ids: set[int] = set()
    titles: set[str] = set()
    grids: list[tuple[int, tuple[int, int, int, int]]] = []
    series: set[str] = set()
    functions: set[str] = set()
    targets = 0
    headed = False
    for index, panel in enumerate(panels):
        where = f"panels[{index}]"
        _require(type(panel) is dict, f"{where} must be an object")
        kind = panel.get("type")
        _require(
            kind == "text" or kind in QUERY_PANEL_TYPES,
            f"{where} has unknown type {kind}",
        )
        expected = _PANEL_BASE if kind == "text" else _QUERY_PANEL
        _require(set(panel) == expected, f"{where} fields do not match a {kind} panel")
        panel_id = _integer(panel["id"], 1, 2**31 - 1, f"{where}.id")
        _require(panel_id not in ids, f"{where} repeats panel id {panel_id}")
        ids.add(panel_id)
        _require(
            type(panel["title"]) is str
            and _TITLE.fullmatch(panel["title"]) is not None
            and panel["title"] not in titles,
            f"{where} needs a unique printable title",
        )
        titles.add(panel["title"])
        _require(type(panel["description"]) is str, f"{where}.description must be text")
        grid = _grid(panel["gridPos"], f"{where}.gridPos")
        for other, placed in grids:
            _require(not _overlaps(grid, placed), f"{where} overlaps panel {other}")
        grids.append((panel_id, grid))
        options = panel["options"]
        _require(type(options) is dict, f"{where}.options must be an object")
        if kind == "text":
            content = options.get("content")
            _require(
                options.get("mode") == "markdown" and type(content) is str,
                f"{where} must be a markdown text panel",
            )
            headed = headed or "Local CPU lab telemetry" in content
            continue
        source = _datasource(panel["datasource"], plugins, f"{where}.datasource")
        field_config = panel["fieldConfig"]
        _require(
            type(field_config) is dict
            and set(field_config) == {"defaults", "overrides"}
            and type(field_config["defaults"]) is dict
            and field_config["defaults"].get("unit") in UNITS
            and field_config["overrides"] == [],
            f"{where}.fieldConfig needs a known unit and no overrides",
        )
        queries = panel["targets"]
        _require(
            type(queries) is list and 1 <= len(queries) <= 26,
            f"{where}.targets must list 1 to 26 queries",
        )
        references: set[str] = set()
        for position, target in enumerate(queries):
            label = f"{where}.targets[{position}]"
            _require(
                type(target) is dict and set(target) == _TARGET,
                f"{label} fields do not match a Prometheus target",
            )
            reference = target["refId"]
            _require(
                type(reference) is str
                and _REF_ID.fullmatch(reference) is not None
                and reference not in references,
                f"{label} needs a unique uppercase refId",
            )
            references.add(reference)
            _require(
                _datasource(target["datasource"], plugins, f"{label}.datasource")
                == source,
                f"{label} queries another datasource than its panel",
            )
            _require(
                type(target["legendFormat"]) is str, f"{label} legend must be text"
            )
            try:
                summary = check_promql(target["expr"], allowed_metrics=allowed_series)
            except PromqlSyntaxError as error:
                raise DashboardValidationError(f"{label}.expr: {error}") from error
            series.update(summary.metrics)
            functions.update(summary.functions)
            targets += 1
    _require(headed, "a text panel must state that this is local CPU lab telemetry")
    return DashboardSummary(
        uid=uid,
        schema_version=DASHBOARD_SCHEMA_VERSION,
        panels=len(panels),
        targets=targets,
        series=tuple(sorted(series)),
        functions=tuple(sorted(functions)),
    )
