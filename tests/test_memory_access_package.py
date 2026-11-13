import importlib
import tomllib
from pathlib import Path

import pytest

from ai_journey import memory_access_cli

ROOT = Path(__file__).resolve().parents[1]


def test_console_script_resolves_to_the_memory_access_cli(capsys):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry = project["project"]["scripts"]["ai-journey-memory-access"]

    assert entry == "ai_journey.memory_access_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is memory_access_cli.main
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith("usage: ai-journey-memory-access ")
    assert "{analyze,benchmark,verify}" in usage
