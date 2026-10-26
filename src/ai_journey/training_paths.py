"""Protect declared training inputs and durable outputs from alias collisions."""

from pathlib import Path


def validate_training_paths(
    *, source: Path, spec: Path, checkpoint: Path, report: Path
):
    """Reject output aliases before any training or artifact publication occurs."""
    paths = {"source": source, "spec": spec, "checkpoint": checkpoint, "report": report}
    if any(not isinstance(path, Path) for path in paths.values()):
        raise TypeError("training paths must be pathlib.Path values")
    for name in ("checkpoint", "report"):
        output = paths[name]
        if output.is_symlink() or (output.exists() and not output.is_file()):
            raise ValueError(f"{name} output must be a regular non-symlink path")
        for other_name, other in paths.items():
            if other_name == name:
                continue
            if output.resolve() == other.resolve() or (
                output.exists() and other.exists() and output.samefile(other)
            ):
                raise ValueError(f"{name} output aliases {other_name}")
