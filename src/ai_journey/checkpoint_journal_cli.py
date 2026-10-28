"""Inspect an existing journal's byte integrity and durable retention budgets."""

import argparse
import json
from pathlib import Path

from .checkpoint_journal import CheckpointJournal, MAX_TOTAL_BYTES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--max-versions", type=int, default=16)
    parser.add_argument("--max-total-bytes", type=int, default=MAX_TOTAL_BYTES)
    arguments = parser.parse_args(argv)
    try:
        journal = CheckpointJournal(
            arguments.directory,
            run_sha256=arguments.run_sha256,
            max_versions=arguments.max_versions,
            max_total_bytes=arguments.max_total_bytes,
            create=False,
        )
        result = journal.inspect()
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
