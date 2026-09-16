# Day 0 — Safe Repository Setup

**Date:** 2026-09-04
**Status:** Added setup day; this was not a curriculum day in the source workbook.

## Goal

Create a reproducible Python 3.11 repository before beginning the workbook plan.

## Work

1. Create and activate a virtual environment.
2. Install the declared dependencies from `requirements.txt`.
3. Run the tests and the Days 0–11 runner.
4. Run the deployment-readiness checker.
5. Keep all passwords, tokens, cookies, and keys outside Git and GitHub.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_days_00_11.py
python tools/deployment_readiness_check.py \
  --root . \
  --config config/deployment_readiness.json \
  --output /tmp/deployment-readiness.json
```

The readiness checker validates required files, directories, commands, and
configured environment-variable names. It never reads or reports secret values.

## Evidence

Code and automated checks are present. No credentials are required.
