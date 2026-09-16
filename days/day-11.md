# Day 11 — Python Fluency Drills

**Date:** 2026-09-15

The lecture takeaway is that next-token prediction is the central training
objective; model serving, tools, safety, and product behavior are engineering
layers built around it.

`src/ai_journey/drills.py` includes executable examples of:

- a filtered list comprehension;
- a lazy running-total generator;
- a signature-preserving, typed decorator;
- typed parameters and return values.

## Virtual-environment drill

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
deactivate
```

Use `python -m pip` so installation targets the currently active interpreter. Do
not commit `.venv` or local `.env` files.

## Videos

- Primary — *Large Language Models explained briefly*: https://www.youtube.com/watch?v=LPZh9BOjkQs
- Support — *Ch 10 — Cross products*: https://www.youtube.com/watch?v=eu6i7WJeinw

## Evidence

The drills and tests are present. Running the virtual-environment sequence and
watching the lectures remain pending user confirmation.
