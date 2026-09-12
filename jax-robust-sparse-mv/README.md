JAX implementation of robust sparse mean-variance portfolios (https://arxiv.org/abs/2609.11749).

This library implements problem (P1), the closed-form support subproblem (Proposition 1),
the dual multiplier (Proposition 2), brute-force enumeration over nonempty supports, and
the enumeration branch-and-bound of Section 5 with the extra pruning rule (Proposition 11).
It does not reproduce the CRSP/market tables or MISOCP timings.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke.py
```
