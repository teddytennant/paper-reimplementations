JAX/NumPy closed-form estimator of LLM-judge anchor contamination ρ under a single-common-factor model (https://arxiv.org/abs/2609.08826).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no HANNA real-panel replay, no SEM/ML comparison, no ordinal mixed-scale estimator, diagnostic thresholds not the paper's 1000-seed 95th percentiles.
