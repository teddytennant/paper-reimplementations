JAX kernel of GA-AMNO operator-readable allocation and gauge-aware transport (https://arxiv.org/abs/2609.15620).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no Navier-Stokes/Darcy/KS, no spectral residual head, no importance-modulated readout / coarse-grid upsampling / differential correction (Table 2 steps 9–12), η_θ/ψ_θ are tiny MLPs not the paper's conv predictor, Jacobian estimator is ours, no MMPDE baseline.
