JAX implementation of the two-step Fourier functional linear regression estimator for noisy regular-grid covariates (https://arxiv.org/abs/2609.08671).

Implemented: discrete Fourier projection reconstruction (paper eqs. 3–4), least-squares slope in trigonometric spaces S_m with Gram Φ_m and the s_n invertibility cutoff, model selection by pen(m)=κ D_m σ²/n with the paper's calibrated κ=4.01 (and the plug-in σ̂_m²=γ_n(β̂_m) variant), inner-product prediction ⟨β̂, X⟩.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no meteorological data, no 25 figures, no Monte-Carlo Γ-norm tables, no p_sim=10000 generation grid. Penalty uses the calibrated product κ=θ(1+δ)=4.01 rather than separate θ>4 and δ>0. Eigenvalue cutoff exponent α defaults to 5 (paper: any α>0 in the definition, theorems need α>2a; simulations do not specify α). At n=40–80 the theoretical bound D_N < n/ln² n is 1 or 3; smoke/tests may pass a larger reconstruction dimension (as in the paper's reconstruction figure, D=11). Discrete Fourier coefficients coincide with L² inner products for trigonometric polynomials when D<p; they are Riemann sums otherwise. Birgé–Massart slope heuristic is not used for choosing m; the paper selects m by the explicit penalty after calibrating κ once.
