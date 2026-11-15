JAX implementation of Differentially Private Average Treatment Effect Estimation by Propensity Score Blocking (https://arxiv.org/abs/2609.09536).

Implements IPW, BPS, AIPW, SeqIPW, and DPBlocking. It does not reproduce the paper's RMSE/bias tables, sklearn propensity fits, Gaussian-mechanism ablations, or comparisons to LGPM19 and LEBJ25.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke.py
```
