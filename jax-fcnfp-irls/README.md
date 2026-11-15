JAX implementation of support-discovery IRLS for single-commodity fixed-charge network flow (https://arxiv.org/abs/2609.09295).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
```

What does not match the paper: no 410-instance bake-off vs MILP / ADSSP / DSSP; no Layer 2 perturbation restarts and no Layer 3 anchor-union restricted IRLS; Newton systems use a dense linear solve plus a tiny ridge rather than approximate-Cholesky Laplacian PCG, and there is no primal-dual IPM fallback; Layer 1 uses one or two feasible starts rather than five complementary initializations; no official code wrapping (there is none to wrap).
