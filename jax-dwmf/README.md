JAX implementation of diagonally-weighted matrix-function dequantization (https://arxiv.org/abs/2609.10729).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
```

Implements Algorithm 1 / Theorem 16 (Lemma 15 heavy-index sampling, principal-block `f(γI + DAD)`, support-sparse `N` of Eq. (12)). The block function is exact dense `eigh`, not QSVT. The ORF smoke is a 1-D grid sampler for `w̃` in (E26), not the full `G^D` rejection procedure of Algorithm 2.
