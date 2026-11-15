# jax-intcf

JAX kernel of interpretable causal forests (IntCF) from [arXiv:2609.16971](https://arxiv.org/abs/2609.16971).

## Install

```bash
pip install -e ".[tests]"
```

## Run

```bash
pytest -q
python scripts/smoke.py
```

## What does not match the paper

- Tiny default depth and number of trees.
- No GRF/DML bake-off.
- No real medical data.
- Population change-point theory is not numerically reproduced at N->inf.
