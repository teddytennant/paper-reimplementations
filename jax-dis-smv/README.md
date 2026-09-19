# jax-dis-smv

JAX kernel of disintegrated PAC-Bayes self-bounding majority votes (arXiv:2609.16803).

Paper: https://arxiv.org/abs/2609.16803

## Install

```bash
pip install -e ".[tests]"
```

## Run

```bash
python -m pytest -q
python scripts/smoke.py
```

## What does not match the paper

- Missing UCI/MNIST.
- No SMV-Exact/MC bake-off.
- Tiny stump voters on synthetic 2D binary points.
- No data-dependent forest voters.
