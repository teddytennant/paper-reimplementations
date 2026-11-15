# ic-tester

Query-access testers for coordinate-wise monotonicity and DSIC of discrete single-parameter allocation mechanisms (arXiv:2609.17406).

## Install

```bash
pip install -e ".[tests]"
```

Requires Python 3.10+ and numpy. Tests need pytest.

## Run

```bash
pytest -q
```

## What does not match the paper

- no lower-bound constructions / query-complexity experiments from the paper
- Algorithm 4 (the 𝑑^{O(m)} better-query 1D tester) is not implemented
- default bid distribution is uniform, not an arbitrary unknown D
- concrete repeat constant C=8 in Alg 1 (paper writes O(n log m / ε))
