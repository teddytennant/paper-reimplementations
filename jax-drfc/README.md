# jax-drfc

JAX kernel of Decision-Relevant Fresh Comparison for decentralized heterogeneous bandits.

https://arxiv.org/abs/2609.16824

## Install

```bash
pip install -e ".[tests]"
```

## Run

```bash
python -m pytest -q
python scripts/smoke.py
```

CPU only.

Radius `β(W) = C sqrt(log(K^2 T^3 / δ) / (N h B))` uses natural log and `C = 8` (paper: `C ≤ 8`). DRFC-Seq anytime radius uses `a0 = 1`, `C0 = 1` (paper: `a0 = Θ(c1)` from Howard et al.).

## What does not match the paper

No MovieLens-1M, no 90-page gossip/flooding graph, complete graph or N=2 shared memory, tiny T/K, no CUSUM-Reset bake-off.
