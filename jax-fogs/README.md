# jax-fogs

JAX implementation of Fractional-Order OLS Grid-Search (FO-GS) for identifying discrete-time fractional-order LTI systems from one trajectory.

https://arxiv.org/abs/2609.18127

## Install
pip install -e ".[tests]"

## Run
python -m pytest -q
python scripts/smoke.py

CPU only.

Grid points are `jnp.linspace(low, high, M)` on the closed interval (paper Algorithm 1 uses interior points α̲ + kε, k=1..M). Row OLS uses ridge λ=1e-6 on XXᵀ by default, as in the paper's synthetic experiments.

## What does not match the paper
No EEG mental-arithmetic dataset, no FO-BS/FO-WT bake-off, no theorems. Tiny synthetic trajectories. Grid is jnp.linspace (paper Alg.1 uses α̲ + kε, k=1..M). Ridge λ=1e-6 as in the paper's synthetic appendix. α=1 uses binomial weights because Γ(−1) is a pole. `fogs_commensurate` is extra; Algorithm 1 is the heterogeneous row-wise search.
