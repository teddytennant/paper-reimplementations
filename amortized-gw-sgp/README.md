JAX library for amortized Gromov-Wasserstein graph matching in supervised graph prediction (https://arxiv.org/abs/2609.15437).

Implements log-domain Sinkhorn on the Birkhoff polytope, the squared-Euclidean Gromov-Wasserstein objective J, a nested GW solver (degree-cost Sinkhorn warm start, then Sinkhorn-in-the-loop on the linearized GW cost), a parametric matcher with cost C_ij = ||phi(u_i) - psi(v_j)||^2, an MLP adjacency predictor, and joint Adam training of predictor plus matcher on a synthetic cycle / path / complete-bipartite task. Omega is the squared deviation of row and column sums of T from 1. Graphs are equal size; n = 6 on the toy task. Nested GW uses a fixed entropy temperature (no schedule). Under the tensor definition, J(P; A, P^T A P) = 0 for a permutation matrix P.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke.py
```

This does not match the paper on Mass-spectra-to-Scaffold, QM9, or FGW vs OT vs GW bake-off tables. The matcher is an MLP on adjacency rows, degree, and Laplacian PE rather than a Transformer decoder with a target GNN. The predictor is an MLP, not a modality-specific encoder. Edge discrepancy is squared Euclidean, not cross-entropy. There is no padding / FGW fusion / labeled-graph reconstruction.
