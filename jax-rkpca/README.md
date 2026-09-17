# jax-rkpca

JAX kernel of rotation-based online robust kernel PCA (arXiv:2609.15488).

## Install

```
pip install -e ".[tests]"
```

## Tests

```
pytest -q
```

## What this is

A small streaming implementation of Algorithm 1 from the paper: the empirical kernel map, half squared distance to the estimated subspace, a robust-weighted gradient step on the mean and basis, and Type I / Type II plane rotations applied to Gamma. The hot path never forms a dense M by M rotation; it applies the 2D plane rotation to the M by d basis in O(M d).

## What does not match the paper

- The paper cites Higuchi and Eguchi 1998 for the robust loss Psi but does not pick a specific function. This kernel uses Psi(z) = log(1 + z/c) with derivative Psi_dot(z) = 1/(c+z), so the influence weight falls as z grows.
- The paper never writes the gradient of the robust objective. From z = 1/2 ||(I-P)x||^2 with x = h - mu and P = Gamma Gamma^T this kernel uses dz/dmu = -(I-P)x and dz/dGamma = -x (Gamma^T x)^T, then multiplies by Psi_dot(z) and takes a Euclidean step mu <- mu - eta g_mu, Gamma <- Gamma - eta g_Gamma, then re-orthonormalizes.
- A rotation theoretically preserves orthonormality. This kernel still re-orthonormalizes after the rotation steps to limit floating-point drift.
- Dense A is built only for tiny tests that compare A Gamma against the optimized apply.
- Smoke and tests use small D, d and short streams, not the paper's D=100, d=10 figures.
- Variance-weighted Grassmann distance is implemented as sqrt(sum_i lambda_i theta_i^2 / sum_i lambda_i). The paper does not give an explicit formula.
