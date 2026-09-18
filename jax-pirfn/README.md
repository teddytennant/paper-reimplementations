# jax-pirfn

Physics-informed random feature neural networks for PDEs in JAX (arXiv:2609.16406).

## Install

```bash
pip install -e ".[tests]"
```

## Run

```bash
pytest -q
python scripts/smoke.py
```

The paper's experiments are nonlinear Poisson in several dimensions, 2D Helmholtz with k=1, advection-diffusion, linear transport, and a 1D wave equation, trained in PyTorch with Adam then L-BFGS on thousands of points; this package uses tiny 1D Poisson, Helmholtz (kappa=8, not in the paper), and heat manufactured problems and trains only the readout with Adam (optional BFGS). Equation (3.4) uses un-normalized residual sums plus a leading ||c||_2^2 term with lambda_1=lambda_2=1; here the loss is mean-squared PDE/BC/IC residuals plus lambda ||c||_2^2 with default lambda=1e-6 (not given for these grids) and no L1 term. Numerical sections often treat sigma as std(omega); sampling here follows the spectral densities (Gaussian variance 1/sigma^4, Matern Student-t, Laplace tensor-product Cauchy). Cosine features follow (2.7) without the sqrt(2/N) kernel-map factor.
