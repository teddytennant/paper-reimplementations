# jax-dstmm

A JAX implementation of "Deep Skew-t Mixture Models" (arXiv:2609.00773): the GHST pathway construction, its conditional GIG law, and the stochastic/Monte Carlo EM estimator. Equation numbers are in the docstrings.

## install

    uv venv && uv pip install -e . && uv pip install pytest scipy

## run

    python -m dstmm.demo
    pytest -q

## what is not here

No UCI handwritten-digit or Gas Sensor Array Drift results, and no reproduction of the paper's BIC architecture search at its architectures. `dstmm.simulate` keeps the Section 4.1 skewness mechanism, including the E(W) recentring, but shrinks it to p = 6 and (r1, r2) = (3, 2) so the tests finish on a CPU; the paper runs p = 20, (r1, r2) = (5, 2), n = 1000 and 1000 replications. The skewness gain reproduces at this scale, about +0.09 nats per held-out point at kappa = 2 and nu = 5, but the absolute ARIs in Table 1 do not transfer.

JAX has no Bessel K, so `dstmm.bessel` computes log K_nu by trapezoid quadrature of its integral representation, accurate to about 1e-13 against scipy; d/dnu log K_nu, which equation (22) needs, comes from autodiff of that quadrature.
