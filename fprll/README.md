fprll is a JAX implementation of restarted follow-the-perturbed-regularized-lazy-leader with a density-mixture master for online linear optimization on the box [-1, 1]^d under indicator switching costs (https://arxiv.org/abs/2609.30556).

pip install -e .

JAX_PLATFORMS=cpu python -m pytest -q

JAX_PLATFORMS=cpu python demo.py

The paper's regret bound is the claim. This repository checks the algorithm's mechanics on a short horizon, not the minimax rate.

Dimension 1 evaluates the closed-form density on a uniform grid of 1024 points. Total variation is the midpoint sum 0.5 * sum |p - q| * dx. The short-horizon comparison allows an additive slack of 8 on the mixture's indicator-augmented regret versus the best single scale, averaged over seeds 0 through 7. On that instance the averaged excess is about 4.2.

The perturbation is Lap(mu) with mu = laplace_scale (default 1). Its density is (2*mu)^{-d} exp(-||p||_1 / mu).

What does not match the paper: the leader uses (1/gamma) times r(x) = (sigma/2) ||x||^2 - sum log(1 - x_i^2), with one gamma, one sigma, and one laplace_scale on every scale instead of the paper's per-block tuning of gamma, mu, and sigma. The high-dimensional sampled surrogate is not used. Signed linear losses are shifted by the gradient bound before ell = g/M, with M = 2 ||g||_1 + lambda, because the paper's reduction to [0, 1] assumes nonnegative losses. D = lambda/M is passed to DM-multiscale even when it is below 1. The constructive writeup assumes D >= 1 and only cites a reduction for the smaller case. Standalone restarted experts draw a fresh action at each block boundary.
