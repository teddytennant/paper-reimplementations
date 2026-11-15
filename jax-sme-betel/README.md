# jax-sme-betel

JAX implementation of "Robust Bayesian Inference for Unnormalized Models with Mixed-Domain Data"
(arXiv:2609.01783): score matching first-order conditions used as moment conditions in a Bayesian
exponentially tilted empirical likelihood.

## install

    uv venv
    uv pip install -e ".[test]"

## run

    .venv/bin/python -m pytest -q
    .venv/bin/python -m sme_betel.demo

## what is not reproduced

No ozone application, no 106-dimensional kernel convolution model, no PRD comparison. The preferential
sampling example is a five-parameter stand-in for the 76-parameter model in Section 3.1, linear trends
instead of low-rank Matern surfaces. The Section 4 coverage numbers do not come out: that moment
function decays to zero far from theta*, so E[g] gains extra roots in the tails and at n=50 the
posterior over-covers instead of landing near 0.92. Assumption 2 rules that out with a compact Theta,
which the paper never pins down. Calibration is checked on a misspecified Gaussian model instead.
