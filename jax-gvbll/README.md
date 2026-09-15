JAX implementation of online Bayesian node classification with a GraphSAGE encoder and variational Bayesian last layer (GVBLL / online GVBLL), https://arxiv.org/abs/2609.13655.

Implemented: two-layer GraphSAGE mean aggregator (no neighbor sampling) that encodes any graph size; fully factorized Gaussian last-layer weights W in R^{C x D} with reparameterization, closed-form KL to N(0, sigma0^2 I), Monte Carlo ELBO under a softmax categorical likelihood, optional linear KL annealing, MAP and MC predictive probabilities with entropy / max-prob uncertainty; and a test-time online Laplace update of the last-layer posterior with exponential forgetting and a KL anchor to the training posterior. Training uses in-house Adam on a tiny two-community synthetic graph.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke.py
```

What does not match the paper: no Cora/CiteSeer/ogbn-arxiv, no GraphGP/GGP-L/PGGP-L baselines, no Figure 3/4 tables; GraphSAGE only (no GCN/GAT); no per-node temperature MLP (softmax temperature is a scalar defaulting to 1); last-layer weights are stored as (C, D) rather than the paper's (d_e, C), which is a transpose of the same Gaussian; the KL is the exact factorized-Gaussian formula rather than the paper's proportional form that drops constants; the online step uses the paper's diagonal softmax Hessian p_c(1-p_c) h_j^2 and the stated power-prior / KL-anchor update, with default eta=1, beta=1, clip_delta=1, eps=1e-6 because those scalars are not numerically specified in the paper; no node-departure bookkeeping.
