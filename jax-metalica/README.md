METALICA replica-exchange metadynamics sampler for diffusion models, JAX, arXiv:2609.17823.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
pytest -q
```

The kernel is replica-exchange metadynamics over cosine VP-SDE levels on the
paper's 1D bimodal GMM toy (Appendix F.1), using the exact mixture score
(eq 25) rather than a learned network.

## What does not match the paper

- No protein unfolding, no MetaDiff / SMC comparison, no η=7 1000-iteration
  rare-event figure.
- Exact GMM score instead of a trained net (the paper does this for the toy).
- β(t) is the closed-form VP conversion β(t) = (π/(1+δ)) tan(π(t+δ)/(2(1+δ)))
  from a_t of the cosine schedule, with t clipped to `[0, 1-1e-3]` so tan
  stays finite. Appendix E only writes the EM kernels in terms of β_lo / β_hi.
- Toy hill height h is not given in the PDF (protein runs use h=1). Library
  default h=0.1. Paper F.1 uses γ=10, σ_hill=0.5, ρ=0.5, 321-point CV grid on
  [-8, 8], M=100; library defaults are the tiny test values γ=2, σ_G=0.2,
  41-point grid, M=8, N=8, K=3.
- MBAR ΔF^ξ uses a 1e-30 floor so the log ratio stays finite if a tiny run
  never visits one basin.

## Layout

`src/jax_metalica/` — `sde.py`, `gmm.py`, `kernels.py`, `bias.py`,
`replica.py`, `mbar.py`, `metalica.py`.
