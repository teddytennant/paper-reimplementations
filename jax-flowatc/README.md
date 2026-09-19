# jax-flowatc

JAX implementation of FlowATC (arXiv:2609.16528).

https://arxiv.org/abs/2609.16528

## Install

```bash
pip install -e ".[tests]"
```

## Run

```bash
python -m pytest -q
python scripts/smoke.py
```

What does not match the paper: no 1.15M Bay Area ADS-B, no flow_small/large training, no DDPM/CVAE/LSTM bake-off, no 20.7M model, Euler steps and widths are tiny, synthetic trajectories only.
