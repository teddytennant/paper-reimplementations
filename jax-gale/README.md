# jax-gale

A JAX implementation of "GaLe: memory-efficient Global Approximate and Local
Exact features" (arXiv:2609.02689).

## install

    uv venv && uv pip install -e .

## run

    pytest -q

## what does not match the paper

No ImageNet numbers, no pretrained backbones, no Cortex-M33 timings; the 65%
speedup and the 90% RAM reduction are deployment figures this repo does not go
near. It implements the mechanism and checks its properties on small synthetic
inputs in float64. The alpha sweep finds an interior optimum, not 0.8.
