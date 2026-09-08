# jax-adaptive-spectral-bandwidth

JAX implementation of per-node Gaussian bandwidth selection by matching
kernel effective rank to MST intrinsic dimension, from "Geometry-Aware
Graph Construction via Adaptive Spectral Bandwidth Control" (arXiv:2609.03306).

## install

uv pip install -e .

## run

python -m pytest -q

Checks the spectral limits, the dimMST scaling estimator, and the J-criterion
on low-dimensional toys. Not a reproduction of the CIFAR-100 SSL tables.
