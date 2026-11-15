# jax-kalman-delta

JAX implementation of Kalman Delta Networks from "Kalman Delta Networks: Uncertainty-aware Associative Memory" (arXiv:2609.07816).

## install

uv pip install -e .

## run

python -m pytest -q

This is the filter and the diagonal/isotropic scans, not the 750M/1.3B FineWeb pretraining. Authors' training code is PyTorch at github.com/ngocbh/kalman-delta-networks. No chunkwise WY kernel.
