# jax-muon-c

JAX implementation of "Muon-C: Operator-Aligned Muon for Convolutional Kernels" (arXiv:2609.09676).

## install

uv pip install -e .

## run

python -m pytest -q

This repo ships the critical-grid Muon-C update and the paper's geometry checks, not CIFAR-10 flow matching or FID. Newton-Schulz is the 5-step complex iteration from Appendix B, not a trained U-Net.
