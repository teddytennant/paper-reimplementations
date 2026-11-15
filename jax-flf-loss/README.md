# jax-flf-loss

A JAX implementation of the Focal Log-Frequency Loss (f-loss) and its frequency-to-pixel schedule from "Balancing Frequencies and Pixels in Flow Matching" (arXiv:2609.02748).

## install

    uv venv
    uv pip install -e .

## run

    python -m flf.train

## what does not match the paper

The paper trains JiT-B/L/XL on ImageNet at 256 and 512 and reports FID, IS, and up to a
40% convergence speedup. None of that is here: no ImageNet, no FID, no speedup at their
model scales. The demo is a 0.26M-parameter MLP on 16x16 synthetic images with a 1/f^2
spectrum plus a few planted high-frequency modes, 900 steps on a CPU. The schedule's
sigmoid temperature is a free parameter; the paper gives only the switch point.
