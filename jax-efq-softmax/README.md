# jax-efq-softmax

JAX implementation of "EFQ-Softmax: Exp-Free Quantization for Softmax" (arXiv:2609.09721).

## install

uv pip install -e .

## run

python -m pytest -q

This is the score-to-E2M1 map and the online recurrence, not Qwen3-8B / WAN2.2 eval or the A5 kernel. Default (tau, h) are the paper's MMLU/Mean/Balance operating points.
