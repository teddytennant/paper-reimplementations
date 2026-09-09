# jax-hero

JAX implementation of "Do Dynamic Routers Need Memory? HeRo: History-Aware Routing for Efficient LLM Inference" (arXiv:2609.08189).

## install

uv pip install -e .

## run

python -m hero.demo --layers 4 --tokens 8

This repo uses a tiny frozen decoder rather than a Llama backbone and does not report 8B numbers. The toy stack checks the memory, gate, and skip-loss identities from the paper.
