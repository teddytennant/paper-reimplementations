# jax-hllm

JAX implementation of "hLLM: Single Pass Decoding for Generative Reranking"
(arXiv:2609.01807).

## install

uv pip install -e .

## run

python -m hllm.train --steps 500

This is the scoring head plus the assignment decoder on synthetic data. The LLM
backbone is an input, not something the repo builds; the heads take per-item
hidden states and stop there. The LoRA distillation from a real teacher is not
included either, so the teacher permutations come from a fixed linear scoring
rule instead of a 32B ranker, and the ablations in Tables 3 and 4 don't
transfer. The Hungarian solver is written out in numpy rather than called from
scipy, which costs speed and buys a file you can read next to equation (1).
None of the paper's latency numbers are reproduced, neither the 28 ms end to
end nor the 64x speedup.
