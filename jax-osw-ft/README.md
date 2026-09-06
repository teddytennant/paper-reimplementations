# jax-osw-ft

JAX implementation of "Online Self-Weighted Fine-Tuning" (arXiv:2609.00734).

## install

uv pip install -e .

## run

python -m osw_ft.train --steps 500

The policy here is a tiny causal transformer trained on synthetic two-digit
addition, where a sampled answer is verified by exact match, not Qwen3 on AIME.
So none of the benchmark numbers in the paper are reproduced; what you get is
the method itself, running end to end on CPU. The tests are where the paper's
claims live: Theorem 1, Remark 1 and eq (11) are checked by enumerating the
binomial outcomes of K rollouts in exact rational arithmetic rather than by
sampling them.
