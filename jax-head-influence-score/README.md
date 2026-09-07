# jax-head-influence-score

A JAX implementation of the attention-head influence score from "Influence Score and Transformers interpretability:
Measure of the Effective Impact of Attention Heads at inference time" (arXiv:2609.05074).

## install

    uv venv
    uv pip install -e ".[test]"

## run

    python -m head_influence.demo
    python -m pytest

The paper's DeBERTa prompt-injection classifier and its dataset are both private, so none of its numbers are reproduced
here. The score is applied instead to a small transformer trained here on a synthetic task, which is enough to check the
exact residual decomposition, the ALTI normalizations, and what ablating the heads the score picks actually costs.
Section 5.1 states C_k in words rather than symbols; the reading used here is in the `layer_contributions` docstring.
