# jax-token-geometry

A JAX build of the controlled simulation in "Learning Representations through Token
Prediction: Geometry, Approximation, and Downstream Guarantees" (arXiv:2608.30072), with
the paper's bounds asserted as numerical inequalities against exactly enumerated
population quantities.

## install

    uv venv
    uv pip install -e ".[test]"

## run

    .venv/bin/python -m pytest

## what does not match the paper

This is the Appendix B.1 process at 16 token types rather than 48, and a numerical check
of the theorems, not a language model trained on text. Most checks use the unrestricted
representation of Section 2.2 fitted on the exact population objective; a small
transformer trained for a couple of thousand steps never reaches a prediction error where
the error terms mean anything, so it is used only where the recurrent-depth claims need
it. Where a bound cannot be made informative at this scale the tests say so. Theorem 5's
certified affinity ratio is vacuous here, so the bound itself is checked on planted
embeddings, and Theorem 6's constants are not explicit in the paper, so the scaling in n
and in prediction error is tested instead of a constant.
