# jax-stratified-allocation

A JAX implementation of the design-based theory in "Optimal Stratified Allocation for
Rare-Event Onset Forecasting in Dependent Sequences" (arXiv:2609.04420): the exact
finite-population variance of a class-weighted risk estimator, the Neyman-optimal
allocation it implies, and the parameter-free efficiency prediction A(pi, f).

## install

    uv venv
    uv pip install -e ".[test]"
    .venv/bin/python -m pytest -q

## run

    .venv/bin/python -m stratified_allocation.demo

Prints Table 2 of the paper, then the simulated variance ratio at three design points.
x64 is switched on at import, because Proposition 6 is a difference of two nearly equal
variances and Corollary 10 is a ratio of them.

## what is not here

Only Section 6 is implemented: Propositions 4, 5, 6, 7, 11, 13 and 15, Theorem 8 and
Corollaries 9, 10 and 12. The empirical study is not reproduced at all. That needs the
350-name US equity panel, the Phillips-Shi-Yu date-stamping procedure that produces the
labels, and the boosted detector fitted on the subsamples, none of which ship with the
paper. Lemma 1 and Lemma 3, the measurability and mixing arguments of Section 6.1, are
statements about a purge gap rather than anything to compute, so they are quoted in
docstrings and not implemented. A(pi, f) drops the finite-population corrections, and
where the truncation binds it understates the true variance ratio by a factor of a few;
the tests say where that happens instead of hiding it.
