"""The efficiency claim, counted rather than timed.

Wall-clock on a shared machine is noisy and would make a flaky assertion, so
these tests count the matrix factorizations each algorithm performs and the
order of each one. That is also the quantity the paper's argument is actually
about: "the E-step often requires inverting the covariance matrix of the
observed component for each unique missingness pattern".

One caveat, stated plainly. To batch observations with different patterns under
vmap, the dense E-step here solves a padded pq x pq system whose missing block
is the real m x m one, so it does not realize the flop saving that the counts
describe; `exact_estep_by_pattern` does the unpadded version and is checked
against it in test_conditional.py. What the counts below establish is which
factorizations the algorithms call for, not how fast this particular layout runs.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import matnorm_pem as mp
from matnorm_pem import opcount, synth
from matnorm_pem.structural import fit_matnorm_structural, structural_estep


def _mcar(p, q, n, rate, seed=0):
    params = synth.paper_params(p, q)
    Y = mp.sample_matnorm(jr.key(seed), params, n)
    mask = synth.mcar_mask(jr.key(seed + 400), n, p, q, rate)
    return params, Y, mask


def _count(fn):
    opcount.reset()
    try:
        fn()
        return {
            "estep": opcount.total("estep"),
            "estep_max_dim": opcount.max_dim("estep"),
            "estep_work": opcount.work("estep"),
            "mstep_max_dim": opcount.max_dim("mstep"),
        }
    finally:
        opcount.disable()


@pytest.mark.parametrize("rate", [0.25, 0.5])
def test_partial_em_factorizes_nothing_the_mstep_did_not_already_need(rate):
    """The point of Section 3.2: no pattern-specific inversion at all.

    Exact EM takes one factorization per observation per iteration, of order m,
    which for these settings is well above both p and q. The partial E-step
    takes none; its largest factorization is the q x q one the M-step performs
    anyway.
    """
    p, q, n, iters = 4, 5, 60, 5
    _, Y, mask = _mcar(p, q, n, rate)
    ex = _count(lambda: mp.fit_matnorm(Y, mask, method="exact", n_iter=iters, track=False))
    pe = _count(lambda: mp.fit_matnorm(Y, mask, method="partial", n_iter=iters, track=False))
    assert ex["estep"] == n * iters
    assert ex["estep_max_dim"] > max(p, q)
    assert pe["estep"] == 0
    assert pe["mstep_max_dim"] == max(p, q)
    assert ex["estep_work"] > 20 * pe["estep_work"] + 1000


def test_exact_estep_cost_grows_with_the_missing_rate():
    """m^3 per observation, so the benchmark gets worse exactly where it hurts."""
    p, q, n = 4, 5, 60
    work = []
    for rate in [0.2, 0.4, 0.8]:
        _, Y, mask = _mcar(p, q, n, rate)
        work.append(_count(lambda: mp.exact_estep(Y, mask, synth.paper_params(p, q)))["estep_work"])
    assert work[0] < work[1] < work[2]
    assert work[2] > 20 * work[0]


def test_structural_estep_splits_one_big_factorization_into_two_small_ones():
    """Section 3.4 replaces an order |R||C| inverse with orders |R| and |C|.

    With a 3 x 4 block inside a 5 x 5 matrix that is 12^3 = 1728 per observation
    against 3^3 + 4^3 = 91, and no factorization is larger than p or q.
    """
    p, q, nr, nc, n = 5, 5, 3, 4, 30
    params = synth.paper_params(p, q)
    Y = mp.sample_matnorm(jr.key(1), params, n)
    arow, acol = synth.rect_indicators(jr.key(2), n, p, q, nr, nc)
    mask = jnp.einsum("np,nq->npq", arow, acol) > 0

    dense = _count(lambda: mp.exact_estep(Y, mask, params))
    block = _count(lambda: structural_estep(Y, arow, acol, params, mode="block"))

    assert dense["estep"] == n
    assert dense["estep_max_dim"] == nr * nc
    assert block["estep"] == 2 * n
    assert block["estep_max_dim"] == max(nr, nc)
    assert block["estep_work"] * 15 < dense["estep_work"]


def test_structural_em_keeps_every_factorization_small():
    """Over a whole fit, nothing bigger than p or q is ever factorized."""
    p, q, n = 5, 5, 60
    params = synth.paper_params(p, q)
    Y = mp.sample_matnorm(jr.key(3), params, n)
    arow, acol = synth.rect_indicators(jr.key(4), n, p, q, 3, 4)
    counts = _count(
        lambda: fit_matnorm_structural(Y, arow, acol, n_iter=5, mode="block", track=False)
    )
    assert counts["estep_max_dim"] <= max(p, q)
    assert counts["mstep_max_dim"] <= max(p, q)
