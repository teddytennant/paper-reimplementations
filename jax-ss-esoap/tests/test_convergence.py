"""Does the thing actually optimize.

Section 5.2's matrix quadratic, shrunk. Its Hessian ``(B B^T) kron (A^T A)`` is
exactly Kronecker, so this is the setting the Kronecker metric is entitled to do
well in, and the paper uses it for the same reason.

Every optimizer gets the same learning-rate grid and keeps its best run per seed,
so the comparison is not decided by which one happened to like the single rate
that got picked. The assertion is on the median over seeds; a single-seed A-beats-B
is a coin flip.
"""

import jax
import numpy as np
import optax
import pytest

import quadratic as qd
from ss_esoap import soap, ss_esoap

SEEDS = range(5)
STEPS = 300
LRS = (3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def _final_loss(opt, x0, a, b, c, steps=STEPS):
    state = opt.init(x0)

    @jax.jit
    def step(x, state):
        g = jax.grad(qd.loss)(x, a, b, c)
        u, state = opt.update(g, state, x)
        return optax.apply_updates(x, u), state

    x = x0
    for _ in range(steps):
        x, state = step(x, state)
    out = float(qd.loss(x, a, b, c))
    return out if np.isfinite(out) else np.inf


def _best_over_lrs(make, seed):
    a, b, c, x0, _ = qd.make(seed)
    return min(_final_loss(make(lr), x0, a, b, c) for lr in LRS)


@pytest.fixture(scope="module")
def results():
    makes = {
        "adam": optax.adam,
        "soap": lambda lr: soap(lr, precondition_frequency=10),
        "ss_esoap": ss_esoap,
    }
    return {
        name: [_best_over_lrs(make, s) for s in SEEDS]
        for name, make in makes.items()
    }


def test_beats_adam_by_a_wide_median_margin(results):
    """Measured 34x to 119x per seed, median about 45x. The bar is 8x on the
    median and 4x on every individual seed, which leaves room for the ordering
    to wobble on a different backend without the assertion becoming a lottery."""
    adam, ours = results["adam"], results["ss_esoap"]
    assert np.median(adam) / np.median(ours) > 8.0
    per_seed = [a / o for a, o in zip(adam, ours)]
    assert min(per_seed) > 4.0, per_seed


def test_beats_the_soap_baseline(results):
    """The SOAP baseline here is the one Section 2.4 describes, fixed-period
    refresh with the second moment carried through the similarity transform.
    That transition is what the gap is mostly measuring."""
    assert np.median(results["soap"]) / np.median(results["ss_esoap"]) > 10.0


def test_reaches_a_small_absolute_residual(results):
    """An A-beats-B ratio is worthless if both are stuck, so pin the absolute
    level too. Measured 2e-5 to 5e-5 from a starting loss above 1e2."""
    assert max(results["ss_esoap"]) < 1e-3


def test_converges_toward_the_known_optimum():
    a, b, c, x0, _ = qd.make(0)
    before = float(qd.loss(x0, a, b, c))
    after = _final_loss(ss_esoap(1e-2), x0, a, b, c, steps=400)
    assert after < 1e-6 * before
