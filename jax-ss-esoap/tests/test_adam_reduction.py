"""The degenerate case: with the basis frozen at identity and self-scaling off,
Algorithm 2 is Adam, and it has to be Adam to the last bit."""

import jax
import jax.numpy as jnp
import optax
import pytest

from ss_esoap import ss_esoap

B1, B2, EPS, LR = 0.9, 0.999, 1e-8, 3e-3
STEPS = 60


def _params(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "w": jax.random.normal(k1, (6, 4)),
        "b": jax.random.normal(k2, (4,)),
        "conv": jax.random.normal(k3, (3, 3, 5)),  # folded to 9 x 5 by to_matrix
        "scalar": jax.random.normal(k4, ()),
    }


def _loss(p, target):
    return sum(
        jnp.sum(jnp.cosh(p[k] - target[k])) + 0.5 * jnp.sum(p[k] ** 2)
        for k in p
    )


def _trajectory(opt, params, target, steps=STEPS):
    state = opt.init(params)

    @jax.jit
    def step(params, state):
        g = jax.grad(_loss)(params, target)
        u, state = opt.update(g, state, params)
        return optax.apply_updates(params, u), state

    out = []
    for _ in range(steps):
        params, state = step(params, state)
        out.append(params)
    return out


def test_reduces_exactly_to_adam():
    key = jax.random.key(0)
    params = _params(key)
    target = _params(jax.random.key(1))

    degenerate = ss_esoap(
        LR, b1=B1, b2=B2, eps=EPS, self_scale=False, basis_updates=False
    )
    ours = _trajectory(degenerate, params, target)
    theirs = _trajectory(optax.adam(LR, b1=B1, b2=B2, eps=EPS), params, target)

    for step, (a, b) in enumerate(zip(ours, theirs)):
        for k in a:
            assert jnp.allclose(a[k], b[k], rtol=0, atol=1e-14), (step, k)


@pytest.mark.parametrize("flag", ["self_scale", "basis_updates"])
def test_either_addition_alone_breaks_the_equivalence(flag):
    """Guards against the equivalence test passing because the extra terms are
    dead code."""
    key = jax.random.key(0)
    params = _params(key)
    target = _params(jax.random.key(1))
    kw = {"self_scale": False, "basis_updates": False, flag: True}
    ours = _trajectory(ss_esoap(LR, b1=B1, b2=B2, eps=EPS, **kw), params, target)
    theirs = _trajectory(optax.adam(LR, b1=B1, b2=B2, eps=EPS), params, target)
    assert not jnp.allclose(ours[-1]["w"], theirs[-1]["w"], rtol=0, atol=1e-14)


def test_rank_one_and_scalar_leaves_take_the_adam_path():
    """Leaves with rank below two have no second Kronecker factor, so they run
    the Adam branch even with everything else switched on."""
    key = jax.random.key(2)
    params = _params(key)
    target = _params(jax.random.key(3))
    ours = _trajectory(ss_esoap(LR, b1=B1, b2=B2, eps=EPS), params, target)
    theirs = _trajectory(optax.adam(LR, b1=B1, b2=B2, eps=EPS), params, target)
    for k in ("b", "scalar"):
        assert jnp.allclose(ours[-1][k], theirs[-1][k], rtol=0, atol=1e-14)
    assert not jnp.allclose(ours[-1]["w"], theirs[-1]["w"], rtol=0, atol=1e-14)
