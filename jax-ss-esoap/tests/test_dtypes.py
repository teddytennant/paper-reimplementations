"""The float32 path.

conftest.py turns on float64 for the suite, so everything else runs in double.
These build float32 arrays explicitly to check the single-precision path runs,
keeps its dtype, and tracks float64 where tracking is meaningful.

Where it is not meaningful, and why. ``L_t`` is a sum of ``t`` terms of rank at
most ``min(m, n)``, so on a wide parameter it is rank deficient for the first few
steps and its eigenbasis is not unique in the null space. Along those directions
both ``M`` and ``V`` are at the noise floor, and Adam's ``M / (sqrt(V) + eps)``
turns a 1e-7 float32 wobble into an order-one step component pointed somewhere
arbitrary. On the wide 24x16 problem float32 and float64 are already 6e-5 apart
by step 3. On a square 6x6 problem, where ``L_1`` is full rank immediately, they
agree to 1.4e-6 after 60 steps. That is the whole difference, and it is worth
knowing about before anyone runs this in float32 on a layer wider than it is
tall.
"""

import jax
import jax.numpy as jnp
import optax

import quadratic as qd
from ss_esoap import ss_esoap


def _run(x0, a, b, c, steps=120, lr=1e-2):
    opt = ss_esoap(lr)
    state = opt.init(x0)

    @jax.jit
    def step(x, state):
        g = jax.grad(qd.loss)(x, a, b, c)
        u, state = opt.update(g, state, x)
        return optax.apply_updates(x, u), state

    x = x0
    for _ in range(steps):
        x, state = step(x, state)
    return x


def _as_f32(arrays):
    return [jnp.asarray(v, jnp.float32) for v in arrays]


def test_float32_runs_and_preserves_dtype():
    a, b, c, x0, _ = qd.make(0)
    fa, fb, fc, fx0 = _as_f32((a, b, c, x0))
    x = _run(fx0, fa, fb, fc)
    assert x.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(x))
    assert float(qd.loss(x, fa, fb, fc)) < float(qd.loss(fx0, fa, fb, fc))


def test_float32_tracks_float64_when_the_factors_are_full_rank():
    """Square parameter, so ``L_1`` and ``R_1`` have no null space."""
    a, b, c, x0, _ = qd.make(0, kappa=2.0, m=6, n=6, p=6, q=6)
    fa, fb, fc, fx0 = _as_f32((a, b, c, x0))
    x64 = _run(x0, a, b, c, steps=60)
    x32 = _run(fx0, fa, fb, fc, steps=60)
    rel = float(jnp.linalg.norm(x32 - x64) / jnp.linalg.norm(x64))
    assert rel < 1e-4, rel


def test_float32_reaches_a_comparable_loss_on_the_wide_problem():
    """The trajectories separate, per the module docstring, but not in a way that
    stops it converging."""
    a, b, c, x0, _ = qd.make(0)
    fa, fb, fc, fx0 = _as_f32((a, b, c, x0))
    l64 = float(qd.loss(_run(x0, a, b, c), a, b, c))
    l32 = float(qd.loss(_run(fx0, fa, fb, fc), fa, fb, fc))
    assert l32 < 100.0 * l64


def test_state_dtypes_follow_the_parameters():
    p = {"w": jnp.zeros((5, 3), jnp.float32)}
    state = ss_esoap(1e-2).init(p)
    leaf = state[0].leaves["w"]
    for field in (leaf.mu, leaf.nu, leaf.left, leaf.right, leaf.ql, leaf.qr):
        assert field.dtype == jnp.float32
