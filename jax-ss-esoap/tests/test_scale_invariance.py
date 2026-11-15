"""Gradient-scale behaviour.

The paper does not claim an invariance, and the interesting result is that
self-scaling breaks the one SOAP has. Adam and SOAP are both invariant to a
constant rescaling of the gradient stream once ``eps = 0``: the Kronecker factors
pick up ``c^2``, which leaves the eigenbasis alone, momentum picks up ``c``, the
second moment picks up ``c^2``, and ``M / sqrt(V)`` cancels.

Equation (3) does not cancel, and the exponent is worse than it looks. The
numerator ``Tr(Y^T S)`` is linear in the gradient. In the denominator ``L`` and
``R`` are each second moments, so each carries ``c^2``, and they multiply:
``Tr(S^T L^-1 S R^-1)`` goes as ``c^-4``. So ``tau`` scales as ``c^5`` and the
step picks up ``c^{-5/2}``. That is a property of the method as specified, not an
implementation artifact, and it is why the floor ``tau_min`` ends up mattering:
a tenfold drop in gradient magnitude moves ``tau`` by five orders of magnitude.

Everything runs on a fixed synthetic gradient and parameter sequence rather than
a live loss, so the two runs differ in exactly one thing.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from ss_esoap._core import Config, eigh_basis, init_leaf, secant_scale, update_leaf

SHAPE = (10, 7)
STEPS = 25
# A power of two, so the rescaling is exact in binary floating point and any
# difference we see is the algorithm, not rounding.
C = 8.0


def _stream(seed=0):
    rng = np.random.default_rng(seed)
    grads = [jnp.asarray(rng.standard_normal(SHAPE)) for _ in range(STEPS)]
    params = [jnp.asarray(rng.standard_normal(SHAPE)) for _ in range(STEPS)]
    return grads, params


def _replay(cfg, grads, params, scale=1.0):
    state = init_leaf(params[0], cfg.eps, cfg.max_precond_dim)
    ups = []
    for t, (g, p) in enumerate(zip(grads, params)):
        u, state = update_leaf(scale * g, state, p, jnp.asarray(t), cfg)
        ups.append(u)
    return ups


def test_soap_style_preconditioning_is_gradient_scale_invariant():
    cfg = Config(eps=0.0, self_scale=False)
    grads, params = _stream()
    base = _replay(cfg, grads, params)
    scaled = _replay(cfg, grads, params, scale=C)
    for t, (a, b) in enumerate(zip(base, scaled)):
        assert jnp.allclose(a, b, rtol=1e-10, atol=1e-12), t


def test_self_scaling_breaks_that_invariance():
    cfg = Config(eps=0.0, self_scale=True, tau_min=0.0)
    grads, params = _stream()
    base = _replay(cfg, grads, params)
    scaled = _replay(cfg, grads, params, scale=C)
    diffs = [
        float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a))
        for a, b in zip(base, scaled)
    ]
    assert max(diffs) > 0.1


def _spd(rng, d, scale):
    m = rng.standard_normal((d, d))
    return jnp.asarray(scale * (np.eye(d) + 0.2 * m @ m.T / d))


@pytest.mark.parametrize("scale", [0.25, 2.0, 4.0])
def test_tau_scales_as_the_fifth_power_of_the_gradient_scale(scale):
    """``c_k`` goes as ``c``, ``a_k`` goes as ``c^-4``, so ``tau`` goes as ``c^5``.

    The Kronecker factors are small enough here that ``tau`` stays well under 1
    at every scale tested, so the ``min(1, .)`` clip is out of the way and the
    cube law is visible.
    """
    rng = np.random.default_rng(5)
    s = jnp.asarray(rng.standard_normal(SHAPE))
    y = jnp.asarray(rng.standard_normal(SHAPE))
    y = jnp.abs(jnp.sum(y * s)) / jnp.sum(s * s) * s  # keep c_k > 0
    ell = _spd(rng, SHAPE[0], 1e-3)
    r = _spd(rng, SHAPE[1], 1e-3)

    def tau_of(c):
        # L and R are second moments of the gradient, so they carry c^2.
        el, rr = c**2 * ell, c**2 * r
        ql, qr = eigh_basis(el), eigh_basis(rr)
        dl = jnp.diagonal(ql.T @ el @ ql)
        dr = jnp.diagonal(qr.T @ rr @ qr)
        return secant_scale(s, c * y, ql, qr, dl, dr, 0.0, 0.0)

    assert float(tau_of(1.0)) < 1e-3 and float(tau_of(scale)) < 1.0
    ratio = float(tau_of(scale) / tau_of(1.0))
    assert ratio == pytest.approx(scale**5, rel=1e-8)


def test_orthogonal_reparameterization_leaves_the_step_direction_alone():
    """A rotation of the row and column spaces rotates the update the same way.

    ``L`` and ``R`` transform by congruence, the eigenbasis absorbs the rotation,
    and the eigenspace step is unchanged; the Kronecker structure is what buys
    this and a diagonal preconditioner does not have it.
    """
    # warmup_steps = -1 so the basis is refreshed from step 0. While Q is still
    # the identity the eigenspace step is a sign pattern, and sign() is not
    # equivariant under a rotation; the property is about the eigenbasis, not
    # about the identity basis.
    cfg = Config(eps=0.0, self_scale=False, trigger=0.0, warmup_steps=-1)
    grads, params = _stream(seed=3)
    rng = np.random.default_rng(9)
    ul, _ = np.linalg.qr(rng.standard_normal((SHAPE[0], SHAPE[0])))
    ur, _ = np.linalg.qr(rng.standard_normal((SHAPE[1], SHAPE[1])))
    ul, ur = jnp.asarray(ul), jnp.asarray(ur)

    base = _replay(cfg, grads, params)
    rot = _replay(
        cfg,
        [ul @ g @ ur.T for g in grads],
        [ul @ p @ ur.T for p in params],
    )
    # L_t is a sum of t rank-min(m, n) terms, so it is rank deficient for the
    # first couple of steps, and the eigenbasis of a rank-deficient factor is not
    # unique. Compare once L and R have filled out.
    warm = 2
    for t, (a, b) in enumerate(zip(base[warm:], rot[warm:]), start=warm):
        assert jnp.allclose(ul @ a @ ur.T, b, rtol=1e-8, atol=1e-9), t
