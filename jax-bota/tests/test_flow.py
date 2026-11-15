"""Flow-matching interpolant and L_BFM identity."""

import jax.numpy as jnp
import numpy as np
from jax import random

from jax_bota.flow import (
    bfm_loss,
    bfm_loss_from_vectors,
    cnf_objective,
    mlp_apply,
    mlp_init,
    path_interpolant,
    sample_interpolant,
)
from jax_bota.particles import euler_trajectories, straight_velocities


def _straight_paths():
    x0 = jnp.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.5]])
    x1 = jnp.array([[0.0, 2.0], [1.0, 2.0], [-1.0, 2.5]])
    t_steps = 4
    v = straight_velocities(x0, x1, t_steps)
    x = euler_trajectories(x0, v, 1.0 / t_steps)
    return x, v, x0, x1


def test_interpolant_matches_endpoints():
    x, v, x0, x1 = _straight_paths()
    xt0, ut0 = path_interpolant(x, v, jnp.asarray(0.0))
    xt1, ut1 = path_interpolant(x, v, jnp.asarray(1.0))
    np.testing.assert_allclose(np.array(xt0), np.array(x0), atol=1e-5)
    np.testing.assert_allclose(np.array(xt1), np.array(x1), atol=1e-5)
    np.testing.assert_allclose(np.array(ut0), np.array(v[0]), atol=1e-5)
    np.testing.assert_allclose(np.array(ut1), np.array(v[-1]), atol=1e-5)


def test_sample_interpolant_endpoints():
    x, v, x0, x1 = _straight_paths()
    n = x0.shape[0]
    idx = jnp.arange(n)
    xt0, ut0 = sample_interpolant(x, v, jnp.zeros((n,)), idx)
    xt1, ut1 = sample_interpolant(x, v, jnp.ones((n,)), idx)
    np.testing.assert_allclose(np.array(xt0), np.array(x0), atol=1e-5)
    np.testing.assert_allclose(np.array(xt1), np.array(x1), atol=1e-5)
    np.testing.assert_allclose(np.array(ut0), np.array(v[0]), atol=1e-5)
    np.testing.assert_allclose(np.array(ut1), np.array(v[-1]), atol=1e-5)


def test_bfm_zero_when_pred_equals_interpolant():
    x, v, _, _ = _straight_paths()
    n = x.shape[1]
    t = jnp.linspace(0.0, 1.0, 7)
    idx = jnp.arange(7) % n
    xt, ut = sample_interpolant(x, v, t, idx)
    loss = bfm_loss_from_vectors(ut, ut)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-7)

    def oracle(xt_, t_):
        return ut

    loss_fn = bfm_loss(oracle, x, v, t, idx)
    np.testing.assert_allclose(float(loss_fn), 0.0, atol=1e-7)


def test_mlp_shapes():
    key = random.PRNGKey(1)
    params = mlp_init(key, in_dim=3, hidden=8, out_dim=2, n_hidden=2)
    x = jnp.ones((5, 2))
    t = jnp.linspace(0.0, 1.0, 5)
    out = mlp_apply(params, x, t)
    assert out.shape == (5, 2)


def test_cnf_objective_finite():
    v = jnp.ones((4, 2))
    logp = jnp.zeros((4,))
    val = cnf_objective(v, logp, alpha=0.5, lam=0.1, log_p_terminal=logp)
    assert jnp.isfinite(val)
