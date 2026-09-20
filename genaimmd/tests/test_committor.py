"""AIMMD committor MLP and loss (eqs 3, 6)."""

import jax
import jax.numpy as jnp
import numpy as np

from genaimmd.committor import aimmd_loss, aimmd_loss_from_logits, init_committor, pB, q
from genaimmd.potential import MIN_A, MIN_B


def test_aimmd_loss_all_B_prefers_positive_q():
    s = -jnp.ones((12,))
    q_pos = jnp.full((12,), 6.0)
    q_neg = jnp.full((12,), -6.0)
    loss_pos = aimmd_loss_from_logits(q_pos, s)
    loss_neg = aimmd_loss_from_logits(q_neg, s)
    assert float(loss_pos) < float(loss_neg)


def test_pb_in_unit_interval_and_sigmoid():
    params = init_committor(jax.random.PRNGKey(0))
    x = jnp.stack([MIN_A, MIN_B, jnp.zeros(2)])
    logits = q(params, x)
    pb = pB(params, x)
    np.testing.assert_allclose(np.asarray(pb), 1.0 / (1.0 + np.exp(-np.asarray(logits))), atol=1e-6)
    assert jnp.all((pb > 0.0) & (pb < 1.0))


def test_aimmd_loss_matches_eq6_on_mlp():
    params = init_committor(jax.random.PRNGKey(3))
    x = jnp.stack([MIN_A, MIN_B])
    s = jnp.array([[1.0, 1.0], [-1.0, -1.0]])
    logits = q(params, x)
    direct = aimmd_loss_from_logits(logits, s)
    via = aimmd_loss(params, x, s)
    np.testing.assert_allclose(float(direct), float(via), atol=1e-5)
