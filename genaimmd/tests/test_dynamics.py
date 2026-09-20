"""Overdamped Langevin, two-way shooting, HAB, and uniform acceptance (2)."""

import jax
import jax.numpy as jnp
import numpy as np

from genaimmd.dynamics import (
    euler_maruyama,
    shooting_acceptance,
    two_way_shoot,
)
from genaimmd.potential import MIN_A, MIN_B, in_A


def test_langevin_stays_near_minimum():
    key = jax.random.PRNGKey(1)
    n_traj, n_steps = 16, 200
    x = jnp.broadcast_to(MIN_B, (n_traj, 2))
    keys = jax.random.split(key, n_traj * n_steps)

    def one_traj(x0, ks):
        def body(x, k):
            return euler_maruyama(x, k), None

        x_end, _ = jax.lax.scan(body, x0, ks)
        return x_end

    ks = keys.reshape(n_traj, n_steps, 2)
    ends = jax.vmap(one_traj)(x, ks)
    disp = jnp.linalg.norm(ends - MIN_B, axis=-1)
    # Well-to-saddle distance is ~1.3; free Brownian RMS at t=n_steps*dt is ~2.8.
    assert float(jnp.median(disp)) < 1.2
    assert float(jnp.mean(disp)) < 2.0
    # Have not crossed into basin A on average.
    assert float(jnp.mean(ends[:, 0])) > 0.0


def test_twoway_from_A_hab_zero():
    key = jax.random.PRNGKey(2)
    x = jnp.stack([MIN_A, MIN_A + jnp.array([0.05, -0.05])])
    keys = jax.random.split(key, x.shape[0])
    result = jax.vmap(lambda xi, ki: two_way_shoot(xi, ki, max_steps=40))(x, keys)
    hab = result["hab"]
    assert jnp.all(hab == 0)
    assert bool(in_A(MIN_A))


def test_uniform_shooting_acceptance_eq2():
    acc_long = shooting_acceptance(L=10.0, L_prime=5.0, hab_prime=1.0)
    acc_short = shooting_acceptance(L=5.0, L_prime=10.0, hab_prime=1.0)
    acc_reject = shooting_acceptance(L=10.0, L_prime=5.0, hab_prime=0.0)
    np.testing.assert_allclose(float(acc_long), 1.0)
    np.testing.assert_allclose(float(acc_short), 0.5)
    np.testing.assert_allclose(float(acc_reject), 0.0)
