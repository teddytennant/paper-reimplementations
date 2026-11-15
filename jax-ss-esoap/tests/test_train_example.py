"""The 1D Poisson demo in ss_esoap.train."""

import jax
import jax.numpy as jnp
import numpy as np

from ss_esoap import soap, ss_esoap, train


def test_hard_constraint_holds_at_the_boundary():
    p = train.init_params(jax.random.key(0))
    assert float(abs(train.u(p, 0.0))) < 1e-12
    assert float(abs(train.u(p, 1.0))) < 1e-12


def test_residual_vanishes_on_the_exact_solution():
    """A direct check that ``source`` and ``residual`` describe the same PDE."""
    xs = jnp.linspace(0.05, 0.95, 17)
    uxx = jax.vmap(jax.grad(jax.grad(train.exact)))(xs)
    assert jnp.allclose(-uxx - train.source(xs), 0.0, atol=1e-8)


def test_pinn_loss_drops_by_orders_of_magnitude():
    xs = jnp.linspace(0.0, 1.0, 128)
    p0 = train.init_params(jax.random.key(0))
    # Measured drops at 300 steps from a starting residual near 3.9e3:
    # about 360x for SS-ESOAP and about 4100x for the SOAP baseline. The bar is
    # 50x, which is far enough below both to survive a backend change.
    for opt in (ss_esoap(1e-2), soap(1e-2, precondition_frequency=50)):
        _, losses = train.run(opt, p0, xs, steps=300)
        assert np.isfinite(losses[-1])
        assert losses[-1] < losses[0] / 50.0
