"""Theorem 6: projected gradient descent driven by gradients that are only
accurate along its own trajectory still lands within the stated additive gap."""

import jax
import jax.numpy as jnp
import pytest

from ugtl.theory import (
    BoxQuadratic,
    adversarial_gradient_error,
    alpha_pgd,
    pgd_step_size,
    projected_gradient_descent,
)


def random_instance(seed, d=6):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    a = jax.random.uniform(k1, (d,), minval=0.5, maxval=2.0)
    b = jax.random.normal(k2, (d,))
    f = BoxQuadratic(a, b, -jnp.ones(d), jnp.ones(d))
    x1 = jax.random.uniform(k3, (d,), minval=-1.0, maxval=1.0)
    return f, x1


@pytest.mark.parametrize("zeta_g", [0.0, 0.05, 0.2, 0.5])
@pytest.mark.parametrize("T", [50, 200])
def test_pgd_gap_is_within_the_theorem_6_bound(zeta_g, T):
    for seed in range(5):
        f, x1 = random_instance(seed)
        D, G = f.diameter, f.grad_bound
        grad_fn = adversarial_gradient_error(f, zeta_g)
        eta = pgd_step_size(zeta_g, D, G, T)
        x_bar, traj = projected_gradient_descent(f, grad_fn, x1, eta, T)

        # Definition 5 holds at every point where a gradient was queried.
        errors = jnp.linalg.norm(jax.vmap(grad_fn)(traj) - jax.vmap(f.grad)(traj), axis=1)
        assert float(errors.max()) <= zeta_g + 1e-4

        gap = float(f(x_bar)) - f.min_value()
        assert gap >= -1e-5
        assert gap <= alpha_pgd(zeta_g, D, G, T) + 1e-6


def test_bound_reduces_to_dg_over_sqrt_t_without_gradient_error():
    f, _ = random_instance(0)
    D, G = f.diameter, f.grad_bound
    for T in [10, 100, 1000]:
        assert alpha_pgd(0.0, D, G, T) == pytest.approx(D * G / T**0.5, rel=1e-12)


def test_worse_gradients_hurt_and_more_steps_help():
    """The trajectory-wise gradient error is what the bound charges for, and the
    adversarial error of Definition 5 really does cost value."""
    f, x1 = random_instance(2)
    D, G, T = f.diameter, f.grad_bound, 400
    gaps = []
    for zeta_g in (0.0, 0.1, 0.4, 1.0):
        x_bar, _ = projected_gradient_descent(
            f, adversarial_gradient_error(f, zeta_g), x1, pgd_step_size(zeta_g, D, G, T), T
        )
        gaps.append(float(f(x_bar)) - f.min_value())
    assert gaps == sorted(gaps)

    longer = []
    for T in (50, 200, 800):
        x_bar, _ = projected_gradient_descent(f, f.grad, x1, pgd_step_size(0.0, D, G, T), T)
        longer.append(float(f(x_bar)) - f.min_value())
    assert longer == sorted(longer, reverse=True)


def test_box_quadratic_constants_are_exact():
    f, _ = random_instance(1)
    key = jax.random.PRNGKey(0)
    xs = jax.random.uniform(key, (2000, f.a.shape[0]), minval=-1.0, maxval=1.0)
    assert float(jnp.max(jnp.linalg.norm(jax.vmap(f.grad)(xs), axis=1))) <= f.grad_bound + 1e-5
    assert float(jnp.min(jax.vmap(f)(xs))) >= f.min_value() - 1e-5
    assert float(jnp.max(jnp.linalg.norm(xs[:1000] - xs[1000:], axis=1))) <= f.diameter + 1e-5
