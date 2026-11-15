"""Wolfe-Quapp potential (S2), states A/B, force, and biased energy (10)."""

import jax
import jax.numpy as jnp
import numpy as np

from genaimmd.potential import (
    KBIAS,
    MIN_A,
    MIN_B,
    RADIUS,
    biased_energy,
    energy,
    force,
    grad_energy,
    in_A,
    in_B,
)


def test_energy_nonnegative_on_grid():
    xs = jnp.linspace(-2.5, 2.5, 51)
    X, Y = jnp.meshgrid(xs, xs, indexing="ij")
    xy = jnp.stack([X, Y], axis=-1).reshape(-1, 2)
    u = energy(xy)
    assert jnp.all(u >= -1e-5)


def test_minima_on_yx_and_are_minima():
    for m in (MIN_A, MIN_B):
        assert abs(float(m[0] - m[1])) < 4e-2
        g = grad_energy(m)
        assert float(jnp.linalg.norm(g)) < 1e-4
        u0 = float(energy(m))
        offsets = jnp.array(
            [[0.02, 0.0], [-0.02, 0.0], [0.0, 0.02], [0.0, -0.02], [0.02, 0.02]]
        )
        for d in offsets:
            assert float(energy(m + d)) > u0
    assert float(energy(MIN_B)) < 0.05
    assert float(energy(MIN_A)) < 0.5


def test_barrier_along_yx_is_order_h():
    t = jnp.linspace(float(MIN_A[0]), float(MIN_B[0]), 81)
    xy = jnp.stack([t, t], axis=-1)
    u = energy(xy)
    peak = float(jnp.max(u))
    assert 3.0 <= peak <= 5.0
    assert float(energy(jnp.array([0.0, 0.0]))) == 4.0


def test_finite_difference_matches_grad():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.5, 1.5, size=(6, 2)).astype(np.float32)
    eps = 1e-3
    for p in pts:
        x = jnp.asarray(p)
        g = np.asarray(grad_energy(x))
        fd = np.zeros(2, dtype=np.float64)
        for i in range(2):
            e = np.zeros(2, dtype=np.float32)
            e[i] = eps
            fd[i] = (float(energy(x + e)) - float(energy(x - e))) / (2.0 * eps)
        np.testing.assert_allclose(g, fd, rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(np.asarray(force(x)), -g, atol=1e-5)


def test_state_disks():
    assert bool(in_A(MIN_A)) and not bool(in_B(MIN_A))
    assert bool(in_B(MIN_B)) and not bool(in_A(MIN_B))
    inside_a = MIN_A + jnp.array([0.2, 0.0])
    assert bool(in_A(inside_a)) and not bool(in_B(inside_a))
    outside = MIN_A + jnp.array([RADIUS + 0.1, 0.0])
    assert not bool(in_A(outside))
    assert float(jnp.linalg.norm(MIN_A - MIN_B)) > 2.0 * RADIUS


def test_conditioned_energy_eq10():
    x = MIN_B
    u = energy(x)
    xi_hat = 0.4
    assert float(biased_energy(x, xi_hat, xi_hat, kbias=KBIAS)) == float(u)
    higher = float(biased_energy(x, 0.9, xi_hat, kbias=KBIAS))
    assert higher > float(u)
    expected = float(u) + 0.5 * KBIAS * (0.9 - xi_hat) ** 2
    np.testing.assert_allclose(higher, expected, rtol=1e-5)
