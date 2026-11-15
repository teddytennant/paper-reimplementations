import jax.numpy as jnp
import numpy as np

from jax_pdeg import G, grad_G, make_problem


def _smooth_f(x, y):
    x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    return (
        0.5 * jnp.dot(x, x)
        + jnp.dot(x, y)
        - 0.35 * jnp.dot(y, y)
        + 0.15 * jnp.sum(jnp.sin(x)) * jnp.sum(jnp.cos(y))
    )


def test_G_and_grad_G_match_finite_differences():
    prob = make_problem(_smooth_f)
    lam, tau = 2.4, 0.3
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=3))
    y = jnp.asarray(rng.normal(size=3))
    z = jnp.asarray(rng.normal(size=3))
    gx, gy = grad_G(prob.grad_x, prob.grad_y, x, y, z, lam, tau)

    def Gval(xx, yy):
        return G(prob.f, xx, yy, z, lam, tau)

    eps = 1e-6
    gx_fd = np.zeros(3)
    gy_fd = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        gx_fd[i] = (Gval(x + e, y) - Gval(x - e, y)) / (2 * eps)
        gy_fd[i] = (Gval(x, y + e) - Gval(x, y - e)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(gx), gx_fd, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(np.asarray(gy), gy_fd, rtol=1e-7, atol=1e-8)

    gx_closed = prob.grad_x(x, y) + lam * (x - z)
    gy_closed = prob.grad_y(x, y) - tau * y
    np.testing.assert_allclose(gx, gx_closed)
    np.testing.assert_allclose(gy, gy_closed)
