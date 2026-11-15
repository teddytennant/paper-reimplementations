import jax.numpy as jnp
import numpy as np

from jax_pdeg import project_ball, project_box, project_simplex, project_unconstrained


def test_box_is_componentwise_clip():
    x = jnp.array([-2.0, 0.3, 5.0])
    lo = jnp.array([-1.0, 0.0, 0.0])
    hi = jnp.array([1.0, 1.0, 2.0])
    out = project_box(x, lo, hi)
    np.testing.assert_allclose(out, jnp.clip(x, lo, hi))
    np.testing.assert_allclose(out, jnp.array([-1.0, 0.3, 2.0]))


def test_box_scalar_bounds():
    x = jnp.array([-2.0, 0.5, 3.0])
    np.testing.assert_allclose(project_box(x, -1.0, 1.0), jnp.array([-1.0, 0.5, 1.0]))


def test_ball_is_scaling():
    x = jnp.array([3.0, 4.0])
    out = project_ball(x, radius=1.0)
    np.testing.assert_allclose(out, x / 5.0, rtol=1e-12)
    inside = jnp.array([0.3, 0.4])
    np.testing.assert_allclose(project_ball(inside, radius=1.0), inside)
    c = jnp.array([1.0, 1.0])
    y = jnp.array([1.0, 5.0])
    py = project_ball(y, radius=2.0, center=c)
    np.testing.assert_allclose(jnp.linalg.norm(py - c), 2.0, rtol=1e-12)
    np.testing.assert_allclose(py, c + 2.0 * (y - c) / jnp.linalg.norm(y - c), rtol=1e-12)


def test_unconstrained_identity():
    x = jnp.array([1.2, -3.4])
    np.testing.assert_allclose(project_unconstrained(x), x)


def test_simplex_sum_and_nonneg():
    v = jnp.array([0.5, -1.0, 2.0, 0.1])
    p = project_simplex(v)
    assert jnp.all(p >= -1e-12)
    np.testing.assert_allclose(jnp.sum(p), 1.0, rtol=1e-12)
    # already on simplex
    u = jnp.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(project_simplex(u), u, atol=1e-12)
