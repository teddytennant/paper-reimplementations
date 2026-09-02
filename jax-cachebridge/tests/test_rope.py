import jax
import jax.numpy as jnp

from cachebridge.rope import apply_rope, invert_rope, rope_angles


def test_inversion_round_trips_to_content_space():
    """Section 3.1: the fit needs the pre-RoPE state back out of the cache."""
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 40, 4, 16))
    ang = rope_angles(jnp.arange(40), 16)[None, :, None, :]
    assert jnp.max(jnp.abs(invert_rope(apply_rope(x, ang), ang) - x)) < 1e-12
    assert jnp.max(jnp.abs(apply_rope(invert_rope(x, ang), ang) - x)) < 1e-12


def test_rotation_is_orthogonal():
    x = jax.random.normal(jax.random.PRNGKey(1), (25, 16))
    ang = rope_angles(jnp.arange(25), 16)
    rotated = apply_rope(x, ang)
    assert jnp.max(jnp.abs(jnp.sum(rotated**2, -1) - jnp.sum(x**2, -1))) < 1e-12


def test_position_zero_is_the_identity():
    x = jax.random.normal(jax.random.PRNGKey(2), (7, 16))
    ang = rope_angles(jnp.zeros(7), 16)
    assert jnp.max(jnp.abs(apply_rope(x, ang) - x)) < 1e-14


def test_scores_depend_only_on_relative_position():
    q = jax.random.normal(jax.random.PRNGKey(3), (16,))
    k = jax.random.normal(jax.random.PRNGKey(4), (16,))
    def score(m, n):
        qa = apply_rope(q, rope_angles(jnp.asarray(float(m)), 16))
        ka = apply_rope(k, rope_angles(jnp.asarray(float(n)), 16))
        return float(qa @ ka)
    assert abs(score(9, 4) - score(30, 25)) < 1e-11
    assert abs(score(2, 7) - score(11, 16)) < 1e-11
