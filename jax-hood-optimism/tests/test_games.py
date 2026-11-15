"""Normal-form games, mixed payoffs, and the centering of eq. (11)."""

import jax
import jax.numpy as jnp
import numpy as np

from hood.games import centered_payoffs, matching_pennies, payoff_vectors, utilities


def test_matching_pennies_mixed_utilities_zero():
    game = matching_pennies()
    x = jnp.ones((2, 2)) / 2.0
    u = utilities(game, x)
    np.testing.assert_allclose(np.asarray(u), 0.0, atol=1e-12)


def test_centered_payoffs_orthogonal_and_bounded():
    game = matching_pennies()
    key = jax.random.key(0)
    for _ in range(8):
        key, sub = jax.random.split(key)
        x = jax.nn.softmax(jax.random.normal(sub, (2, 2)), axis=-1)
        g = centered_payoffs(game, x)
        inner = jnp.sum(x * g, axis=-1)
        np.testing.assert_allclose(np.asarray(inner), 0.0, atol=1e-12)
        assert float(jnp.max(jnp.abs(g))) <= 2.0 + 1e-12


def test_payoff_vectors_matching_pennies_pure():
    game = matching_pennies()
    x = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    v = np.asarray(payoff_vectors(game, x))
    np.testing.assert_allclose(v[0], [1.0, -1.0])
    np.testing.assert_allclose(v[1], [-1.0, 1.0])
