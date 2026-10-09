import jax
import jax.numpy as jnp
import numpy as np

from refinemix.algorithm import select_domain


def test_select_domain_rates_match_algorithm_1():
    key = jax.random.PRNGKey(0)
    low = select_domain(jnp.full((2000,), 0.2), tmax=0.5, lam=0.3, key=key)
    frac = float(np.mean(low == "source"))
    assert abs(frac - 0.3) < 0.03

    high = select_domain(jnp.full((2000,), 0.8), tmax=0.5, lam=0.3, key=key)
    assert np.all(high == "target")
    # Exactly zero, including when lambda would otherwise always pick the source.
    high_lam1 = select_domain(jnp.full((2000,), 0.8), tmax=0.5, lam=1.0, key=key)
    assert np.all(high_lam1 == "target")

    assert select_domain(0.8, 0.5, 0.9, jax.random.PRNGKey(1)) == "target"
    assert select_domain(0.5, 0.5, 1.0, jax.random.PRNGKey(2)) == "source"
    assert select_domain(0.5, 0.5, 0.0, jax.random.PRNGKey(3)) == "target"
