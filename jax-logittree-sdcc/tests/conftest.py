import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from memorywalker.model import default_config, init_params  # noqa: E402
from memorywalker.tree import Eviction, Rollout, running_example  # noqa: E402


def make_model(seed=0, **kw):
    cfg = default_config(**kw)
    params = init_params(jax.random.key(seed), cfg)
    return params, cfg


def random_rollout(seed, length=14, n_evictions=3, vocab=23, response_start=1):
    """A rollout with a random eviction schedule: junctions strictly increasing,
    each removing one earlier response position, no position evicted twice."""
    rng = jax.random.key(seed)
    k_tok, k_j = jax.random.split(rng)
    tokens = tuple(int(x) for x in jax.random.randint(k_tok, (length,), 0, vocab))
    if n_evictions == 0:
        return Rollout(tokens=tokens, evictions=(), response_start=response_start)
    lo = response_start + 2
    junctions = sorted(set(int(x) for x in jax.random.randint(
        k_j, (n_evictions,), lo, length - 1)))
    evictions, used = [], set()
    for i, j in enumerate(junctions):
        options = [p for p in range(response_start, j) if p not in used]
        if not options:
            continue
        p = options[i % len(options)]
        used.add(p)
        evictions.append(Eviction(junction=j, span=(p,)))
    return Rollout(tokens=tokens, evictions=tuple(evictions),
                   response_start=response_start)


def tree_allclose(a, b, atol=1e-9, rtol=1e-9):
    """Max absolute and relative deviation over every leaf of two pytrees."""
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    worst = 0.0
    for x, y in zip(la, lb):
        assert x.shape == y.shape
        d = jnp.max(jnp.abs(x - y))
        scale = jnp.maximum(jnp.max(jnp.abs(x)), 1.0)
        worst = max(worst, float(d / scale))
        assert jnp.allclose(x, y, atol=atol, rtol=rtol), f"max abs diff {float(d)}"
    return worst


@pytest.fixture
def example():
    return running_example()
