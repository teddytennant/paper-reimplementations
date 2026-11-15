"""Monte Carlo drivers for the design tests.

The estimators are cheap but the replicate counts are not, so every driver runs the
replicates in chunks of a fixed size under one jitted vmap. Chunk size is a memory knob
only; the draws are the same either way, since each replicate gets its own split key.
"""

import jax
import jax.numpy as jnp

from stratified_allocation.designs import (
    class_index_sets,
    stratified_estimate,
    uniform_estimate,
)

CHUNK = 2000


def _run(fn, key, reps, chunk=CHUNK):
    assert reps % chunk == 0, "replicate count must be a multiple of the chunk size"
    keys = jax.random.split(key, reps).reshape(reps // chunk, chunk)
    batched = jax.jit(jax.vmap(fn))
    return jnp.concatenate([batched(keys[i]) for i in range(keys.shape[0])])


def stratified_draws(key, losses, labels, k0, k1, reps, chunk=CHUNK):
    idx0, idx1 = class_index_sets(labels)
    return _run(
        lambda k: stratified_estimate(k, losses, idx0, idx1, k0, k1), key, reps, chunk
    )


def uniform_draws(key, losses, budget, reps, chunk=CHUNK):
    return _run(lambda k: uniform_estimate(k, losses, budget), key, reps, chunk)


def summary(draws):
    """Mean, unbiased variance, and the standard error of the mean."""
    r = draws.shape[0]
    mean = jnp.mean(draws)
    var = jnp.sum((draws - mean) ** 2) / (r - 1.0)
    return mean, var, jnp.sqrt(var / r)


def variance_relative_se(draws):
    """Standard error of the sample variance, relative to the variance itself.

    Var(s^2) ~ (mu_4 - sigma^4)/R, so the relative error is sqrt((kappa - 1)/R) with
    kappa the kurtosis. Estimated from the draws rather than assumed Gaussian, because
    the rare-event strata are anything but.
    """
    r = draws.shape[0]
    mean = jnp.mean(draws)
    d2 = (draws - mean) ** 2
    var = jnp.sum(d2) / (r - 1.0)
    mu4 = jnp.mean(d2**2)
    return jnp.sqrt(jnp.maximum(mu4 - var**2, 0.0) / r) / var


_CACHE = {}


def cached_stratified(tag, seed, losses, labels, k0, k1, reps):
    """Reuse a replicate set across tests that ask the same question of it.

    The unbiasedness and variance tests want the same 200k draws from the same design,
    and drawing them twice doubles the suite's runtime for nothing. ``tag`` identifies
    the population; the caller is responsible for it naming the population uniquely.
    """
    ck = (tag, seed, int(k0), int(k1), reps)
    if ck not in _CACHE:
        _CACHE[ck] = stratified_draws(jax.random.key(seed), losses, labels, k0, k1, reps)
    return _CACHE[ck]


def cached_uniform(tag, seed, losses, budget, reps):
    ck = (tag, seed, "unif", budget, reps)
    if ck not in _CACHE:
        _CACHE[ck] = uniform_draws(jax.random.key(seed), losses, budget, reps)
    return _CACHE[ck]


def strat_pop(n, n1):
    """The population the class-conditional tests share."""
    from stratified_allocation import random_population

    return random_population(jax.random.key(1), n, n1, scale=(1.0, 2.0), shift=(0.0, 1.5))


def unif_pop(n, n1):
    """The population the uniform-design tests share."""
    from stratified_allocation import random_population

    return random_population(jax.random.key(0), n, n1, scale=(1.0, 2.0), shift=(0.0, 1.5))
