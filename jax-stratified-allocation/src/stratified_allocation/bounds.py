"""Proposition 11 of arXiv:2609.04420, uniform control over a finite candidate set."""

import jax
import jax.numpy as jnp


def serfling_uniform_bound(
    counts: jax.Array, allocation, n_candidates: int, bound_lambda: float, epsilon: float
) -> jax.Array:
    """Equation (16).

    For a finite candidate set C with |C| = M and per-row contributions bounded by
    |l_i| <= Lambda, under the class-conditional designs, with probability at least
    1 - epsilon,

        sup_{Phi in C} |Lhat(Phi) - L(Phi)|
            <= Lambda sum_c N_c sqrt( 2 (1 - (K_c - 1)/N_c) / K_c * log(2M/epsilon) ).

    Serfling's inequality for sampling without replacement, plus a union bound over C.
    The rate is O(K_c^{-1/2}) per class with the same N_c weighting that drives
    Equation (11), which is the point being made. It says nothing about a boosting path,
    only about one node with a fixed candidate set.
    """
    counts = jnp.asarray(counts, jnp.float64)
    alloc = jnp.asarray(allocation, jnp.float64)
    log_term = jnp.log(2.0 * n_candidates / epsilon)
    per_class = counts * jnp.sqrt(2.0 * (1.0 - (alloc - 1.0) / counts) / alloc * log_term)
    return bound_lambda * jnp.sum(per_class)
