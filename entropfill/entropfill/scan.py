"""Null-calibrated chunk scores and the feasible scan, Section 4.2.

Scores are equation (9). The scan sorts by increasing score, breaks ties
by decreasing chunk length and then by index, and adds a chunk only while
every row budget and the count budget still hold.
"""

import jax.numpy as jnp


def chunk_scores(pooled, document, chunks, epsilon_s, lam):
    """Return (ell, kappa, score) for each chunk.

    `document` marks positions in D. `chunks` is (k, P), one boolean row
    per chunk, supported inside D.
    """
    z = jnp.sum(pooled * document)
    safe_z = jnp.maximum(z, 1e-30)
    p = jnp.where(document, pooled / safe_z, 0.0)
    n = chunks.shape[1]
    lengths = jnp.sum(chunks, axis=-1)
    pi = lengths / n
    mass = chunks @ p
    ell = (mass + epsilon_s * pi) / ((1.0 + epsilon_s) * jnp.maximum(pi, 1e-30))
    smoothed = (p + epsilon_s * pi[:, None] / jnp.maximum(lengths[:, None], 1.0)) / (
        mass[:, None] + epsilon_s * pi[:, None]
    )
    inside = jnp.sum(chunks * smoothed**2, axis=-1)
    kappa = jnp.where(lengths > 1, (lengths * inside - 1.0) / (lengths - 1.0), 0.0)
    score = jnp.log(jnp.maximum(ell, 1e-30)) + lam * kappa
    return ell, kappa, score


def feasible_scan(scores, lengths, row_mass, row_budget, max_removed, token_target):
    """Greedy removal set. Returns a boolean mask over chunks.

    `row_mass` is (A, k), the mass each observer row puts on each chunk.
    A chunk is added only when every row stays inside its budget and the
    number removed stays at most `max_removed`.
    """
    order = jnp.lexsort((jnp.arange(scores.shape[0]), -lengths, scores))
    removed = jnp.zeros(scores.shape[0], dtype=bool)
    spent = jnp.zeros(row_mass.shape[0])
    count = jnp.array(0)
    tokens = jnp.array(0)

    def step(state, index):
        removed, spent, count, tokens = state
        fits_rows = jnp.all(spent + row_mass[:, index] <= row_budget + 1e-9)
        fits_count = count + 1 <= max_removed
        still_short = tokens < token_target
        take = fits_rows & fits_count & still_short
        removed = removed.at[index].set(take)
        spent = jnp.where(take, spent + row_mass[:, index], spent)
        count = count + take.astype(jnp.int32)
        tokens = tokens + jnp.where(take, lengths[index], 0)
        return (removed, spent, count, tokens), None

    (removed, _, _, _), _ = jax_scan(step, (removed, spent, count, tokens), order)
    return removed


def jax_scan(body, init, xs):
    import jax.lax

    return jax.lax.scan(body, init, xs)
