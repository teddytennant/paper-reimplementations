"""One-atom Iterative Atom Refinement.

The update is equations (2.1)-(2.2) of arXiv:2609.23812: select the L columns
of Y with the largest inner products against the current unit vector, and
replace it by their normalized sum. L = M / K in the paper's model.

Ties. The threshold form ``<d, y> >= alpha`` with alpha the L-th largest score
is the same set when every score is unique. When several columns share the
cutoff score, top-L keeps the count exact. The tie-break is a stable ascending
argsort, the same order as ``numpy.argsort(..., kind='mergesort')``, and the
last L indices are kept. Equal scores therefore retain smaller original indices
earlier in the order, so the larger column index is preferred at the cutoff.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def uniform_sphere(key, n):
    """Draw a vector uniform on the unit sphere in R^n.

    ``key`` may be an integer seed or a JAX PRNG key. A standard Gaussian
    vector normalized to unit length is uniform on the sphere.
    """
    if isinstance(key, (int,)):
        key = jax.random.PRNGKey(key)
    v = jax.random.normal(key, (int(n),), dtype=jnp.float64)
    return v / jnp.linalg.norm(v)


def top_l_indices(scores, L):
    """Indices of the L largest scores, mergesort tie-break, last L of argsort.

    The returned indices are the tail of a stable ascending argsort, so they
    run from the L-th largest score up to the largest. Among equal scores the
    smaller original index comes first.
    """
    scores = jnp.asarray(scores)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-d, got shape {scores.shape}")
    M = int(scores.shape[0])
    L = int(L)
    if not 1 <= L <= M:
        raise ValueError(f"L must be in 1..M = 1..{M}, got {L}")
    # stable=True matches numpy mergesort: ties keep the original order.
    order = jnp.argsort(scores, stable=True)
    return order[-L:]


def refine_step(d, Y, L):
    """One IAR step.

    Returns ``(d_next, selected)`` where ``d_next = Y chi / ||Y chi||`` and
    ``selected`` indexes the L columns that chi marks.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    if Y.ndim != 2:
        raise ValueError(f"Y must be N by M, got shape {Y.shape}")
    d = jnp.asarray(d, dtype=jnp.float64)
    d = d / jnp.linalg.norm(d)
    scores = d @ Y
    selected = top_l_indices(scores, L)
    chi = jnp.zeros((Y.shape[1],), dtype=jnp.float64).at[selected].set(1.0)
    acc = Y @ chi
    d_next = acc / jnp.linalg.norm(acc)
    return d_next, selected


def one_atom_iar(Y, L, n_steps, d0=None, key=None):
    """Run one-atom IAR for ``n_steps`` refinements.

    Returns the trajectory of unit vectors with shape ``(n_steps + 1, N)``.
    Index r is d^{(r)}, including the initial vector at r = 0. If ``d0`` is
    omitted, ``key`` (an integer seed or a PRNG key) draws d0 uniform on the
    sphere.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    if Y.ndim != 2:
        raise ValueError(f"Y must be N by M, got shape {Y.shape}")
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    if d0 is None:
        if key is None:
            raise ValueError("one_atom_iar requires d0 or key")
        d = uniform_sphere(key, Y.shape[0])
    else:
        d = jnp.asarray(d0, dtype=jnp.float64)
        d = d / jnp.linalg.norm(d)
    trajectory = [d]
    for _ in range(n_steps):
        d, _ = refine_step(d, Y, L)
        trajectory.append(d)
    return jnp.stack(trajectory, axis=0)


def selection_counts(selected, X):
    """How many selected codes contain each atom.

    ``X`` is K by M. ``selected`` indexes columns. Entry k is
    ``|{i in selected : k in supp(x_i)}|``, with support the nonzero entries.
    """
    X = jnp.asarray(X)
    selected = jnp.asarray(selected)
    if selected.ndim != 1:
        raise ValueError(f"selected must be 1-d, got shape {selected.shape}")
    cols = X[:, selected]
    return jnp.sum(cols != 0, axis=1).astype(jnp.int32)


def selection_frequencies(selected, X):
    """Atom-selection frequencies p_k = count_k / L, equations (3.3) and (5.1).

    L is the number of selected columns.
    """
    selected = jnp.asarray(selected)
    L = int(selected.shape[0])
    if L == 0:
        raise ValueError("selected index set is empty")
    return selection_counts(selected, X).astype(jnp.float64) / L


def multi_restart_recovery(
    Y, L, n_restarts, n_steps=3, seed=0, dedup_threshold=0.5
):
    """Independent restarts, then greedy deduplication.

    Each restart draws d0 uniform on the sphere from ``jax.random.PRNGKey(seed)``
    split into ``n_restarts`` keys, and runs ``n_steps`` refinements (three in
    the paper's recovery argument). An estimate is kept when its maximum
    absolute inner product with already-kept estimates is strictly below
    ``dedup_threshold`` (0.5). Estimates are considered in restart order.
    Returns the kept atoms, shape ``(n_kept, N)``.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    n_restarts = int(n_restarts)
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")
    if isinstance(seed, (int,)):
        key = jax.random.PRNGKey(int(seed))
    else:
        key = seed
    keys = jax.random.split(key, n_restarts)
    estimates = []
    for subkey in keys:
        trajectory = one_atom_iar(Y, L, n_steps=n_steps, key=subkey)
        estimates.append(trajectory[-1])
    kept = []
    for estimate in estimates:
        if not kept:
            kept.append(estimate)
            continue
        overlaps = jnp.stack([jnp.abs(jnp.vdot(estimate, other)) for other in kept])
        if float(jnp.max(overlaps)) < dedup_threshold:
            kept.append(estimate)
    return jnp.stack(kept, axis=0)
