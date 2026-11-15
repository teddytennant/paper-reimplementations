"""Per-head QCOC: prototype a key-value memory and fit values in closed form."""

import jax.numpy as jnp
from jax.nn import softmax

_SIGMA_FLOOR = 1e-12


def _as_float(x):
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(jnp.float32)
    return x


def _sqdist(z, centers):
    # (N, M) squared Euclidean distance. argmin/argmax ties keep the lowest index.
    diff = z[:, None, :] - centers[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def _assign(z, centers):
    return jnp.argmin(_sqdist(z, centers), axis=-1)


def _counts(labels, m):
    return jnp.bincount(labels, length=m)


def _means(x, labels, m):
    width = x.shape[-1]
    sums = jnp.zeros((m, width), dtype=x.dtype)
    sums = sums.at[labels].add(x)
    counts = _counts(labels, m).astype(x.dtype)
    return sums / counts[:, None]


def _farthest_point_init(z, m):
    """First center is row 0. Each next center maximizes min distance; ties take the lowest index."""
    min_dist = jnp.sum((z - z[0]) * (z - z[0]), axis=-1)
    chosen = [0]
    for _ in range(1, m):
        idx = int(jnp.argmax(min_dist))
        chosen.append(idx)
        dist = jnp.sum((z - z[idx]) * (z - z[idx]), axis=-1)
        min_dist = jnp.minimum(min_dist, dist)
    return z[jnp.asarray(chosen, dtype=jnp.int32)]


def _farthest_from_assigned(z, centers, labels, counts):
    """Index of the point farthest from its assigned center. Ties: lowest index.

    Prefer a point whose cluster still has another member, so the move can fill
    an empty slot without emptying a second cluster.
    """
    dist = jnp.sum((z - centers[labels]) * (z - centers[labels]), axis=-1)
    eligible = counts[labels] >= 2
    scored = jnp.where(eligible, dist, -jnp.inf)
    if not jnp.any(jnp.isfinite(scored)):
        scored = dist
    return int(jnp.argmax(scored))


def _repair_empty_centers(z, labels, centers):
    """Replace each empty center with the farthest point from its center, then reassign.

    Replacement points are chosen together so one reassignment cannot hand the
    same point to every empty center. If ties leave a repaired center empty,
    that replacement point is moved onto it.
    """
    m = int(centers.shape[0])
    for _ in range(m):
        counts = _counts(labels, m)
        empty = [int(j) for j in jnp.where(counts == 0)[0]]
        if not empty:
            break
        labels_sel = labels
        counts_sel = counts
        chosen = []
        for j in empty:
            i = _farthest_from_assigned(z, centers, labels_sel, counts_sel)
            centers = centers.at[j].set(z[i])
            old = int(labels_sel[i])
            labels_sel = labels_sel.at[i].set(j)
            counts_sel = counts_sel.at[old].add(-1).at[j].add(1)
            chosen.append((j, i))
        labels = _assign(z, centers)
        for j, i in chosen:
            if int(jnp.sum(labels == j)) == 0:
                labels = labels.at[i].set(j)
    return labels, centers


def _repair_empty_weights(z, prototypes, labels):
    """If reassignment emptied a prototype, move the farthest eligible point onto it."""
    m = prototypes.shape[0]
    for _ in range(m):
        counts = _counts(labels, m)
        empty = jnp.where(counts == 0)[0]
        if empty.size == 0:
            break
        j = int(empty[0])
        i = _farthest_from_assigned(z, prototypes, labels, counts)
        labels = labels.at[i].set(j)
    return labels


def _lloyd(z, m, max_iter):
    centers = _farthest_point_init(z, m)
    labels = _assign(z, centers)
    labels, _repaired = _repair_empty_centers(z, labels, centers)
    for _ in range(max_iter):
        centers = _means(z, labels, m)
        new_labels = _assign(z, centers)
        new_labels, _repaired = _repair_empty_centers(z, new_labels, centers)
        if bool(jnp.array_equal(new_labels, labels)):
            break
        labels = new_labels
    return labels


def phi(queries):
    """Vanilla scaled map phi(q) = q / sqrt(d)."""
    d = queries.shape[-1]
    scale = jnp.sqrt(jnp.asarray(d, dtype=queries.dtype))
    return queries / scale


def attend(queries, memory):
    """Softmax attention with the same phi and a natural-log multiplicity bias."""
    queries = _as_float(queries)
    keys = memory["keys"]
    values = memory["values"]
    log_w = memory["log_w"]
    logits = phi(queries) @ jnp.swapaxes(keys, -1, -2) + log_w
    return softmax(logits, axis=-1) @ values


def compress(keys, values, queries, n_prototypes, ridge=1e-3, eps_num=1e-12, max_iter=15):
    """Compress one head of a key-value memory into ``n_prototypes`` operators.

    keys: (N, d), values: (N, dv), queries: (N, d), with M = n_prototypes <= N.
    """
    keys = _as_float(keys)
    values = _as_float(values)
    queries = _as_float(queries)
    n = int(keys.shape[0])
    m = int(n_prototypes)
    if m < 1 or m > n:
        raise ValueError(f"n_prototypes must satisfy 1 <= M <= N, got M={m}, N={n}")
    if queries.shape[0] != n:
        raise ValueError("queries and keys must have the same number of rows")

    sigma_k = jnp.maximum(jnp.std(keys), _SIGMA_FLOOR)
    sigma_v = jnp.maximum(jnp.std(values), _SIGMA_FLOOR)
    z = jnp.concatenate([keys / sigma_k, values / sigma_v], axis=-1)

    labels = _lloyd(z, m, max_iter)
    k_tilde = _means(keys, labels, m)
    v0 = _means(values, labels, m)

    z_proto = jnp.concatenate([k_tilde / sigma_k, v0 / sigma_v], axis=-1)
    weight_labels = _assign(z, z_proto)
    weight_labels = _repair_empty_weights(z, z_proto, weight_labels)
    weights = _counts(weight_labels, m)
    log_w = jnp.log(weights.astype(keys.dtype))

    logits_p = phi(queries) @ jnp.swapaxes(k_tilde, -1, -2) + log_w[None, :]
    p = softmax(logits_p, axis=-1)
    y = softmax(phi(queries) @ jnp.swapaxes(keys, -1, -2), axis=-1) @ values

    gamma = jnp.maximum(jnp.sum(p * p) / m, eps_num)
    eye = jnp.eye(m, dtype=keys.dtype)
    gram = p.T @ p + (ridge * gamma) * eye
    rhs = p.T @ y + (ridge * gamma) * v0
    v_star = jnp.linalg.solve(gram, rhs)

    return {
        "keys": k_tilde,
        "values": v_star,
        "log_w": log_w,
        "weights": weights,
        "n_examples": n,
        "anchor_values": v0,
        "P": p,
        "Y": y,
    }
