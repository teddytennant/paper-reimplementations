import jax.numpy as jnp

from scr.cluster import anchor_cluster, fingerprints
from scr.select import augment_indices, select_indices


def sparse_attention(Q, K, V, query_pos, key_pos):
    """Softmax attention with a causal mask on explicit positions."""
    Q = jnp.asarray(Q, dtype=jnp.float32)
    K = jnp.asarray(K, dtype=jnp.float32)
    V = jnp.asarray(V, dtype=jnp.float32)
    query_pos = jnp.asarray(query_pos)
    key_pos = jnp.asarray(key_pos)
    d_h = Q.shape[-1]
    scale = jnp.sqrt(jnp.asarray(d_h, dtype=jnp.float32))
    logits = jnp.einsum("hqd,hkd->hqk", Q, K) / scale
    # 0 when key_pos[s] <= query_pos[r], else -inf
    allow = key_pos[None, :] <= query_pos[:, None]
    mask = jnp.where(allow, 0.0, -jnp.inf)
    logits = logits + mask[None, :, :]
    # rows with no allowed key stay zero rather than NaN
    all_blocked = jnp.all(~allow, axis=1)
    weights = jax_softmax(logits, axis=-1)
    weights = jnp.where(all_blocked[None, :, None], 0.0, weights)
    return jnp.einsum("hqk,hkd->hqd", weights, V)


def jax_softmax(x, axis):
    finite = jnp.where(jnp.isfinite(x), x, -jnp.inf)
    m = jnp.max(finite, axis=axis, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    e = jnp.exp(jnp.where(jnp.isfinite(x), x - m, -jnp.inf))
    z = jnp.sum(e, axis=axis, keepdims=True)
    return jnp.where(z > 0, e / z, 0.0)


def _flat_queries(Q):
    """(n_heads, length, d_h) -> (length, n_heads * d_h)."""
    n_heads, length, d_h = Q.shape
    return jnp.transpose(Q, (1, 0, 2)).reshape(length, n_heads * d_h)


def _cluster_span(cluster, L, length):
    start = cluster[0] * L
    end = min(cluster[-1] * L + L, length)
    return start, end


def selection_calls(Q, L, theta, Lmax):
    """How many select_indices calls scr_attention would issue."""
    flat = _flat_queries(jnp.asarray(Q, dtype=jnp.float32))
    fps = fingerprints(flat, L)
    clusters = anchor_cluster(fps, theta, L, Lmax)
    return len(clusters)


def scr_attention(Q, K, V, L, theta, Lmax, k, n_init, query_pos, key_pos):
    """One selection per anchor cluster, then sparse attention over the cluster."""
    Q = jnp.asarray(Q, dtype=jnp.float32)
    K = jnp.asarray(K, dtype=jnp.float32)
    V = jnp.asarray(V, dtype=jnp.float32)
    query_pos = jnp.asarray(query_pos)
    key_pos = jnp.asarray(key_pos)
    n_heads, length, d_h = Q.shape
    flat = _flat_queries(Q)
    fps = fingerprints(flat, L)
    clusters = anchor_cluster(fps, theta, L, Lmax)
    out = jnp.zeros_like(Q)
    n_cache = int(key_pos.shape[0])
    for cluster in clusters:
        a = cluster[0]
        a0 = a * L
        a1 = min(a0 + L, length)
        q_anchor = jnp.mean(Q[:, a0:a1, :], axis=1)
        ic = select_indices(q_anchor, K, k)
        start, end = _cluster_span(cluster, L, length)
        local = [int(p) for p in list(key_pos) if start <= int(p) < end]
        idx = augment_indices(ic, n_cache, n_init, local)
        idx_arr = jnp.asarray(idx, dtype=jnp.int32)
        K_sel = K[:, idx_arr, :]
        V_sel = V[:, idx_arr, :]
        q_idx = jnp.arange(start, end)
        O = sparse_attention(Q[:, start:end, :], K_sel, V_sel, query_pos[q_idx], key_pos[idx_arr])
        out = out.at[:, start:end, :].set(O)
    return out


def per_subchunk_select(Q, K, k, L):
    """select_indices once per sub-chunk. Returns (calls, list of index arrays)."""
    Q = jnp.asarray(Q, dtype=jnp.float32)
    K = jnp.asarray(K, dtype=jnp.float32)
    n_heads, length, d_h = Q.shape
    M = (length + L - 1) // L
    chosen = []
    for i in range(M):
        a0 = i * L
        a1 = min(a0 + L, length)
        q = jnp.mean(Q[:, a0:a1, :], axis=1)
        chosen.append(select_indices(q, K, k))
    return M, chosen
