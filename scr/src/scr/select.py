import jax.numpy as jnp


def select_indices(q_heads, K, k):
    """Top-k cache positions by the sum of per-head softmax scores."""
    q = jnp.asarray(q_heads, dtype=jnp.float32)
    K = jnp.asarray(K, dtype=jnp.float32)
    n_heads, d_h = q.shape
    if K.shape[0] != n_heads or K.shape[2] != d_h:
        raise ValueError("K must be (n_heads, N, d_h)")
    scale = jnp.sqrt(jnp.asarray(d_h, dtype=jnp.float32))
    # score_h = softmax(q_h @ K_h.T / sqrt(d_h)) over the N keys
    logits = jnp.einsum("hd,hnd->hn", q, K) / scale
    scores = jax_softmax(logits, axis=1)
    criticality = jnp.sum(scores, axis=0)
    N = int(criticality.shape[0])
    k = min(int(k), N)
    # stable: smaller index wins ties. argsort is stable in recent jax; break ties explicitly.
    order = jnp.argsort(-criticality, stable=True)
    return order[:k]


def jax_softmax(x, axis):
    x = x - jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=axis, keepdims=True)


def augment_indices(ic, n_cache, n_init, local_positions):
    """Add attention sinks and a local window, then dedupe and sort."""
    sinks = list(range(min(int(n_init), int(n_cache))))
    local = [int(p) for p in local_positions if 0 <= int(p) < int(n_cache)]
    chosen = [int(i) for i in list(ic)]
    return sorted(set(sinks + local + chosen))
