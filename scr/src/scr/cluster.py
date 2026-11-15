import jax.numpy as jnp


def fingerprints(Q, L):
    """Mean-pool query sub-chunks and L2-normalize. Q is (length, d)."""
    Q = jnp.asarray(Q, dtype=jnp.float32)
    length, d = Q.shape
    if L <= 0:
        raise ValueError("L must be positive")
    M = (length + L - 1) // L
    chunks = []
    for i in range(M):
        sl = Q[i * L : min((i + 1) * L, length)]
        chunks.append(jnp.mean(sl, axis=0))
    V = jnp.stack(chunks, axis=0)
    norms = jnp.linalg.norm(V, axis=1, keepdims=True)
    return jnp.where(norms > 0, V / norms, jnp.zeros_like(V))


def anchor_cluster(fps, theta, L, Lmax):
    """Contiguous anchor clusters. Compare to the anchor, cap tokens at Lmax."""
    fps = jnp.asarray(fps, dtype=jnp.float32)
    M = int(fps.shape[0])
    clusters = []
    a = 0
    while a < M:
        Ca = [a]
        j = a + 1
        while j < M and float(jnp.dot(fps[a], fps[j])) >= theta and len(Ca) * L < Lmax:
            Ca.append(j)
            j += 1
        clusters.append(Ca)
        a = j
    return clusters
