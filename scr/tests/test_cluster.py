import math

import jax.numpy as jnp

from scr.cluster import anchor_cluster, fingerprints


def test_identical_fingerprints_fill_lmax():
    Q = jnp.ones((16, 4))
    fps = fingerprints(Q, L=4)
    assert fps.shape == (4, 4)
    clusters = anchor_cluster(fps, theta=0.95, L=4, Lmax=16)
    assert clusters == [[0, 1, 2, 3]]


def test_anchor_stops_drift_that_a_predecessor_rule_would_keep():
    # Consecutive unit vectors stay above theta; the last is below theta with the anchor.
    theta = 0.95
    step = 0.1
    angles = [i * step for i in range(6)]
    fps = jnp.stack(
        [jnp.array([math.cos(a), math.sin(a)]) for a in angles]
    )
    # consecutive cosine is cos(0.1) ~ 0.995 > theta
    assert float(jnp.dot(fps[0], fps[1])) >= theta
    # last vs anchor is cos(0.5) ~ 0.878 < theta
    assert float(jnp.dot(fps[0], fps[-1])) < theta
    clusters = anchor_cluster(fps, theta=theta, L=1, Lmax=100)
    assert len(clusters) > 1
    assert clusters[0][-1] < len(fps) - 1


def test_orthogonal_alternation_is_all_singletons():
    fps = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    clusters = anchor_cluster(fps, theta=0.95, L=1, Lmax=100)
    assert clusters == [[0], [1], [2], [3]]


def test_clusters_partition_and_are_contiguous():
    Q = jnp.arange(20 * 3, dtype=jnp.float32).reshape(20, 3)
    fps = fingerprints(Q, L=4)
    clusters = anchor_cluster(fps, theta=0.5, L=4, Lmax=12)
    flat = [i for c in clusters for i in c]
    assert flat == list(range(int(fps.shape[0])))
    for c in clusters:
        assert c == list(range(c[0], c[-1] + 1))
