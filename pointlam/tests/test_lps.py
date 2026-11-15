import jax
import jax.numpy as jnp
import numpy as np

from pointlam.grid import neighbor_mask
from pointlam.lps import apply_lps, deviation, dss_indices, mlp_apply
from pointlam.nn import Linear


def _high_dev_cloud():
    """Cluster in cell (0,0,0), a few adjacent-cell outliers, far junk."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(loc=0.5, scale=0.05, size=(40, 3)).astype(np.float32)
    cluster = np.clip(cluster, 0.05, 0.95)
    outliers = np.array(
        [
            [1.6, 0.5, 0.5],
            [0.5, 1.6, 0.5],
            [0.5, 0.5, 1.6],
            [1.6, 1.6, 0.5],
        ],
        dtype=np.float32,
    )
    far = rng.normal(loc=8.0, scale=0.05, size=(12, 3)).astype(np.float32)
    points = jnp.asarray(np.concatenate([cluster, outliers, far], axis=0))
    return points, outliers.shape[0]


def test_dss_high_dev_subset_unique_features():
    points, _n_out = _high_dev_cloud()
    n = int(points.shape[0])
    m = 8
    features = jnp.arange(n, dtype=jnp.float32)[:, None] * jnp.array(
        [1.0, 0.25, -0.5], dtype=jnp.float32
    )
    key = jax.random.PRNGKey(1)
    pts_out, feats_out, idx, dev = apply_lps(
        points, features, key, m, cell_size=1.0, mlp_params=None
    )
    assert pts_out.shape == (m, 3)
    assert feats_out.shape == (m, 3)
    assert idx.shape == (m,)
    assert len(set(np.array(idx).tolist())) == m
    top_dev = np.array(jnp.argsort(-dev, stable=True)[:m])
    out_set = set(np.array(idx).tolist())
    assert set(top_dev.tolist()).issubset(out_set)
    np.testing.assert_allclose(np.array(pts_out), np.array(points)[np.array(idx)])
    np.testing.assert_allclose(np.array(feats_out), np.array(features)[np.array(idx)])


def test_dss_union_then_crop():
    n, m = 16, 6
    dev = jnp.arange(n, dtype=jnp.float32)
    rand = jnp.linspace(0.3, 0.1, n)
    idx = dss_indices(dev, rand, m)
    assert idx.shape == (m,)
    assert len(set(np.array(idx).tolist())) == m
    top = set(np.array(jnp.argsort(-dev)[:m]).tolist())
    assert top.issubset(set(np.array(idx).tolist()))


def test_devnet_identity_two_cell_cloud():
    cell_size = 1.0
    points = jnp.array(
        [
            [0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60],
            [5.10, 5.20, 5.30],
            [5.40, 5.50, 5.60],
        ],
        dtype=jnp.float32,
    )
    features = points
    mask = neighbor_mask(points, cell_size)
    mask_np = np.array(mask)
    assert mask_np[0, 1] and mask_np[1, 0]
    assert mask_np[2, 3] and mask_np[3, 2]
    assert not mask_np[0, 2]
    dev = np.array(deviation(features, mask))
    pts = np.array(points)
    for i in range(pts.shape[0]):
        nbrs = pts[mask_np[i]]
        mean = nbrs.mean(axis=0)
        expected = np.linalg.norm(pts[i] - mean)
        np.testing.assert_allclose(dev[i], expected, atol=1e-5, rtol=1e-5)


def test_devnet_fixed_linear():
    cell_size = 1.0
    points = jnp.array(
        [
            [0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60],
            [5.10, 5.20, 5.30],
            [5.40, 5.50, 5.60],
        ],
        dtype=jnp.float32,
    )
    W = jnp.array([[1.0, 0.0], [0.0, 2.0], [0.5, 0.5]], dtype=jnp.float32)
    lin = Linear(W, None)
    h = mlp_apply(lin, points)
    mask = neighbor_mask(points, cell_size)
    dev = np.array(deviation(h, mask))
    h_np = np.array(h)
    mask_np = np.array(mask)
    for i in range(h_np.shape[0]):
        mean = h_np[mask_np[i]].mean(axis=0)
        expected = np.linalg.norm(h_np[i] - mean)
        np.testing.assert_allclose(dev[i], expected, atol=1e-5, rtol=1e-5)
