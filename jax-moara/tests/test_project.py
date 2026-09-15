import jax.numpy as jnp
import numpy as np
from jax import random

from moara.project import project, reconstruct_direction, truncated_svd


def test_truncated_svd_orthonormal_and_rank():
    key = random.PRNGKey(0)
    v = random.normal(key, (12, 20))
    r = 5
    p = truncated_svd(v, r)
    assert p.shape == (12, r)
    gram = p.T @ p
    np.testing.assert_allclose(np.array(gram), np.eye(r), atol=1e-6)
    recon = reconstruct_direction(project(v, p), p)
    # P P^T V has rank at most r
    s = jnp.linalg.svd(recon, compute_uv=False)
    significant = s > 1e-8 * s[0]
    assert int(jnp.sum(significant)) <= r
