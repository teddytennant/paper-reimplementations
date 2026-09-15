import jax.numpy as jnp
from jax import random

from moara.align import alignment_score, cosine_alignment, module_alignment
from moara.decompose import decompose, recompose
from moara.project import project, reconstruct_direction, truncated_svd


def test_alignment_self_and_random():
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    g = random.normal(k1, (10, 14))
    g_rand = random.normal(k2, (10, 14))
    self_s = float(cosine_alignment(g, g))
    rand_s = float(cosine_alignment(g, g_rand))
    assert abs(self_s - 1.0) < 1e-6
    assert abs(rand_s - 1.0) > 0.2
    assert abs(float(alignment_score(g, g)) - 1.0) < 1e-6


def test_eq2_recon_alignment_and_module_mean():
    key = random.PRNGKey(1)
    g = random.normal(key, (8, 16))
    m, v = decompose(g, 4)
    p = truncated_svd(v, 3)
    g_recon = recompose(m, reconstruct_direction(project(v, p), p), 4)
    s = float(cosine_alignment(g, g_recon))
    assert 0.0 < s <= 1.0 + 1e-6
    scores = {"attn.q": jnp.array([0.9, 0.7]), "mlp.down": jnp.array([0.5])}
    means = module_alignment(scores)
    assert abs(float(means["attn.q"]) - 0.8) < 1e-9
    assert abs(float(means["mlp.down"]) - 0.5) < 1e-9
