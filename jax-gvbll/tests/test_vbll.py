"""KL, reparameterization, and ELBO checks (real JAX, no mocks)."""

import jax
import jax.numpy as jnp

from gvbll.data import two_community_graph
from gvbll.elbo import elbo, expected_log_likelihood, map_nll, neg_elbo
from gvbll.encoder import encode, init_graphsage
from gvbll.vbll import init_vbll, kl_divergence, sample_weights, std_from_log_std


def test_kl_zero_at_prior():
    mean = jnp.zeros((3, 5))
    std = jnp.ones((3, 5))
    kl = kl_divergence(mean, std, prior_std=1.0)
    assert jnp.isfinite(kl)
    assert float(kl) == 0.0


def test_kl_positive_away_from_prior():
    mean = jnp.zeros((3, 5)).at[0, 0].set(1.3)
    std = jnp.ones((3, 5))
    kl_mean = kl_divergence(mean, std, prior_std=1.0)
    std_wide = jnp.full((3, 5), 2.0)
    kl_std = kl_divergence(jnp.zeros((3, 5)), std_wide, prior_std=1.0)
    assert float(kl_mean) > 0.0
    assert float(kl_std) > 0.0


def test_kl_matches_univariate_formula():
    mean = jnp.array([[0.4]])
    std = jnp.array([[1.5]])
    prior_std = 0.8
    got = float(kl_divergence(mean, std, prior_std=prior_std))
    var = 1.5**2
    prior_var = 0.8**2
    want = 0.5 * (var / prior_var + 0.4**2 / prior_var - 1.0 - jnp.log(var / prior_var))
    assert abs(got - float(want)) < 1e-6


def test_reparameterized_weights_shape():
    mean = jnp.zeros((4, 7))
    std = 0.2 * jnp.ones((4, 7))
    key = jax.random.PRNGKey(0)
    for s in (4, 8):
        w = sample_weights(mean, std, key, s)
        assert w.shape == (s, 4, 7)


def test_elbo_finite_on_twelve_node_graph():
    key = jax.random.PRNGKey(1)
    k_data, k_enc, k_elbo = jax.random.split(key, 3)
    x, adj, y = two_community_graph(k_data, n_per=6, feat_dim=4)
    assert x.shape[0] == 12
    enc = init_graphsage(k_enc, in_dim=4, hidden_dim=8, out_dim=6)
    h = encode(enc, x, adj)
    vbll = init_vbll(n_classes=2, embed_dim=6, prior_std=1.0)
    std = std_from_log_std(vbll["log_std"])
    ell = expected_log_likelihood(vbll["mean"], std, h, y, k_elbo, n_samples=6)
    e = elbo(vbll["mean"], std, h, y, k_elbo, n_samples=6, beta=1.0)
    nll = neg_elbo(vbll["mean"], std, h, y, k_elbo, n_samples=6, beta=1.0)
    det = map_nll(vbll["mean"], h, y)
    for val in (ell, e, nll, det):
        assert jnp.isfinite(val)
