"""Checks for the tiny hierarchical continuous diffusion slice."""

import jax
import jax.numpy as jnp
from jax import random

import hc_dlm


def test_entropy_hand_expansion():
    var = jnp.array([0.25, 1.0, 4.0, 0.5])
    hand = 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * jnp.e * var))
    assert jnp.allclose(hc_dlm.gaussian_entropy(var), hand)

    var_b = jnp.array([[0.5, 2.0], [1.5, 0.3]])
    hand_b = 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * jnp.e * var_b), axis=-1)
    assert jnp.allclose(hc_dlm.gaussian_entropy(var_b, axis=-1), hand_b)


def test_reparam_gradient_reaches_mean():
    mean = jnp.array([0.4, -1.2, 0.7])
    log_std = jnp.array([0.1, -0.3, 0.0])
    eps = jnp.array([0.2, -0.5, 1.1])

    def through_mean(m):
        x0 = hc_dlm.reparameterize(m, log_std, eps)
        return jnp.sum(jnp.sin(x0))

    g_mean = jax.grad(through_mean)(mean)
    x0 = hc_dlm.reparameterize(mean, log_std, eps)
    g_x = jax.grad(lambda x: jnp.sum(jnp.sin(x)))(x0)
    assert jnp.allclose(g_mean, g_x)
    assert jnp.any(g_mean != 0)

    key = random.PRNGKey(2)
    params = hc_dlm.init_params(key, vocab=5, d=3, hidden=8, embed_dim=4)
    k0 = jnp.array([[1, 0, 2, 1]])
    eps_e = random.normal(random.PRNGKey(3), (1, 4, 3))

    def through_encode(p):
        x0, _mean, _log_std = hc_dlm.encode(p, k0, eps_e)
        return jnp.sum(jnp.sin(x0))

    g = jax.grad(through_encode)(params)
    x0, _mean, _log_std = hc_dlm.encode(params, k0, eps_e)
    expected_b = jnp.sum(jnp.cos(x0), axis=(0, 1))
    assert jnp.allclose(g["encoder"]["b_mean"], expected_b, atol=1e-5)
    assert jnp.any(g["encoder"]["b_mean"] != 0)


def test_mc_log_density_matches_neg_entropy():
    key = random.PRNGKey(7)
    mean = jnp.array([0.5, -1.0, 2.0, 0.25])
    log_std = jnp.array([-0.2, 0.3, 0.0, 0.15])
    var = jnp.exp(2.0 * log_std)
    entropy = hc_dlm.gaussian_entropy(var)
    n = 30000
    eps = random.normal(key, (n, mean.shape[0]))
    samples = mean + jnp.exp(log_std) * eps
    log_q = jax.vmap(lambda x: hc_dlm.gaussian_log_density(x, mean, log_std))(samples)
    mc = jnp.mean(log_q)
    assert jnp.isfinite(mc)
    assert jnp.abs(mc - (-entropy)) < 0.05


def test_interpolant_endpoints_and_midpoint():
    x0 = jnp.arange(12.0).reshape(2, 2, 3)
    eps = jnp.ones_like(x0) * 3.0
    assert jnp.allclose(hc_dlm.interpolant(x0, eps, 0.0), x0)
    assert jnp.allclose(hc_dlm.interpolant(x0, eps, 1.0), eps)
    s = jnp.array([0.25, 0.5])
    got = hc_dlm.interpolant(x0, eps, s)
    expected = (1.0 - s)[:, None, None] * x0 + s[:, None, None] * eps
    assert jnp.allclose(got, expected)


def test_absorbing_marginal_frequency():
    key = random.PRNGKey(1)
    k0 = jnp.zeros((128, 16), dtype=jnp.int32)
    s = 0.3
    mid = 5
    keys = random.split(key, 40)

    def rate(k):
        k_t = hc_dlm.absorbing_forward(k0, s, k, mid)
        return jnp.mean((k_t == mid).astype(jnp.float32))

    mean_rate = jnp.mean(jax.vmap(rate)(keys))
    assert jnp.abs(mean_rate - 0.3) < 0.015
    # Endpoints of alpha(s) = s.
    assert jnp.all(hc_dlm.absorbing_forward(k0, 0.0, random.PRNGKey(0), mid) == k0)
    assert jnp.all(hc_dlm.absorbing_forward(k0, 1.0, random.PRNGKey(0), mid) == mid)
    kept = hc_dlm.absorbing_forward(k0, 0.0, random.PRNGKey(4), mid)
    assert jnp.all(kept == 0)


def _zero_module(in_dim, hidden, out_dim):
    return {
        "w1": jnp.zeros((in_dim, hidden)),
        "b1": jnp.zeros((hidden,)),
        "w2": jnp.zeros((hidden, out_dim)),
        "b2": jnp.zeros((out_dim,)),
    }


def _scaffold_params():
    """Denoiser is a pure function of the token embedding (constructed weights)."""
    vocab, d, embed, hidden, time_dim = 4, 2, 2, 2, 2
    tok = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [-1.0, 0.25],
        ]
    )
    w1 = jnp.zeros((d + embed + time_dim, hidden))
    # Input layout: [x_t (d), token embed (embed), time (time_dim)].
    w1 = w1.at[d + 0, 0].set(1.0)
    w1 = w1.at[d + 1, 1].set(1.0)
    return {
        "encoder": {
            "tok_embed": jnp.zeros((vocab, embed)),
            "w1": jnp.zeros((embed, hidden)),
            "b1": jnp.zeros((hidden,)),
            "w_mean": jnp.zeros((hidden, d)),
            "b_mean": jnp.zeros((d,)),
            "w_logstd": jnp.zeros((hidden, d)),
            "b_logstd": jnp.zeros((d,)),
        },
        "predictor": _zero_module(d, hidden, vocab - 1),
        "denoiser": {
            "tok_embed": tok,
            "w1": w1,
            "b1": jnp.zeros((hidden,)),
            "w2": jnp.eye(hidden, d),
            "b2": jnp.zeros((d,)),
        },
    }


def test_scaffold_embedding_in_tree_and_changes_output():
    key = random.PRNGKey(0)
    params = hc_dlm.init_params(key, vocab=7, d=8, hidden=16, embed_dim=8, time_dim=8)
    assert "tok_embed" in params["denoiser"]
    assert params["denoiser"]["tok_embed"].shape == (7, 8)

    built = _scaffold_params()
    x_t = jnp.zeros((2, 3, 2))
    s = jnp.array(0.4)
    k_a = jnp.array([[0, 1, 2], [1, 1, 0]])
    k_b = jnp.array([[1, 0, 3], [0, 2, 1]])
    y_a = hc_dlm.denoise(built, x_t, k_a, s)
    y_b = hc_dlm.denoise(built, x_t, k_b, s)
    assert y_a.shape == x_t.shape
    assert not jnp.allclose(y_a, y_b)
    # Same tokens, same x_t: same output. Time columns of W are zero, so s is ignored.
    assert jnp.allclose(y_a, hc_dlm.denoise(built, x_t, k_a, jnp.array(0.9)))
    # Hand check: position embedding [1, 0] maps to gelu([1, 0]).
    assert jnp.allclose(y_a[0, 0], jax.nn.gelu(jnp.array([1.0, 0.0])))


def test_recon_grad_does_not_change_denoiser():
    key = random.PRNGKey(9)
    params = hc_dlm.init_params(key, vocab=7, d=8, hidden=16, embed_dim=8)
    k0 = random.randint(random.PRNGKey(1), (4, 5), 0, 6)
    x_t = random.normal(random.PRNGKey(2), (4, 5, 8))
    k_t = random.randint(random.PRNGKey(3), (4, 5), 0, 7)
    s = jnp.array(0.35)

    def loss(p):
        x0_hat = hc_dlm.denoise(p, x_t, k_t, s)
        return hc_dlm.reconstruction_nll(p, k0, x0_hat)

    grads = jax.grad(loss)(params)
    den_leaves = jax.tree_util.tree_leaves(grads["denoiser"])
    assert all(jnp.all(g == 0) for g in den_leaves)
    pred_leaves = jax.tree_util.tree_leaves(grads["predictor"])
    assert any(jnp.any(g != 0) for g in pred_leaves)
    # Encoder is not on this path either.
    enc_leaves = jax.tree_util.tree_leaves(grads["encoder"])
    assert all(jnp.all(g == 0) for g in enc_leaves)


def _predictor_on_x0_hat_params():
    """Constant x0_hat = 1, and p(token | x) prefers class 0 at ones, class 1 at zeros."""
    vocab, d, embed, hidden, time_dim = 3, 2, 2, 2, 2
    w1 = jnp.array([[8.0, 0.0], [8.0, 0.0]])
    w2 = jnp.array([[1.0, -1.0], [0.0, 0.0]])
    b2 = jnp.array([0.0, 8.0])
    return {
        "encoder": {
            "tok_embed": jnp.zeros((vocab, embed)),
            "w1": jnp.zeros((embed, hidden)),
            "b1": jnp.zeros((hidden,)),
            "w_mean": jnp.zeros((hidden, d)),
            "b_mean": jnp.zeros((d,)),
            "w_logstd": jnp.zeros((hidden, d)),
            "b_logstd": jnp.zeros((d,)),
        },
        "predictor": {
            "w1": w1,
            "b1": jnp.zeros((hidden,)),
            "w2": w2,
            "b2": b2,
        },
        "denoiser": {
            "tok_embed": jnp.zeros((vocab, embed)),
            "w1": jnp.zeros((d + embed + time_dim, hidden)),
            "b1": jnp.zeros((hidden,)),
            "w2": jnp.zeros((hidden, d)),
            "b2": jnp.ones((d,)),
        },
    }


def test_reverse_unmasked_stay_and_predictor_sees_x0_hat():
    params = _predictor_on_x0_hat_params()
    # mask id = 2. Positions 0 and 2 are already real tokens.
    k_t = jnp.array([[0, 2, 1, 2]])
    x_t = jnp.zeros((1, 4, 2))
    x_prev, k_prev, x0_hat = hc_dlm.reverse_step(
        params, x_t, k_t, jnp.array(1.0), jnp.array(0.0), random.PRNGKey(11)
    )
    assert jnp.all(k_prev[0, 0] == 0)
    assert jnp.all(k_prev[0, 2] == 1)
    # x0_hat is the constant ones vector, not x_t.
    assert jnp.allclose(x0_hat, jnp.ones_like(x_t))
    assert not jnp.allclose(x0_hat, x_t)
    logits_hat = hc_dlm.token_logits(params, x0_hat)
    logits_xt = hc_dlm.token_logits(params, x_t)
    assert jnp.all(jnp.argmax(logits_hat, axis=-1) == 0)
    assert jnp.all(jnp.argmax(logits_xt, axis=-1) == 1)
    # s=1 -> s_prev=0 unmasks every masked site. Samples must follow p(. | x0_hat).
    assert jnp.all(k_prev[0, 1] == 0)
    assert jnp.all(k_prev[0, 3] == 0)
    # Latent update at s_prev = 0 is exactly x0_hat.
    assert jnp.allclose(x_prev, x0_hat)

    # A coin that never unmasks leaves every token alone, including masks.
    x_stay, k_stay, _ = hc_dlm.reverse_step(
        params, x_t, k_t, jnp.array(0.4), jnp.array(0.4), random.PRNGKey(12)
    )
    assert jnp.all(k_stay == k_t)


def test_following_latent_depends_on_new_tokens():
    params = _scaffold_params()
    # Force the predictor to emit a fixed real token (class 0) so the new
    # scaffold is deterministic and different from the mask row.
    params["predictor"]["b2"] = jnp.array([20.0, -20.0, -20.0])
    x_t = jnp.zeros((1, 3, 2))
    k_t = jnp.full((1, 3), hc_dlm.mask_id(params))
    # s_prev = 0 unmasks every masked site, so the next scaffold is fully new.
    x_prev, k_prev, x0_hat = hc_dlm.reverse_step(
        params, x_t, k_t, jnp.array(1.0), jnp.array(0.0), random.PRNGKey(5)
    )
    assert jnp.all(k_prev == 0)
    assert jnp.all(k_t == hc_dlm.mask_id(params))
    # Next latent update reads the updated scaffold, not the mask row.
    eps_new = jnp.ones_like(x_prev)
    s_prev = jnp.array(0.0)
    next_x0 = hc_dlm.denoise(params, x_prev, k_prev, s_prev)
    stale_x0 = hc_dlm.denoise(params, x_prev, k_t, s_prev)
    assert not jnp.allclose(next_x0, stale_x0)
    next_latent = (1.0 - s_prev) * next_x0 + s_prev * eps_new
    stale_latent = (1.0 - s_prev) * stale_x0 + s_prev * eps_new
    assert not jnp.allclose(next_latent, stale_latent)
    # This step's x0_hat itself used the old scaffold (mask embedding [-1, 0.25]).
    assert jnp.allclose(x0_hat, jnp.broadcast_to(jax.nn.gelu(jnp.array([-1.0, 0.25])), x0_hat.shape))


def test_objective_uses_closed_form_entropy_and_is_deterministic():
    key = random.PRNGKey(4)
    params = hc_dlm.init_params(key, vocab=7, d=4, hidden=8, embed_dim=4, time_dim=4)
    k0 = jnp.array([[0, 1, 2, 3], [4, 5, 1, 0]])
    eps_enc = random.normal(random.PRNGKey(5), (2, 4, 4))
    eps_flow = random.normal(random.PRNGKey(6), (2, 4, 4))
    s = jnp.array([0.2, 0.8])
    mask_key = random.PRNGKey(8)
    omega = jnp.array([1.0, 2.0])
    j1 = hc_dlm.objective(
        params, k0, eps_enc, eps_flow, s, mask_key, omega=omega
    )
    j2 = hc_dlm.objective(
        params, k0, eps_enc, eps_flow, s, mask_key, omega=omega
    )
    assert j1 == j2
    assert jnp.isfinite(j1)

    terms = hc_dlm.objective_terms(
        params,
        k0,
        eps_enc,
        eps_flow,
        s,
        mask_key,
        lambda_recon=1.0,
        lambda_ent=0.01,
        lambda_cont=1.0,
        omega=omega,
    )
    x0, _mean, log_std = hc_dlm.encode(params, k0, eps_enc)
    entropy = hc_dlm.gaussian_entropy(jnp.exp(2.0 * log_std), axis=(-1, -2))
    x0_sg = jax.lax.stop_gradient(x0)
    nll = -hc_dlm.sequence_log_prob(hc_dlm.token_logits(params, x0_sg), k0)
    x_t = hc_dlm.interpolant(x0, eps_flow, s)
    k_t = hc_dlm.absorbing_forward(k0, s, mask_key, hc_dlm.mask_id(params))
    x0_hat = hc_dlm.denoise(params, x_t, k_t, s)
    sq = jnp.sum((x0_hat - x0_sg) ** 2, axis=(-1, -2))
    expected = jnp.mean(nll) + 0.01 * jnp.mean(-entropy) + jnp.mean(omega * sq)
    assert jnp.allclose(terms["J"], expected)
    assert jnp.allclose(terms["ent"], jnp.mean(-entropy))


def test_copy_task_loss_decreases():
    result = hc_dlm.train_copy_task(steps=200, seed=0, batch=32, lr=1e-3)
    assert jnp.isfinite(result["initial_loss"])
    assert jnp.isfinite(result["final_loss"])
    assert jnp.all(jnp.isfinite(result["step_losses"]))
    assert result["final_loss"] < result["initial_loss"]
