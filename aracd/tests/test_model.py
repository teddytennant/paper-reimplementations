"""Full ARACD: residual init, batching, shapes, supervised loss."""

import jax
import jax.numpy as jnp
import numpy as np

from aracd.loss import binary_cross_entropy, supervised_loss
from aracd.model import ARACDConfig, forward, forward_from_features, init_params
from aracd.ncdm import ncdm_base
from aracd.residual import ability_correct, ability_residual, concept_residual, stu_item_residual


def _tiny_config(**kw):
    defaults = dict(
        n_students=5,
        n_items=7,
        n_concepts=4,
        rank=6,
        d_att=8,
        d_aff=3,
        mlp_hidden=5,
        affect_hidden=4,
    )
    defaults.update(kw)
    return ARACDConfig(**defaults)


def _q_matrix(I, K, key=0):
    rng = np.random.default_rng(key)
    q = rng.integers(0, 2, size=(I, K)).astype(np.float32)
    q[:, 0] = 1.0
    return jnp.asarray(q)


def test_residual_init_zero_b_only_interaction_then_zero_r():
    cfg = _tiny_config()
    params = init_params(jax.random.PRNGKey(1), cfg.n_students, cfg.n_items, cfg.n_concepts, config=cfg)
    np.testing.assert_array_equal(np.asarray(params["residual"]["b_s"]), 0)
    np.testing.assert_array_equal(np.asarray(params["residual"]["b_i"]), 0)
    np.testing.assert_array_equal(np.asarray(params["residual"]["b_k"]), 0)
    np.testing.assert_array_equal(np.asarray(params["residual"]["b_s_k"]), 0)

    s, i = 2, 3
    u = params["residual"]["u_s"][s]
    v = params["residual"]["v_i"][i]
    r_si = stu_item_residual(
        params["residual"]["b_s"][s],
        params["residual"]["b_i"][i],
        u,
        v,
        gamma_si=params["gamma_si"],
    )
    d = u.shape[-1]
    only_int = float(params["gamma_si"]) * float(np.dot(np.asarray(u), np.asarray(v))) / np.sqrt(d)
    np.testing.assert_allclose(np.asarray(r_si), only_int, rtol=1e-6, atol=1e-6)

    # zero low-rank factors as well: r_stu-item = 0, r_concept = 0 => r_ability = 0
    u0 = jnp.zeros_like(u)
    v0 = jnp.zeros_like(v)
    r_si0 = stu_item_residual(0.0, 0.0, u0, v0, gamma_si=params["gamma_si"])
    alpha = jnp.array([0.25, 0.25, 0.25, 0.25], dtype=jnp.float32)
    r_c0 = concept_residual(alpha, params["residual"]["b_k"], params["residual"]["b_s_k"][s])
    r_ab = ability_residual(r_si0, r_c0, params["residual"]["w_si"], params["residual"]["w_c"])
    np.testing.assert_allclose(np.asarray(r_si0), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(r_c0), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(r_ab), 0.0, atol=1e-7)

    y_base = jnp.array(0.62, dtype=jnp.float32)
    y_star = ability_correct(y_base, r_ab)
    np.testing.assert_allclose(np.asarray(y_star), np.asarray(y_base), rtol=1e-5, atol=1e-6)


def test_forward_zero_factors_y_star_equals_y_base():
    cfg = _tiny_config()
    params = init_params(jax.random.PRNGKey(2), cfg.n_students, cfg.n_items, cfg.n_concepts, config=cfg)
    res = params["residual"]
    zeros = {
        **params,
        "residual": {
            **res,
            "b_s": jnp.zeros_like(res["b_s"]),
            "b_i": jnp.zeros_like(res["b_i"]),
            "u_s": jnp.zeros_like(res["u_s"]),
            "v_i": jnp.zeros_like(res["v_i"]),
            "b_k": jnp.zeros_like(res["b_k"]),
            "b_s_k": jnp.zeros_like(res["b_s_k"]),
        },
    }
    q = _q_matrix(cfg.n_items, cfg.n_concepts)
    s_ids = jnp.array([0, 1, 4])
    i_ids = jnp.array([1, 2, 5])
    out = forward(zeros, s_ids, i_ids, q)
    np.testing.assert_allclose(np.asarray(out["r_ability"]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out["y_star"]), np.asarray(out["y_base"]), rtol=1e-5, atol=1e-6)


def test_shapes_students_items_concepts_batch():
    S, I, K, B = 6, 8, 5, 4
    cfg = _tiny_config(n_students=S, n_items=I, n_concepts=K, d_aff=3, d_att=8, rank=6)
    params = init_params(jax.random.PRNGKey(3), S, I, K, config=cfg)
    q = _q_matrix(I, K)
    s_ids = jnp.array([0, 2, 5, 1])
    i_ids = jnp.array([1, 0, 7, 3])
    out = forward(params, s_ids, i_ids, q)
    assert out["y_hat"].shape == (B,)
    assert out["y_star"].shape == (B,)
    assert out["y_base"].shape == (B,)
    assert out["x_si"].shape == (B, K)
    assert out["r_stu_item"].shape == (B,)
    assert out["r_concept"].shape == (B,)
    assert out["r_ability"].shape == (B,)
    assert out["alpha"].shape == (B, K)
    assert out["scores"].shape == (B, K)
    assert out["q_att"].shape == (B, cfg.d_att)
    assert out["a_si"].shape == (B, cfg.d_aff)
    assert out["g"].shape == (B,)
    assert out["s"].shape == (B,)
    np.testing.assert_allclose(np.asarray(out["alpha"].sum(axis=-1)), 1.0, rtol=1e-5, atol=1e-5)


def test_batched_matches_unbatched():
    cfg = _tiny_config()
    params = init_params(jax.random.PRNGKey(4), cfg.n_students, cfg.n_items, cfg.n_concepts, config=cfg)
    q = _q_matrix(cfg.n_items, cfg.n_concepts)
    s_ids = jnp.array([0, 1, 3, 4])
    i_ids = jnp.array([2, 0, 6, 1])
    batched = forward(params, s_ids, i_ids, q)

    ncdm = params["ncdm"]
    res = params["residual"]
    aff = params["affect"]
    ncdm_mlp = (ncdm["mlp_w1"], ncdm["mlp_b1"], ncdm["mlp_w2"], ncdm["mlp_b2"])

    for b, (s, i) in enumerate(zip(np.asarray(s_ids), np.asarray(i_ids))):
        e_s = jax.nn.sigmoid(ncdm["student_mastery"][s])
        d_kc = jax.nn.sigmoid(ncdm["item_diff"][i])
        d_disc = jax.nn.sigmoid(ncdm["item_disc"][i])
        q_i = q[i]
        one = forward_from_features(
            e_s,
            d_disc,
            d_kc,
            q_i,
            ncdm_mlp,
            res["b_s"][s],
            res["b_i"][i],
            res["u_s"][s],
            res["v_i"][i],
            params["gamma_si"],
            res["w_si"],
            res["w_c"],
            res["b_k"],
            res["b_s_k"][s],
            res["e_s_att"][s],
            res["v_i_att"][i],
            res["k_k_att"],
            aff["a_s"][s],
            aff,
            ln_eps=params["ln_eps"],
            prob_eps=params["prob_eps"],
        )
        for name in ("y_hat", "y_star", "y_base", "r_ability", "r_stu_item", "r_concept", "g", "s"):
            np.testing.assert_allclose(
                np.asarray(batched[name][b]),
                np.asarray(one[name]),
                rtol=1e-5,
                atol=1e-6,
                err_msg=name,
            )
        np.testing.assert_allclose(np.asarray(batched["alpha"][b]), np.asarray(one["alpha"]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(batched["x_si"][b]), np.asarray(one["x_si"]), rtol=1e-5, atol=1e-6)


def test_vmap_matches_batch_ncdm():
    B, K = 3, 4
    rng = np.random.default_rng(5)
    e = jnp.asarray(rng.random((B, K)).astype(np.float32))
    disc = jnp.asarray(rng.random((B, K)).astype(np.float32))
    diff = jnp.asarray(rng.random((B, K)).astype(np.float32))
    q = jnp.asarray(rng.integers(0, 2, (B, K)).astype(np.float32))
    q = q.at[:, 0].set(1.0)
    w1 = jnp.asarray(rng.normal(size=(K, 3)).astype(np.float32) * 0.1)
    b1 = jnp.zeros((3,), dtype=jnp.float32)
    w2 = jnp.asarray(rng.normal(size=(3,)).astype(np.float32) * 0.1)
    b2 = jnp.zeros((), dtype=jnp.float32)

    y_b, x_b = ncdm_base(e, disc, diff, q, w1, b1, w2, b2)

    def one(es, di, df, qi):
        return ncdm_base(es, di, df, qi, w1, b1, w2, b2)

    y_v, x_v = jax.vmap(one)(e, disc, diff, q)
    np.testing.assert_allclose(np.asarray(y_b), np.asarray(y_v), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(x_b), np.asarray(x_v), rtol=1e-6, atol=1e-6)


def test_supervised_loss_bce_and_optional_mse():
    y_hat = jnp.array([0.2, 0.8], dtype=jnp.float32)
    y = jnp.array([0.0, 1.0], dtype=jnp.float32)
    bce = binary_cross_entropy(y_hat, y)
    a_si = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    a_gt = jnp.array([[1.0, 2.0], [3.0, 5.0]], dtype=jnp.float32)
    loss = supervised_loss(y_hat, y, a_si=a_si, a_gt=a_gt, lambda_a=0.5)
    mse = np.mean((np.asarray(a_si) - np.asarray(a_gt)) ** 2)
    np.testing.assert_allclose(np.asarray(loss), float(bce) + 0.5 * mse, rtol=1e-5, atol=1e-6)
    only = supervised_loss(y_hat, y)
    np.testing.assert_allclose(np.asarray(only), np.asarray(bce), rtol=1e-6)


def test_forward_and_loss_finite():
    cfg = _tiny_config()
    params = init_params(jax.random.PRNGKey(6), cfg.n_students, cfg.n_items, cfg.n_concepts, config=cfg)
    q = _q_matrix(cfg.n_items, cfg.n_concepts)
    s_ids = jnp.arange(4) % cfg.n_students
    i_ids = jnp.arange(4) % cfg.n_items
    out = forward(params, s_ids, i_ids, q)
    y = jnp.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    loss = supervised_loss(out["y_hat"], y, a_si=out["a_si"], a_gt=jnp.zeros_like(out["a_si"]))
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(out["y_hat"]))
    grads = jax.grad(lambda p: binary_cross_entropy(forward(p, s_ids, i_ids, q)["y_hat"], y))(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
