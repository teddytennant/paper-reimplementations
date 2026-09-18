"""CPU tests for FCx (arXiv:2609.19383). Keep the whole file under ~60s."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_fcx.classifier import accuracy, train_classifier
from jax_fcx.lof import lof_scores
from jax_fcx.losses import hinge_validity, reconstruction_loss, soft_causal_loss
from jax_fcx.train import FCxConfig, make_synthetic_data, train_fcx
from jax_fcx.vae import generate_cf, init_cvae


def test_classifier_trains_above_chance():
    data = make_synthetic_data(n=256, d=8, seed=1)
    x = jnp.asarray(data["x"])
    y = jnp.asarray(data["y"])
    params, _ = train_classifier(jax.random.PRNGKey(1), x, y, hidden=16, steps=120, batch_size=64)
    acc = float(accuracy(params, x, y))
    assert acc > 0.6, acc


def test_vae_proximity_finite():
    data = make_synthetic_data(n=32, d=8, seed=2)
    x = jnp.asarray(data["x"])
    mutable = jnp.asarray(data["mutable_idx"])
    y = jnp.zeros((x.shape[0],), dtype=jnp.int32)
    params = init_cvae(jax.random.PRNGKey(2), d_mutable=int(mutable.shape[0]), z_dim=8, hidden=16, n_linear=3)
    x_cf = generate_cf(params, x, y, mutable, key=jax.random.PRNGKey(3))
    prox = reconstruction_loss(x, x_cf)
    assert jnp.isfinite(prox)
    assert x_cf.shape == x.shape


def test_immutables_copied_unchanged():
    data = make_synthetic_data(n=16, d=8, seed=3)
    x = jnp.asarray(data["x"])
    mutable = jnp.asarray(data["mutable_idx"])
    y = jnp.ones((x.shape[0],), dtype=jnp.int32)
    params = init_cvae(jax.random.PRNGKey(3), d_mutable=int(mutable.shape[0]), z_dim=4, hidden=8, n_linear=3)
    x_cf = generate_cf(params, x, y, mutable, key=jax.random.PRNGKey(4))
    imm = int(data["immutable_idx"][0])
    np.testing.assert_allclose(np.asarray(x_cf[:, imm]), np.asarray(x[:, imm]), atol=1e-6)


def test_soft_causal_zero_when_cov_matches_a():
    b, d = 64, 8
    t = jnp.linspace(-1.0, 1.0, b)
    x = jnp.zeros((b, d))
    x = x.at[:, 0].set(t)
    x = x.at[:, 1].set(t)
    A = jnp.zeros((d, d))
    A = A.at[0, 1].set(1.0)
    loss = soft_causal_loss(x, A, theta=0.1)
    assert jnp.isfinite(loss)
    assert float(loss) < 1e-6, float(loss)


def test_lof_inliers_near_one_vs_outlier():
    key = jax.random.PRNGKey(5)
    inliers = jax.random.normal(key, (24, 4)) * 0.05
    outlier = jnp.array([[40.0, 0.0, 0.0, 0.0]])
    z = jnp.concatenate([inliers, outlier], axis=0)
    scores = lof_scores(z, k=5)
    inlier_lof = float(jnp.mean(scores[:-1]))
    outlier_lof = float(scores[-1])
    assert jnp.isfinite(scores).all()
    assert outlier_lof > inlier_lof
    assert abs(inlier_lof - 1.0) < abs(outlier_lof - 1.0)
    assert outlier_lof > 1.5


def test_validity_hinge_finite():
    logits = jnp.array([[0.2, -0.1], [1.0, 0.0], [-0.5, 0.4]])
    y = jnp.array([1, 0, 1])
    h = hinge_validity(logits, y, margin=1.0)
    assert jnp.isfinite(h)


def test_train_loss_feasibility_validity():
    data = make_synthetic_data(n=128, d=8, seed=7)
    x = jnp.asarray(data["x"])
    y = jnp.asarray(data["y"])
    key = jax.random.PRNGKey(7)
    key, k_clf, k_fcx = jax.random.split(key, 3)
    clf_params, _ = train_classifier(k_clf, x, y, hidden=16, steps=80, batch_size=64)
    cfg = FCxConfig(
        hidden=16,
        n_linear=3,
        z_dim=6,
        steps=30,
        batch_size=32,
        lr=2e-2,
        lambda_rec=0.2,
        lambda_val=3.0,
        lambda_feas=4.0,
        lambda_causal=0.05,
        lambda_spar=0.2,
        lambda_lof=0.05,
        k_lof=5,
    )
    vae_params, logs = train_fcx(
        k_fcx,
        x,
        clf_params,
        config=cfg,
        adjacency=jnp.asarray(data["A"]),
        mutable_idx=jnp.asarray(data["mutable_idx"]),
    )
    loss = logs["loss"]
    feas = logs["feas_violation"]
    val = logs["validity"]
    feas_rate = logs["feasibility_rate"]
    assert all(np.isfinite(loss))
    assert loss[-1] < loss[0], (loss[0], loss[-1])
    assert feas[-1] < feas[0] or feas_rate[-1] > feas_rate[0], (feas[0], feas[-1], feas_rate)
    assert val[-1] > 0.5 or val[-1] > val[0], (val[0], val[-1])
    x_cf = generate_cf(vae_params, logs["eval_x"], logs["eval_y"], logs["mutable_idx"], key=None)
    imm = int(data["immutable_idx"][0])
    np.testing.assert_allclose(np.asarray(x_cf[:, imm]), np.asarray(logs["eval_x"][:, imm]), atol=1e-6)
