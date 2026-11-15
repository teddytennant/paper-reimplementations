"""Numerical checks of GraLoD equations (arXiv:2609.16578) on tiny NHWC tensors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_gralod.align import apply_aligner, init_aligner, map_pyramid_to_stage, resize_to_lattice
from jax_gralod.backbone import encode, forward_infer as restore_infer, init_restorer
from jax_gralod.estimator import (
    apply_adaptor,
    apply_estimator,
    apply_probe,
    init_adaptor,
    init_estimator,
    init_probe,
    residual_inject,
)
from jax_gralod.losses import (
    calibration_ramp,
    l_cal,
    l_probe,
    l_str,
    local_error,
    msfc_costs,
    msfc_distribution,
    msfc_target,
    reconstruction_loss,
    stage_structure_penalty,
    total_loss,
)
from jax_gralod.model import GraLoDConfig, drop_probes, forward_infer, init
from jax_gralod.ops import antialiased_resize, box_filter, simple_gate
from jax_gralod.query import lod_query, lod_query_lerp, triangular_weights


B, H, L, C = 1, 8, 3, 8


def _z_levels(seed: int = 0):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, L)
    return [jax.random.normal(k, (B, H, H, C)) for k in keys]


# --- Eq. 13 / 14 / 12 / 15: continuous query ---------------------------------


def test_eq14_integer_lambda_selects_that_level():
    zs = _z_levels()
    for k in range(L):
        lam = jnp.full((B, H, H, 1), float(k))
        f = lod_query(lam, zs)
        np.testing.assert_allclose(f, zs[k], atol=1e-5)
        np.testing.assert_allclose(lod_query_lerp(lam, zs), zs[k], atol=1e-5)


def test_eq13_triangular_weights_partition_of_unity():
    lam = jnp.array([0.0, 0.25, 1.0, 1.6, 2.0], dtype=jnp.float32)
    w = triangular_weights(lam, L)
    assert jnp.all(w >= 0.0)
    np.testing.assert_allclose(w.sum(axis=-1), jnp.ones((5,)), atol=1e-6)
    nz = (w > 1e-6).sum(axis=-1)
    assert int(nz.max()) <= 2


def test_eq12_lerp_matches_two_level_formula():
    zs = _z_levels(1)
    lam = jnp.full((B, H, H, 1), 0.37)
    f_tri = lod_query(lam, zs)
    f_lerp = lod_query_lerp(lam, zs)
    alpha = 0.37
    expected = (1.0 - alpha) * zs[0] + alpha * zs[1]
    np.testing.assert_allclose(f_tri, expected, atol=1e-5)
    np.testing.assert_allclose(f_lerp, expected, atol=1e-5)
    np.testing.assert_allclose(f_tri, f_lerp, atol=1e-5)


def test_eq15_finite_diff_matches_level_gap():
    zs = _z_levels(2)
    lam0 = jnp.full((B, H, H, 1), 0.41)
    eps = 1e-3
    num = (lod_query(lam0 + eps, zs) - lod_query(lam0, zs)) / eps
    expected = zs[1] - zs[0]
    np.testing.assert_allclose(num, expected, atol=2e-3, rtol=2e-3)

    def mean_f(lam):
        return lod_query(lam, zs).sum()

    g = jax.grad(mean_f)(lam0)
    # d(sum F)/d lambda_{bhw} = sum_c (Z^{1}-Z^{0})_{bhwc}
    np.testing.assert_allclose(g[..., 0], expected.sum(axis=-1), atol=1e-4)


def test_lambda_clamped_to_unit_interval_of_levels():
    zs = _z_levels()
    lam = jnp.array([-2.0, 99.0]).reshape(1, 1, 2, 1)
    z_tiny = [jnp.reshape(z[0, 0, 0], (1, 1, 1, C)) for z in zs]
    # query clamps internally
    f = lod_query(jnp.full((1, 1, 1, 1), 99.0), z_tiny)
    np.testing.assert_allclose(f, z_tiny[L - 1], atol=1e-5)
    f0 = lod_query(jnp.full((1, 1, 1, 1), -5.0), z_tiny)
    np.testing.assert_allclose(f0, z_tiny[0], atol=1e-5)


# --- Eq. 25 / 8 / 26: MSFC cost prefers coarser iff error drop beats gamma ---


def test_eq25_coarser_preferred_only_when_error_drop_exceeds_penalty():
    gamma = 0.05
    l_idx, k_idx = 0, 2
    thresh = gamma * (k_idx - l_idx) / (L - 1)
    # drop larger than thresh -> coarser cheaper
    e = jnp.array([0.20, 1.0, 0.20 - thresh - 0.01], dtype=jnp.float32)
    c = msfc_costs(e, gamma=gamma)
    assert float(c[k_idx]) < float(c[l_idx])
    # drop smaller than thresh -> finer cheaper
    e2 = jnp.array([0.20, 1.0, 0.20 - thresh + 0.01], dtype=jnp.float32)
    c2 = msfc_costs(e2, gamma=gamma)
    assert float(c2[k_idx]) > float(c2[l_idx])


def test_q_is_softmax_of_neg_c_over_tau():
    e = jnp.array([0.4, 0.2, 0.5], dtype=jnp.float32)
    c = msfc_costs(e, gamma=0.05)
    tau = 0.20
    q = msfc_distribution(c, tau=tau)
    np.testing.assert_allclose(q, jax.nn.softmax(-c / tau), atol=1e-6)
    np.testing.assert_allclose(float(q.sum()), 1.0, atol=1e-6)


def test_l_cal_stop_gradient_on_lambda_star():
    lam = jnp.array([0.5, 1.2], dtype=jnp.float32)
    errors = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]], dtype=jnp.float32)

    def loss_from_errors(err):
        star, _q, _c = msfc_target(err, gamma=0.05, tau=0.20)
        return l_cal([lam.reshape(1, 1, 2, 1)], [star.reshape(1, 1, 2)])

    g_err = jax.grad(loss_from_errors)(errors)
    np.testing.assert_allclose(g_err, jnp.zeros_like(errors), atol=1e-8)

    star, _q, _c = msfc_target(errors, gamma=0.05, tau=0.20)

    def loss_from_lam(pred):
        return l_cal([pred.reshape(1, 1, 2, 1)], [star.reshape(1, 1, 2)])

    g_lam = jax.grad(loss_from_lam)(lam)
    assert float(jnp.max(jnp.abs(g_lam))) > 0.0
    expected = jnp.sign(lam - jax.lax.stop_gradient(star)) / 2.0
    np.testing.assert_allclose(g_lam, expected, atol=1e-5)


# --- Eq. 10: structure-aware regularization ----------------------------------


def test_l_str_equal_x_penalizes_lambda_jump_zero_when_constant():
    x = jnp.ones((1, 2, 2, 3))
    lam_jump = jnp.array([[[0.0, 1.0], [0.0, 1.0]]], dtype=jnp.float32)
    # 4-connected, w=1: four directed horizontal edges of |d lambda|=1, vertical 0
    # sum = 4, |Omega|=4, penalty = 1
    pen = stage_structure_penalty(lam_jump, x, beta=10.0)
    np.testing.assert_allclose(float(pen), 1.0, atol=1e-5)

    lam_const = jnp.ones((1, 2, 2))
    pen0 = stage_structure_penalty(lam_const, x, beta=10.0)
    np.testing.assert_allclose(float(pen0), 0.0, atol=1e-6)

    # multi-stage average
    loss = l_str([lam_const, lam_const], [x, x], beta=10.0)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)


# --- probes, residual, inference ---------------------------------------------


def test_probe_l1_and_stopgrad_inputs_do_not_flow_into_d_or_z():
    key = jax.random.PRNGKey(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    p = init_probe(k1, dec_ch=C, lod_ch=C, out_ch=3)
    d = jax.random.normal(k2, (B, H, H, C))
    z = jax.random.normal(k3, (B, H, H, C))
    x = jax.random.normal(k4, (B, H, H, 3))

    xhat = apply_probe(p, d, z, detach_inputs=True)
    assert xhat.shape == x.shape
    probe_loss = jnp.mean(jnp.abs(xhat - x))
    assert jnp.isfinite(probe_loss)

    def loss_dz(d_, z_):
        pred = apply_probe(p, d_, z_, detach_inputs=True)
        return jnp.mean(jnp.abs(pred - x))

    gd, gz = jax.grad(loss_dz, argnums=(0, 1))(d, z)
    np.testing.assert_allclose(gd, jnp.zeros_like(d), atol=1e-8)
    np.testing.assert_allclose(gz, jnp.zeros_like(z), atol=1e-8)

    def loss_p(params):
        pred = apply_probe(params, d, z, detach_inputs=True)
        return jnp.mean(jnp.abs(pred - x))

    gp = jax.grad(loss_p)(p)
    leaves = jax.tree.leaves(gp)
    assert any(float(jnp.max(jnp.abs(v))) > 0.0 for v in leaves)

    # L_probe helper
    loss = l_probe([[xhat, xhat]], [x])
    np.testing.assert_allclose(float(loss), float(probe_loss), atol=1e-6)


def test_residual_adaptor_zero_h_leaves_decoder_unchanged():
    d = jax.random.normal(jax.random.PRNGKey(4), (B, H, H, C))
    h = jnp.zeros_like(d)
    np.testing.assert_allclose(residual_inject(d, h), d)

    # Zero-weight adaptor also yields H=0
    f_lod = jax.random.normal(jax.random.PRNGKey(5), (B, H, H, C))
    p = init_adaptor(jax.random.PRNGKey(6), dec_ch=C, lod_ch=C)
    p_zero = jax.tree.map(jnp.zeros_like, p)
    np.testing.assert_allclose(apply_adaptor(p_zero, d, f_lod), jnp.zeros_like(d), atol=1e-6)
    np.testing.assert_allclose(residual_inject(d, apply_adaptor(p_zero, d, f_lod)), d, atol=1e-6)


def test_inference_path_does_not_require_probe_params():
    cfg = GraLoDConfig(
        encoder_channels=(C, 2 * C, 4 * C),
        decoder_channels=(C, 2 * C, 4 * C),
        lod_channels=C,
        out_channels=3,
    )
    params = init(jax.random.PRNGKey(7), cfg)
    params_inf = drop_probes(params)
    assert all("P" not in st for st in params_inf["stages"])

    key = jax.random.PRNGKey(8)
    feats = [
        jax.random.normal(k, (1, H // (2**i), H // (2**i), ch))
        for i, (k, ch) in enumerate(zip(jax.random.split(key, 3), cfg.encoder_channels))
    ]
    decs = [
        jax.random.normal(k, (1, H // (2**i), H // (2**i), ch))
        for i, (k, ch) in enumerate(zip(jax.random.split(key, 3), cfg.decoder_channels))
    ]
    d_tildes, lams = forward_infer(params_inf, feats, decs, cfg)
    assert len(d_tildes) == cfg.n_stages
    for lam in lams:
        assert float(lam.min()) >= 0.0
        assert float(lam.max()) <= float(cfg.n_levels - 1)
        assert jnp.all(jnp.isfinite(lam))


def test_estimator_lambda_in_range():
    cfg_n = L
    key = jax.random.PRNGKey(9)
    p = init_estimator(key, dec_ch=C, lod_ch=C, n_levels=cfg_n)
    d = jax.random.normal(jax.random.PRNGKey(10), (B, H, H, C))
    zs = _z_levels(11)
    lam = apply_estimator(p, d, zs, n_levels=cfg_n)
    assert lam.shape == (B, H, H, 1)
    assert float(lam.min()) >= 0.0
    assert float(lam.max()) <= float(cfg_n - 1)


def test_total_loss_weights_eq11():
    rec, cal, st, pr = 2.0, 3.0, 4.0, 5.0
    loss = total_loss(rec, cal, st, pr, lambda_cal=0.10, lambda_str=0.05, lambda_probe=0.10)
    np.testing.assert_allclose(float(loss), 2.0 + 0.10 * 3.0 + 0.05 * 4.0 + 0.10 * 5.0)
    # default ramp r=1
    np.testing.assert_allclose(float(calibration_ramp(0, 0, 0.10)), 0.10)


def test_reconstruction_is_l1():
    a = jnp.array([0.0, 1.0, -1.0])
    b = jnp.array([0.5, 1.0, 1.0])
    np.testing.assert_allclose(float(reconstruction_loss(a, b)), float(jnp.mean(jnp.abs(a - b))))


# --- aligner / resize / dummy backbone ---------------------------------------


def test_simple_gate_and_aligner_shapes():
    x = jnp.arange(8, dtype=jnp.float32).reshape(1, 1, 1, 8)
    g = simple_gate(x)
    np.testing.assert_allclose(g, x[..., :4] * x[..., 4:])
    p = init_aligner(jax.random.PRNGKey(12), in_ch=16, lod_ch=C)
    f = jax.random.normal(jax.random.PRNGKey(13), (B, H, H, 16))
    z = apply_aligner(p, f)
    assert z.shape == (B, H, H, C)


def test_antialiased_downsample_is_not_nearest():
    x = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4, 1)
    y = antialiased_resize(x, 2, 2)
    nearest = x[:, ::2, ::2, :]
    assert not np.allclose(np.asarray(y), np.asarray(nearest))
    same = resize_to_lattice(x, 4, 4)
    np.testing.assert_allclose(same, x)
    up = antialiased_resize(x, 8, 8)
    assert up.shape == (1, 8, 8, 1)


def test_box_filter_uniform_is_identity():
    x = jnp.ones((1, 6, 6, 2))
    y = box_filter(x, ksize=5)
    np.testing.assert_allclose(y, x, atol=1e-5)
    err = local_error(x, x, ksize=5)
    np.testing.assert_allclose(err, jnp.zeros((1, 6, 6)), atol=1e-6)


def test_map_pyramid_to_stage_resolutions():
    zs = [
        jnp.ones((1, 8, 8, C)),
        jnp.ones((1, 4, 4, C)),
        jnp.ones((1, 2, 2, C)),
    ]
    mapped = map_pyramid_to_stage(zs, 4, 4)
    assert all(z.shape == (1, 4, 4, C) for z in mapped)


def test_dummy_restorer_infer_and_l4_support():
    cfg3 = GraLoDConfig(
        encoder_channels=(8, 16, 32),
        decoder_channels=(8, 16, 32),
        lod_channels=8,
    )
    params = init_restorer(jax.random.PRNGKey(14), cfg3)
    y = jax.random.uniform(jax.random.PRNGKey(15), (1, 16, 16, 3))
    rgb = restore_infer(params, y, cfg3)
    assert rgb.shape == y.shape
    assert jnp.all(jnp.isfinite(rgb))
    # dropping probes still infers
    params["gralod"] = drop_probes(params["gralod"])
    rgb2 = restore_infer(params, y, cfg3)
    assert rgb2.shape == y.shape

    cfg4 = GraLoDConfig(
        encoder_channels=(8, 16, 24, 32),
        decoder_channels=(8, 16, 24, 32),
        lod_channels=8,
    )
    p4 = init_restorer(jax.random.PRNGKey(16), cfg4)
    y4 = jax.random.uniform(jax.random.PRNGKey(17), (1, 16, 16, 3))
    feats = encode(p4["backbone"], y4)
    assert [f.shape[1] for f in feats] == [16, 8, 4, 2]
    rgb4 = restore_infer(p4, y4, cfg4)
    assert rgb4.shape == y4.shape
