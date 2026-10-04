"""Tests for the transcribed NormPaST-Risk equations."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normpast_risk.loss import classification_loss, normative_loss, segment_loss, total_loss
from normpast_risk.model import (
    NormPaSTRisk,
    apply_ssm_layer,
    init_params,
    paper_air_gate,
    paper_air_modulate,
    ssm_scan,
    transition_A,
)
from normpast_risk.segments import aggregate_segments, project_tokens
import train


def test_aggregation_does_not_mix_paper_and_air():
    gamma = 8
    dim = 4
    paper = jnp.full((1, gamma // 2, dim), 2.0)
    air = jnp.zeros((1, gamma // 2, dim))
    h = jnp.concatenate([paper, air], axis=1)
    q = jnp.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=jnp.float32)
    out = aggregate_segments(h, q, gamma)
    assert out["paper_mean"].shape == (1, 1, dim)
    assert jnp.allclose(out["paper_mean"], 2.0)
    assert jnp.allclose(out["air_mean"], 0.0)
    assert jnp.allclose(out["n_paper"], 4.0)
    assert jnp.allclose(out["n_air"], 4.0)
    assert jnp.allclose(out["features"][..., -2], 4.0 / gamma)
    assert jnp.allclose(out["features"][..., -1], 4.0 / gamma)

    weight = jnp.zeros((dim, 2 * dim + 2))
    weight = weight.at[0, -2].set(1.0)
    weight = weight.at[1, -1].set(1.0)
    bias = jnp.zeros((dim,))
    token = project_tokens(out["features"], weight, bias)
    assert jnp.allclose(token[0, 0, 0], 4.0 / gamma)
    assert jnp.allclose(token[0, 0, 1], 4.0 / gamma)

    q_more_paper = jnp.array([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=jnp.float32)
    more = aggregate_segments(h, q_more_paper, gamma)
    token_more = project_tokens(more["features"], weight, bias)
    assert not jnp.allclose(token_more[0, 0, 0], token[0, 0, 0])
    assert not jnp.allclose(token_more[0, 0, 1], token[0, 0, 1])


def test_no_paper_points_and_counts_normalized_by_gamma():
    gamma = 8
    dim = 3
    h = jnp.ones((1, gamma, dim))
    q = jnp.zeros((1, gamma), dtype=jnp.float32)
    out = aggregate_segments(h, q, gamma)
    assert jnp.allclose(out["paper_mean"], 0.0)
    assert jnp.allclose(out["air_mean"], 1.0)
    assert jnp.allclose(out["n_paper"], 0.0)
    assert jnp.allclose(out["n_air"], gamma)
    assert jnp.all(jnp.isfinite(out["paper_mean"]))
    assert jnp.all(jnp.isfinite(out["features"]))
    assert jnp.allclose(out["features"][..., -2], 0.0)
    assert jnp.allclose(out["features"][..., -1], 1.0)

    # Padding is masked, so it is not counted as air and does not dilute the mean.
    h_short = jnp.full((2, 5, dim), 3.0)
    q_short = jnp.ones((2, 5), dtype=jnp.float32)
    short = aggregate_segments(h_short, q_short, gamma)
    assert jnp.allclose(short["paper_mean"], 3.0)
    assert jnp.allclose(short["air_mean"], 0.0)
    assert jnp.allclose(short["n_paper"], 5.0)
    assert jnp.allclose(short["n_air"], 0.0)
    assert jnp.allclose(short["features"][..., -2], 5.0 / gamma)
    assert jnp.allclose(short["features"][..., -1], 0.0)
    assert jnp.all(jnp.isfinite(short["paper_mean"]))


def test_ssm_scan_matches_hand_unroll():
    abar = jnp.array(
        [
            [0.5, 0.25, 0.8],
            [0.9, 0.1, 0.3],
            [0.2, 0.7, 0.4],
        ]
    )
    bbar = jnp.array(
        [
            [0.1, -0.2, 0.3],
            [0.0, 0.5, -0.1],
            [0.4, 0.2, 0.1],
        ]
    )
    c = jnp.array(
        [
            [1.5, -0.5, 0.25],
            [0.3, 0.3, -1.0],
            [1.0, 0.0, 0.5],
        ]
    )
    tilde = jnp.array(
        [
            [0.3, 0.7, -0.2],
            [1.0, -0.4, 0.5],
            [0.2, 0.2, 0.2],
        ]
    )
    v0 = jnp.array([0.4, -0.1, 0.0])
    y, v = ssm_scan(abar, bbar, c, tilde, v0=v0)

    v_hand = v0
    for j in range(abar.shape[0]):
        v_hand = abar[j] * v_hand + bbar[j] * tilde[j]
        y_hand = c[j] * v_hand
        assert jnp.allclose(v[j], v_hand)
        assert jnp.allclose(y[j], y_hand)

    y1, v1 = ssm_scan(abar[:1], bbar[:1], c[:1], tilde[:1], v0=v0)
    v_new = abar[0] * v0 + bbar[0] * tilde[0]
    assert jnp.allclose(v1[0], v_new)
    assert jnp.allclose(y1[0], c[0] * v_new)


def test_A_is_strictly_negative():
    raw = jnp.array([0.0, 1.5, -2.0, 4.0, -0.3])
    a_coeff = transition_A(raw)
    assert jnp.allclose(a_coeff, -jax.nn.softplus(raw))
    assert jnp.all(a_coeff < 0.0)

    params = init_params(0, d=8, n_layers=2, n_experts=2, n_tasks=3)
    for layer in params["layers"]:
        a_layer = transition_A(layer["a_raw"])
        assert jnp.all(a_layer < 0.0)
        assert jnp.allclose(a_layer, -jax.nn.softplus(layer["a_raw"]))


def test_modulation_uses_bias_plus_half_gate():
    h = jnp.array([[1.0, -2.0, 0.5]])
    embedding = jnp.array([[0.25, 0.5, -0.5]])
    gate = jnp.array([[0.0, 0.5, 1.0]])
    got = paper_air_modulate(h, embedding, gate)
    expect = h + embedding * (0.75 + 0.5 * gate)
    assert jnp.allclose(got, expect)

    dim = 3
    h_b = jnp.ones((2, 4, dim))
    e_b = jnp.full((2, 4, dim), 0.3)
    weight = jnp.zeros((dim, 2 * dim))
    target = 0.2
    bias = jnp.full((dim,), jnp.log(target / (1.0 - target)))
    g = paper_air_gate(h_b, e_b, weight, bias)
    tilde = paper_air_modulate(h_b, e_b, g)
    assert jnp.allclose(g, target, atol=1e-6)
    assert jnp.allclose(tilde, h_b + e_b * (0.75 + 0.5 * target), atol=1e-5)

    layer = {
        "W_g": weight,
        "b_g": bias,
        "a_raw": jnp.zeros((dim,)),
        "W_delta": jnp.zeros((dim, dim)),
        "b_delta": jnp.zeros((dim,)),
        "W_B": jnp.zeros((dim, dim)),
        "b_B": jnp.zeros((dim,)),
        "W_C": jnp.zeros((dim, dim)),
        "b_C": jnp.zeros((dim,)),
        "ln_scale": jnp.ones((dim,)),
        "ln_bias": jnp.zeros((dim,)),
    }
    e_table = jnp.stack([jnp.full((dim,), 0.3), jnp.full((dim,), -0.4)], axis=0)
    rho = jnp.array([[0.0, 1.0, 0.2, 0.8]])
    _h_out, aux, _rng = apply_ssm_layer(layer, h_b[:1], e_table, rho)
    assert jnp.allclose(aux["g"], target, atol=1e-5)
    assert jnp.allclose(
        aux["tilde_h"],
        h_b[:1] + aux["embedding"] * (0.75 + 0.5 * aux["g"]),
        atol=1e-5,
    )


def test_normative_loss_masks_ad_and_is_positive_when_wrong():
    hat = jnp.zeros((4, 5, 3))
    target = jnp.ones((4, 5, 3))
    all_ad = jnp.ones((4,))
    assert float(normative_loss(hat, target, all_ad)) == 0.0

    healthy = jnp.zeros((4,))
    wrong = normative_loss(hat, target, healthy)
    assert float(wrong) > 0.0

    mixed_y = jnp.array([0.0, 1.0, 0.0, 1.0])
    mixed = normative_loss(hat, target, mixed_y)
    healthy_only = normative_loss(hat[:1], target[:1], jnp.array([0.0]))
    assert jnp.allclose(mixed, healthy_only)

    model = NormPaSTRisk(d=8, n_layers=1, n_experts=2, n_tasks=3, gamma=4, key=1)
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 8, 4))
    q = jnp.ones((4, 8))
    task_id = jnp.zeros((4,), dtype=jnp.int32)
    outputs = model.apply(x, q, task_id)
    ad_losses = total_loss(outputs, jnp.ones((4,)))
    assert float(ad_losses["norm"]) == 0.0
    hc_losses = total_loss(outputs, jnp.zeros((4,)))
    assert float(hc_losses["norm"]) > 0.0


def test_loss_weights_match_weighted_sum():
    model = NormPaSTRisk(d=8, n_layers=1, n_experts=2, n_tasks=3, gamma=4, key=0)
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 8, 4))
    q = jax.random.bernoulli(jax.random.PRNGKey(1), 0.5, (4, 8)).astype(jnp.float32)
    task_id = jnp.array([0, 1, 2, 1])
    y = jnp.array([0.0, 1.0, 0.0, 1.0])
    outputs = model.apply(x, q, task_id)
    losses = total_loss(outputs, y, lambda_seg=0.1, lambda_norm=0.2)
    cls = classification_loss(outputs["sample_prob"], y)
    seg = segment_loss(outputs["segment_risk"], y)
    norm = normative_loss(outputs["hat_next"], outputs["tokens"][:, 1:], y)
    assert jnp.allclose(losses["cls"], cls)
    assert jnp.allclose(losses["seg"], seg)
    assert jnp.allclose(losses["norm"], norm)
    assert jnp.allclose(losses["loss"], cls + 0.1 * seg + 0.2 * norm)


def test_forward_shapes_and_probability_range():
    model = NormPaSTRisk(d=8, n_layers=1, n_experts=2, n_tasks=3, gamma=4, key=0)
    batch, length = 2, 10
    x = jax.random.normal(jax.random.PRNGKey(3), (batch, length, 4))
    q = jax.random.bernoulli(jax.random.PRNGKey(4), 0.4, (batch, length)).astype(jnp.float32)
    task_id = jnp.array([0, 2])
    outputs = model.apply(x, q, task_id)
    n_seg = (length + 4 - 1) // 4
    assert outputs["segment_risk"].shape == (batch, n_seg)
    assert outputs["sample_prob"].shape == (batch,)
    assert jnp.all(outputs["segment_risk"] > 0.0)
    assert jnp.all(outputs["segment_risk"] < 1.0)
    assert jnp.all(outputs["sample_prob"] > 0.0)
    assert jnp.all(outputs["sample_prob"] < 1.0)
    assert jnp.all(jnp.isfinite(outputs["segment_logit"]))
    assert jnp.all(jnp.isfinite(outputs["sample_logit"]))


def test_defaults_and_calibration_init():
    model = NormPaSTRisk()
    assert model.d == 96
    assert model.n_layers == 2
    assert model.n_experts == 4
    assert model.gamma == 128
    assert model.n_tasks == 25
    assert model.dropout_rate == 0.0
    assert model.kernel_sizes == (3, 7)
    assert model.dilations == (1, 2)
    assert float(model.params["risk"]["w_chi"]) == 1.0
    assert float(model.params["risk"]["w_delta"]) == 1.0


def test_training_step_is_finite_and_does_not_explode():
    result = train.run_training(steps=30, seed=0, lr=1e-3)
    losses = result["losses"]
    assert losses.shape == (30,)
    assert jnp.all(jnp.isfinite(losses))
    assert jnp.isfinite(result["loss_after"])
    assert float(result["loss_after"]) <= float(losses[0]) + 1.0
