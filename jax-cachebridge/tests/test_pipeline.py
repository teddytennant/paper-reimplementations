import jax.numpy as jnp
import pytest

from cachebridge import evaluate, pipeline, support
from cachebridge.rope import rope_angles

CHUNK = 64
P_LOCAL = 2 * 16


def _fit_and_score(toy_pair, kind, assign, weights=None):
    sk, sv, tk, tv = toy_pair["flat_train"]
    hsk, hsv, htk, htv = toy_pair["flat_held"]
    chunk = None if kind == "full" else CHUNK
    m = pipeline.fit_mapper(sk, sv, tk, tv, toy_pair["layers"], assign,
                            kind=kind, weights=weights, chunk=chunk)
    pk, pv = pipeline.apply_mapper(m, hsk, hsv)
    drift = evaluate.attention_drift(toy_pair["held"], pk, pv, toy_pair["sample_pos"],
                                     toy_pair["boundaries"],
                                     toy_pair["tgt_cfg"].q_per_kv,
                                     toy_pair["tgt_cfg"].head_dim)
    return dict(mapper=m, r2_k=evaluate.r2(pk, htk), r2_v=evaluate.r2(pv, htv), drift=drift)


@pytest.fixture(scope="module")
def arms(toy_pair):
    w, alphas, tau = pipeline.attn_repair_weights(
        toy_pair["train"], toy_pair["sample_pos"], toy_pair["boundaries"],
        toy_pair["tgt_cfg"].q_per_kv, P_LOCAL)
    assign = toy_pair["assign"]
    return dict(
        full=_fit_and_score(toy_pair, "full", assign),
        local=_fit_and_score(toy_pair, "local", assign),
        bridge=_fit_and_score(toy_pair, "local", assign, weights=w),
        weights=w, alphas=alphas, tau=tau,
    )


def test_head_local_uses_hs_times_fewer_coefficients(arms, toy_pair):
    """Eqs 12-13 on the fitted artifacts, not just the formula."""
    ratio = arms["full"]["mapper"].n_coefficients / arms["local"]["mapper"].n_coefficients
    assert ratio == toy_pair["src_cfg"].n_kv_heads
    assert arms["bridge"]["mapper"].n_coefficients == arms["local"]["mapper"].n_coefficients


def test_head_local_transfers_at_least_as_well_as_full_head(arms):
    """O1: the wider fan-in buys calibration fit that does not survive to held-out data."""
    assert arms["local"]["r2_k"] > arms["full"]["r2_k"]
    assert arms["local"]["r2_v"] > arms["full"]["r2_v"]
    assert arms["local"]["drift"] < arms["full"]["drift"]


def test_attn_repair_trades_coordinate_fit_for_receiver_drift(arms):
    """O2: R^2 moves the wrong way while the receiver-composed error improves."""
    assert arms["bridge"]["drift"] < arms["local"]["drift"]
    assert arms["bridge"]["r2_k"] < arms["local"]["r2_k"]
    assert arms["local"]["r2_k"] - arms["bridge"]["r2_k"] < 0.05


def test_attn_repair_weights_are_mean_one_and_bounded(arms, toy_pair):
    for w in arms["weights"]:
        assert w.shape[:2] == (toy_pair["tgt_cfg"].n_layers, toy_pair["tgt_cfg"].n_kv_heads)
        assert jnp.all(w >= 0)
        assert jnp.max(jnp.abs(jnp.mean(w, axis=-1) - 1.0)) < 1e-10
    for alpha in arms["alphas"]:
        assert jnp.all((alpha >= 0) & (alpha <= 1))


def test_identity_assignment_beats_every_cyclic_shift(toy_pair):
    """The figure 5 ablation at toy scale, on held-out reconstruction and drift."""
    ht = toy_pair["tgt_cfg"].n_kv_heads
    base = _fit_and_score(toy_pair, "local", support.identity_assignment(ht))
    for shift in range(1, ht):
        shifted = _fit_and_score(toy_pair, "local", support.cyclic_assignment(ht, shift))
        assert base["r2_k"] > shifted["r2_k"] + 0.2, shift
        assert base["r2_v"] > shifted["r2_v"] + 0.2, shift
        assert base["drift"] < shifted["drift"], shift


def test_fused_and_one_shot_paths_give_the_same_mapper(toy_pair):
    """Section 3.4: the construction path must not change the artifact."""
    sk, sv, tk, tv = toy_pair["flat_train"]
    args = (sk, sv, tk, tv, toy_pair["layers"], toy_pair["assign"])
    streamed = pipeline.fit_mapper(*args, kind="local", chunk=CHUNK)
    one_shot = pipeline.fit_mapper(*args, kind="local", chunk=None)
    assert jnp.max(jnp.abs(streamed.w_k - one_shot.w_k)) < 1e-9
    assert jnp.max(jnp.abs(streamed.b_v - one_shot.b_v)) < 1e-9


def test_content_space_round_trip_through_the_online_transforms(toy_pair):
    """Figure 2's 'remove source position' and 'apply target position' must compose."""
    content = toy_pair["flat_train"][0][0]         # (n, Hs, ds)
    pos = jnp.arange(content.shape[0])
    cache = pipeline.content_to_cache(content, pos, toy_pair["src_cfg"].head_dim)
    back = pipeline.cache_to_content(cache, pos, toy_pair["src_cfg"].head_dim)
    assert jnp.max(jnp.abs(back - content)) < 1e-12
    ang = rope_angles(pos, toy_pair["src_cfg"].head_dim)[..., None, :]
    assert jnp.max(jnp.abs(jnp.sum(cache**2, -1) - jnp.sum(content**2, -1))) < 1e-11
    assert ang.shape == (content.shape[0], 1, toy_pair["src_cfg"].head_dim // 2)
