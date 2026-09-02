"""End-to-end run on the toy pair: layer selection, three fitted arms, figure 5 ablation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import evaluate, layer_select, pipeline, support, toy
from .fused_fit import scratch_elements
from .ridge import LAMBDA_DEFAULT

SRC_CFG = toy.GQAConfig(n_layers=4, n_kv_heads=4, q_per_kv=2, head_dim=16)
TGT_CFG = toy.GQAConfig(n_layers=4, n_kv_heads=4, q_per_kv=3, head_dim=16)
SEQ_LEN = 64
STRIDE = 4
K_LAYERS = 2
CHUNK = 96


def build(seed: int = 0, n_train: int = 24, n_eval: int = 8):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    src_params = toy.init_source(keys[0], SRC_CFG)
    tgt_params = toy.init_target(keys[1], TGT_CFG)
    plant = toy.init_plant(keys[2], SRC_CFG, TGT_CFG)
    train = toy.make_traces(keys[3], SRC_CFG, TGT_CFG, plant, n_train, SEQ_LEN,
                            src_params, tgt_params)
    held = toy.make_traces(keys[4], SRC_CFG, TGT_CFG, plant, n_eval, SEQ_LEN,
                           src_params, tgt_params)
    return plant, train, held


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    plant, train, held = build()
    sample_pos = jnp.arange(0, SEQ_LEN, STRIDE)
    boundaries = jnp.unique(jnp.int32(jnp.round(jnp.geomspace(12, SEQ_LEN - 1, 8))))
    ht, hs = TGT_CFG.n_kv_heads, SRC_CFG.n_kv_heads
    ds, dt = SRC_CFG.head_dim, TGT_CFG.head_dim
    p_local, p_full = K_LAYERS * ds, K_LAYERS * hs * ds

    sk, sv, tk, tv = toy.flatten_positions(train, sample_pos)
    hsk, hsv, htk, htv = toy.flatten_positions(held, sample_pos)
    n = sk.shape[1]
    assign = support.identity_assignment(ht)
    layers, scores = layer_select.select_layers(sk, sv, tk, tv, assign, K_LAYERS)

    print(f"calibration rows n={n}, p_local={p_local}, p_full={p_full}, lambda={LAMBDA_DEFAULT}")
    print("selected source layers S_l:", [list(map(int, r)) for r in layers])
    print("planted source support:    ", [list(s) for s in plant.support])

    weights, alphas, tau = pipeline.attn_repair_weights(
        train, sample_pos, boundaries, TGT_CFG.q_per_kv, p_local)
    print(f"attn-repair: tau={tau}, alpha* in "
          f"[{float(jnp.min(alphas[0])):.3f}, {float(jnp.max(alphas[0])):.3f}] for K, "
          f"[{float(jnp.min(alphas[1])):.3f}, {float(jnp.max(alphas[1])):.3f}] for V")

    arms = {
        "full-head mapping": dict(kind="full", weights=None, chunk=None),
        "head-local only": dict(kind="local", weights=None, chunk=CHUNK),
        "cachebridge": dict(kind="local", weights=weights, chunk=CHUNK),
    }
    print(f"\n{'arm':<20}{'coeffs':>10}{'K R2':>9}{'V R2':>9}{'attn drift':>12}")
    for name, cfg in arms.items():
        m = pipeline.fit_mapper(sk, sv, tk, tv, layers, assign, lam=LAMBDA_DEFAULT, **cfg)
        pk, pv = pipeline.apply_mapper(m, hsk, hsv)
        drift = evaluate.attention_drift(held, pk, pv, sample_pos, boundaries,
                                         TGT_CFG.q_per_kv, dt)
        print(f"{name:<20}{m.n_coefficients:>10}{evaluate.r2(pk, htk):>9.4f}"
              f"{evaluate.r2(pv, htv):>9.4f}{drift:>12.4f}")

    c_full = support.coefficient_count(TGT_CFG.n_layers, ht, K_LAYERS, hs, ds, dt)
    c_local = support.coefficient_count(TGT_CFG.n_layers, ht, K_LAYERS, 1, ds, dt)
    print(f"\nC_full/C_local = {c_full // c_local} (eqs 12-13, equals Hs={hs})")
    streamed = pipeline.fit_mapper(sk, sv, tk, tv, layers, assign, chunk=CHUNK)
    one_shot = pipeline.fit_mapper(sk, sv, tk, tv, layers, assign, chunk=None)
    gap = float(jnp.max(jnp.abs(streamed.w_k - one_shot.w_k)))
    print(f"fused-fit chunk b={CHUNK}, scratch "
          f"{scratch_elements(ht, CHUNK, p_local, dt)} elements per chunk (eq 22); "
          f"max |W_streamed - W_oneshot| = {gap:.2e}")

    print("\ncyclic head-assignment ablation (figure 5)")
    print(f"{'shift':>6}{'K R2':>9}{'V R2':>9}{'attn drift':>12}")
    for shift in range(ht):
        a = support.cyclic_assignment(ht, shift)
        m = pipeline.fit_mapper(sk, sv, tk, tv, layers, a, kind="local", chunk=CHUNK)
        pk, pv = pipeline.apply_mapper(m, hsk, hsv)
        drift = evaluate.attention_drift(held, pk, pv, sample_pos, boundaries,
                                         TGT_CFG.q_per_kv, dt)
        print(f"{shift:>6}{evaluate.r2(pk, htk):>9.4f}{evaluate.r2(pv, htv):>9.4f}"
              f"{drift:>12.4f}")


if __name__ == "__main__":
    main()
