"""FlashBoB checks: identities, autodiff, tiling, Lemma E.2, affine fusion, HBM ledger."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from flashbob import (
    causal_mask,
    flashbob_autodiff,
    flashbob_dense,
    flashbob_tiled,
    forward_attention,
)

ATOL = 1e-8
RTOL = 1e-6


def _inputs(seed, n, d, scale=1.0):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 8)
    mats = [
        scale * jax.random.normal(k, (n, d), dtype=jnp.float64) for k in keys[:7]
    ]
    Q, K, V, dO, UQ, UK, UV = mats
    bool_mask = jax.random.bernoulli(keys[7], 0.55, (n, n))
    bool_mask = bool_mask.at[jnp.arange(n), 0].set(True)
    return Q, K, V, dO, UQ, UK, UV, bool_mask


def _assert_close(a, b, what):
    diff = float(jnp.max(jnp.abs(a - b)))
    assert jnp.allclose(a, b, rtol=RTOL, atol=ATOL), f"{what}: max abs {diff}"


def _modes(bool_mask):
    return (
        ("unmasked", None, False),
        ("causal", None, True),
        ("bool", bool_mask, False),
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("causal", [False, True])
def test_D_equals_rowsum_and_dot(seed, causal):
    n, d = 12, 5
    Q, K, V, dO, _, _, _, bool_mask = _inputs(seed, n, d)
    masks = [(None, causal)]
    if not causal:
        masks.append((bool_mask, False))
    for mask, use_causal in masks:
        O, P, _ = forward_attention(Q, K, V, mask=mask, causal=use_causal)
        dP = dO @ V.T
        D_dot = jnp.sum(dO * O, axis=-1)
        D_row = jnp.sum(P * dP, axis=-1)
        _assert_close(D_dot, D_row, "D dot vs rowsum(P * dP)")
        dense = flashbob_dense(Q, K, V, dO, Q, K, V, mask=mask, causal=use_causal)
        _assert_close(dense.D, D_dot, "saved D vs dO · O")
        _assert_close(dense.D, D_row, "saved D vs rowsum(P * dP)")


@pytest.mark.parametrize("seed", [0, 1, 2, 4])
@pytest.mark.parametrize("mode", ["unmasked", "causal"])
def test_dense_matches_autodiff(seed, mode):
    n, d = 9, 4
    Q, K, V, dO, UQ, UK, UV, bool_mask = _inputs(seed, n, d)
    causal = mode == "causal"
    mask = None
    dense = flashbob_dense(Q, K, V, dO, UQ, UK, UV, mask=mask, causal=causal)
    grad = flashbob_autodiff(Q, K, V, dO, UQ, UK, UV, mask=mask, causal=causal)
    for name, got, ref in zip(("dQ", "dK", "dV", "ddO"), dense[:4], grad):
        _assert_close(got, ref, f"{mode} dense {name}")


@pytest.mark.parametrize("seed", [1, 5])
def test_dense_matches_autodiff_boolean_mask(seed):
    n, d = 8, 3
    Q, K, V, dO, UQ, UK, UV, bool_mask = _inputs(seed, n, d)
    dense = flashbob_dense(Q, K, V, dO, UQ, UK, UV, mask=bool_mask)
    grad = flashbob_autodiff(Q, K, V, dO, UQ, UK, UV, mask=bool_mask)
    for name, got, ref in zip(("dQ", "dK", "dV", "ddO"), dense[:4], grad):
        _assert_close(got, ref, f"bool dense {name}")


@pytest.mark.parametrize(
    "n,d,br,bc",
    [
        (8, 4, 4, 4),
        (8, 4, 3, 5),
        (7, 3, 2, 4),
        (16, 8, 5, 3),
        (32, 16, 8, 7),
        (5, 1, 2, 2),
        (9, 4, 4, 3),
        (11, 6, 11, 1),
    ],
)
@pytest.mark.parametrize("mode", ["unmasked", "causal"])
def test_tiled_matches_dense(n, d, br, bc, mode):
    Q, K, V, dO, UQ, UK, UV, bool_mask = _inputs(10 + n + d, n, d)
    causal = mode == "causal"
    dense = flashbob_dense(Q, K, V, dO, UQ, UK, UV, causal=causal)
    tiled = flashbob_tiled(Q, K, V, dO, UQ, UK, UV, br, bc, causal=causal)
    for name in ("dQ", "dK", "dV", "ddO", "alpha", "E"):
        _assert_close(getattr(tiled, name), getattr(dense, name), f"{mode} {(n, d, br, bc)} {name}")
    grad = flashbob_autodiff(Q, K, V, dO, UQ, UK, UV, causal=causal)
    for name, got, ref in zip(("dQ", "dK", "dV", "ddO"), tiled[:4], grad):
        _assert_close(got, ref, f"{mode} tiled vs autodiff {name}")


def test_tiled_matches_dense_boolean_mask():
    n, d, br, bc = 13, 5, 4, 6
    Q, K, V, dO, UQ, UK, UV, bool_mask = _inputs(21, n, d)
    dense = flashbob_dense(Q, K, V, dO, UQ, UK, UV, mask=bool_mask)
    tiled = flashbob_tiled(Q, K, V, dO, UQ, UK, UV, br, bc, mask=bool_mask)
    for name in ("dQ", "dK", "dV", "ddO"):
        _assert_close(getattr(tiled, name), getattr(dense, name), f"bool {name}")
    # Explicit causal mask matches the causal flag.
    explicit = causal_mask(n)
    a = flashbob_tiled(Q, K, V, dO, UQ, UK, UV, br, bc, mask=explicit)
    b = flashbob_tiled(Q, K, V, dO, UQ, UK, UV, br, bc, causal=True)
    _assert_close(a.dQ, b.dQ, "causal flag vs mask dQ")
    _assert_close(a.ddO, b.ddO, "causal flag vs mask ddO")
    both = flashbob_tiled(Q, K, V, dO, UQ, UK, UV, br, bc, mask=bool_mask, causal=True)
    ref = flashbob_dense(Q, K, V, dO, UQ, UK, UV, mask=bool_mask, causal=True)
    _assert_close(both.ddO, ref.ddO, "causal and bool together")
    _assert_close(both.dK, ref.dK, "causal and bool dK")


@pytest.mark.parametrize("seed", [0, 2, 7])
@pytest.mark.parametrize("n,d,br,bc", [(6, 3, 2, 2), (10, 4, 3, 4), (8, 2, 8, 1)])
def test_lemma_e2(seed, n, d, br, bc):
    Q, K, _, _, _, _, W, _ = _inputs(seed, n, d)
    zeros = jnp.zeros((n, d), dtype=jnp.float64)
    tau = 1.0 / jnp.sqrt(jnp.asarray(d, dtype=jnp.float64))
    expected = jax.nn.softmax(tau * (Q @ K.T), axis=-1) @ W
    dense = flashbob_dense(Q, K, zeros, zeros, zeros, zeros, W)
    tiled = flashbob_tiled(Q, K, zeros, zeros, zeros, zeros, W, br, bc)
    _assert_close(dense.ddO, expected, "lemma E.2 dense ddO")
    _assert_close(tiled.ddO, expected, "lemma E.2 tiled ddO")
    for name in ("dQ", "dK", "dV"):
        _assert_close(getattr(dense, name), zeros, f"lemma E.2 dense {name}")
        _assert_close(getattr(tiled, name), zeros, f"lemma E.2 tiled {name}")


@pytest.mark.parametrize("n,d,br,bc", [(8, 4, 3, 5), (12, 6, 4, 4), (7, 2, 2, 3)])
@pytest.mark.parametrize("causal", [False, True])
def test_affine_fusion_and_pass1_accumulators(n, d, br, bc, causal):
    Q, K, V, dO, UQ, UK, UV, _ = _inputs(3, n, d)
    dense = flashbob_dense(Q, K, V, dO, UQ, UK, UV, causal=causal)
    tiled = flashbob_tiled(Q, K, V, dO, UQ, UK, UV, br, bc, causal=causal)
    tau = 1.0 / jnp.sqrt(jnp.asarray(d, dtype=jnp.float64))
    for label, res in (("dense", dense), ("tiled", tiled)):
        _assert_close(res.E, res.E_base - res.alpha * res.D, f"{label} E fusion")
        _assert_close(
            res.ddO,
            res.ddO_base - res.alpha[:, None] * res.O,
            f"{label} ddO fusion",
        )
        dQ_omega = tau * (
            res.Omega - res.alpha[:, None] * res.R - res.E[:, None] * res.B
        )
        _assert_close(res.dQ, dQ_omega, f"{label} Omega form")
    for name in ("alpha", "E", "E_base", "Omega", "B", "R", "ddO_base", "dQ", "ddO"):
        _assert_close(
            getattr(tiled, name),
            getattr(dense, name),
            f"pass-1 {name}",
        )


def _theorem_traffic(n, d, br, bc):
    """Theorem 4.3 with Br = Br_pass1 and Bc = Bc_pass2."""
    return 12 * n * d + 4 * n + (4 * d * n * n) // br + (3 * d + 4) * n * n // bc


@pytest.mark.parametrize(
    "n,d,br,bc",
    [
        (8, 4, 4, 4),
        (16, 8, 8, 4),
        (16, 8, 4, 8),
        (12, 3, 6, 4),
        (9, 1, 3, 3),
        (32, 16, 8, 16),
        (24, 6, 8, 6),
        (6, 2, 2, 3),
    ],
)
@pytest.mark.parametrize("mode", ["unmasked", "causal"])
def test_hbm_traffic_matches_theorem(n, d, br, bc, mode):
    assert n % br == 0 and n % bc == 0
    Q, K, V, dO, UQ, UK, UV, bool_mask = _inputs(0, n, d)
    causal = mode == "causal"
    tiled = flashbob_tiled(Q, K, V, dO, UQ, UK, UV, br, bc, causal=causal)
    expected = _theorem_traffic(n, d, br, bc)
    assert isinstance(tiled.traffic, int)
    assert tiled.traffic == expected, f"traffic {tiled.traffic} != {expected}"
    if mode == "unmasked":
        masked = flashbob_tiled(Q, K, V, dO, UQ, UK, UV, br, bc, mask=bool_mask)
        assert masked.traffic == expected
