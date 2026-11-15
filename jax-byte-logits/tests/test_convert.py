"""Figure 2 numbers and Appendix F.2 qualitative cases."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_byte_logits import EOT, byte_probabilities, from_logits, token_byte_tables

FIGURE2_TOKENS = [
    "isu",
    "isk",
    "is",
    "impl",
    "imap",
    "ino",
    "mit",
    "max",
    "new",
]
FIGURE2_PROBS = np.asarray(
    [0.4, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1], dtype=np.float64
)


def np_byte_probabilities(probs, token_to_bytes, lengths, gold, use_eot=False):
    """Numpy reference of Appendix F.1."""
    n_bytes = 257 if use_eot else 256
    live = np.ones(len(probs), dtype=bool)
    dist = []
    gold = np.asarray(gold)
    token_to_bytes = np.asarray(token_to_bytes)
    lengths = np.asarray(lengths)
    probs = np.asarray(probs)
    for k in range(len(gold)):
        active = live & (lengths > k)
        bytes_at_k = token_to_bytes[:, k]
        accum = np.bincount(
            bytes_at_k[active], probs[active], minlength=n_bytes
        )
        total = accum.sum()
        dist.append(accum / total if total > 0 else accum)
        live &= (bytes_at_k == gold[k]) & (lengths > k + 1)
    return dist


def _ord(*chars):
    return tuple(ord(c) for c in chars)


def test_figure2_marginalize_it():
    table, lengths = token_byte_tables(FIGURE2_TOKENS, use_eot=False)
    dist = byte_probabilities(
        FIGURE2_PROBS, table, lengths, "isu", use_eot=False
    )
    dist = np.asarray(dist)
    i, m, n, s, u, k = _ord("i", "m", "n", "s", "u", "k")
    np.testing.assert_allclose(dist[0, i], 0.8, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[0, m], 0.1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[0, n], 0.1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[1, s], 0.75, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[1, m], 0.125, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[1, n], 0.125, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[2, u], 0.8, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[2, k], 0.2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[0].sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(dist[1].sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(dist[2].sum(), 1.0, atol=1e-6)


def test_figure2_end_of_token():
    table, lengths = token_byte_tables(FIGURE2_TOKENS, use_eot=True)
    gold = list(b"isu") + [EOT]
    dist = byte_probabilities(
        FIGURE2_PROBS, table, lengths, gold, use_eot=True
    )
    dist = np.asarray(dist)
    i, m, n, s, u, k = _ord("i", "m", "n", "s", "u", "k")
    np.testing.assert_allclose(dist[0, i], 0.8, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[0, m], 0.1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[0, n], 0.1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[1, s], 0.75, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[1, m], 0.125, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[1, n], 0.125, rtol=1e-6, atol=1e-6)
    # Paper writes 0.5/0.75 after B1 renormalization; same as 0.4/0.6 on raw mass.
    np.testing.assert_allclose(dist[2, u], 0.4 / 0.6, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[2, k], 0.1 / 0.6, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[2, EOT], 0.1 / 0.6, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dist[3, EOT], 1.0, rtol=1e-6, atol=1e-6)


def test_jax_matches_numpy_f1():
    table, lengths = token_byte_tables(FIGURE2_TOKENS, use_eot=False)
    gold = list(b"isu")
    jax_dist = np.asarray(
        byte_probabilities(
            FIGURE2_PROBS, table, lengths, gold, use_eot=False
        )
    )
    np_dist = np.stack(
        np_byte_probabilities(
            FIGURE2_PROBS, np.asarray(table), np.asarray(lengths), gold, False
        )
    )
    np.testing.assert_allclose(jax_dist, np_dist, rtol=1e-6, atol=1e-6)

    table_e, lengths_e = token_byte_tables(FIGURE2_TOKENS, use_eot=True)
    gold_e = gold + [EOT]
    jax_e = np.asarray(
        byte_probabilities(
            FIGURE2_PROBS, table_e, lengths_e, gold_e, use_eot=True
        )
    )
    np_e = np.stack(
        np_byte_probabilities(
            FIGURE2_PROBS,
            np.asarray(table_e),
            np.asarray(lengths_e),
            gold_e,
            True,
        )
    )
    np.testing.assert_allclose(jax_e, np_e, rtol=1e-6, atol=1e-6)


def test_from_logits_softmax_then_convert():
    logits = np.log(FIGURE2_PROBS)
    table, lengths = token_byte_tables(FIGURE2_TOKENS, use_eot=False)
    from_l = np.asarray(
        from_logits(logits, table, lengths, "isu", use_eot=False)
    )
    from_p = np.asarray(
        byte_probabilities(
            FIGURE2_PROBS, table, lengths, "isu", use_eot=False
        )
    )
    np.testing.assert_allclose(from_l, from_p, rtol=1e-5, atol=1e-6)


def test_f2_case_i_first_byte_exact_for_both():
    table, lengths = token_byte_tables(FIGURE2_TOKENS, use_eot=False)
    table_e, lengths_e = token_byte_tables(FIGURE2_TOKENS, use_eot=True)
    marg = np.asarray(
        byte_probabilities(
            FIGURE2_PROBS, table, lengths, "isu", use_eot=False
        )
    )
    eot = np.asarray(
        byte_probabilities(
            FIGURE2_PROBS,
            table_e,
            lengths_e,
            list(b"isu") + [EOT],
            use_eot=True,
        )
    )
    first_bytes = np.array([ord(t[0]) for t in FIGURE2_TOKENS])
    true_b1 = np.bincount(first_bytes, FIGURE2_PROBS, minlength=256)
    true_b1 = true_b1 / true_b1.sum()
    np.testing.assert_allclose(marg[0], true_b1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(eot[0, :256], true_b1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(eot[0, EOT], 0.0, atol=1e-6)


def test_f2_case_ii_equal_length_paths_match():
    tokens = ["isu", "isk", "ism", "ist"]
    probs = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    table, lengths = token_byte_tables(tokens, use_eot=False)
    table_e, lengths_e = token_byte_tables(tokens, use_eot=True)
    marg = np.asarray(
        byte_probabilities(probs, table, lengths, "isu", use_eot=False)
    )
    eot = np.asarray(
        byte_probabilities(
            probs, table_e, lengths_e, list(b"isu") + [EOT], use_eot=True
        )
    )
    u, k, m, t = _ord("u", "k", "m", "t")
    np.testing.assert_allclose(marg[2, u], 0.4, atol=1e-6)
    np.testing.assert_allclose(marg[2, k], 0.3, atol=1e-6)
    np.testing.assert_allclose(marg[2, m], 0.2, atol=1e-6)
    np.testing.assert_allclose(marg[2, t], 0.1, atol=1e-6)
    np.testing.assert_allclose(eot[2, :256], marg[2], atol=1e-6)
    np.testing.assert_allclose(eot[2, EOT], 0.0, atol=1e-6)
    np.testing.assert_allclose(eot[3, EOT], 1.0, atol=1e-6)


def test_f2_case_iii_eot_absorbs_shorter_paths():
    tokens = ["isu", "isk", "is"]
    probs = np.asarray([0.5, 0.2, 0.3], dtype=np.float64)
    table, lengths = token_byte_tables(tokens, use_eot=False)
    table_e, lengths_e = token_byte_tables(tokens, use_eot=True)
    marg = np.asarray(
        byte_probabilities(probs, table, lengths, "isu", use_eot=False)
    )
    eot = np.asarray(
        byte_probabilities(
            probs, table_e, lengths_e, list(b"isu") + [EOT], use_eot=True
        )
    )
    u, k = _ord("u", "k")
    # Marginalize-It drops `is` before B3; remaining isu+isk.
    np.testing.assert_allclose(marg[2, u], 0.5 / 0.7, atol=1e-6)
    np.testing.assert_allclose(marg[2, k], 0.2 / 0.7, atol=1e-6)
    assert marg.shape[1] == 256
    # End-Of-Token keeps `is` as EOT mass at B3.
    np.testing.assert_allclose(eot[2, u], 0.5 / 1.0, atol=1e-6)
    np.testing.assert_allclose(eot[2, k], 0.2 / 1.0, atol=1e-6)
    np.testing.assert_allclose(eot[2, EOT], 0.3 / 1.0, atol=1e-6)
    assert eot[2, EOT] > 0.0


def test_f2_case_iv_eot_absorbs_longer_paths():
    tokens = ["is", "isu", "isk"]
    probs = np.asarray([0.4, 0.4, 0.2], dtype=np.float64)
    table, lengths = token_byte_tables(tokens, use_eot=False)
    table_e, lengths_e = token_byte_tables(tokens, use_eot=True)
    marg = np.asarray(
        byte_probabilities(probs, table, lengths, "is", use_eot=False)
    )
    eot = np.asarray(
        byte_probabilities(
            probs, table_e, lengths_e, list(b"is") + [EOT], use_eot=True
        )
    )
    u, k, s = _ord("u", "k", "s")
    # Marginalize-It gold is only two bytes, so longer tokens never get a B3 slot.
    assert marg.shape[0] == 2
    np.testing.assert_allclose(marg[1, s], 1.0, atol=1e-6)
    # End-Of-Token B3: `is` -> EOT, `isu` -> u, `isk` -> k.
    np.testing.assert_allclose(eot[2, EOT], 0.4, atol=1e-6)
    np.testing.assert_allclose(eot[2, u], 0.4, atol=1e-6)
    np.testing.assert_allclose(eot[2, k], 0.2, atol=1e-6)
    assert eot[2, EOT] < 1.0


def test_from_logits_jittable():
    table, lengths = token_byte_tables(FIGURE2_TOKENS, use_eot=False)
    logits = jnp.log(jnp.asarray(FIGURE2_PROBS))
    gold = jnp.asarray(list(b"isu"), dtype=jnp.int32)
    fn = jax.jit(
        lambda lg: from_logits(lg, table, lengths, gold, use_eot=False)
    )
    out = np.asarray(fn(logits))
    assert out.shape == (3, 256)
