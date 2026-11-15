"""Scaled-down section 5.1 drift process. Asserts recall, not a coin-flip reroute rate."""

import jax

from routerelay.synthetic import run_stack, toy_chunk_stack


def test_low_drift_recall_and_fewer_score_pairs():
    key = jax.random.PRNGKey(0)
    n_q, n_k, d, n_layers = 64, 64, 64, 6
    k, m, r = 12, 24, 12
    q_c, k_c = toy_chunk_stack(
        key,
        n_q=n_q,
        n_k=n_k,
        d=d,
        n_pref=k,
        drift_std=0.01,
        n_layers=n_layers,
    )
    out = run_stack(
        q_c,
        k_c,
        k=k,
        m=m,
        r=r,
        gamma=0.0,
        rng_key=jax.random.PRNGKey(1),
    )
    assert float(out.recall) >= 0.999
    assert int(out.n_score_pairs) < int(out.n_full_pairs)
