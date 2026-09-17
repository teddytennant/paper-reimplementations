"""SOX baseline (Wang and Yang, 2022): moving-average inner + gradient trackers."""

from __future__ import annotations

from jax_msvr.estimator import block_chain_mean, sox_update
from jax_msvr.msvrm import apply_w_update, _sample_probe


def sox_step(
    state,
    oracle,
    key,
    *,
    B1,
    alpha,
    beta,
    eta,
    adaptive=False,
    eps=1e-8,
):
    """One SOX step: eq. 3 inner tracker and a v1-style moving-average z.

    SOX is MSVR with gamma = 0 on the inner estimator.
    """
    idx, g_now, _g_prev, jac_now, _jac_prev = _sample_probe(
        oracle, state.w, state.w_prev, key, B1
    )
    u_old = state.u
    u_new = sox_update(u_old, g_now, idx, beta)
    chain = block_chain_mean(oracle.grad_f(u_old)[idx], jac_now)
    z_new = (1.0 - alpha) * state.z + alpha * chain
    w_new = apply_w_update(state.w, z_new, eta, adaptive=adaptive, eps=eps)
    return state._replace(
        w=w_new,
        u=u_new,
        z=z_new,
        u_prev=u_old,
        w_prev=state.w,
        t=state.t + 1,
    )
