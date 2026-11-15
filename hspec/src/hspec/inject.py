"""Last-hidden injection and a diagonal selective scan."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def s_init(h_last, proj_r, proj_ds):
    """proj_ds @ proj_r @ concat(selected last-position hiddens).

    One shared projection of the concatenation, not a projection per layer.
    ``ndim >= 3`` treats axis -2 as time and keeps only the last step
    (shape ``(..., seq, d)``). ``ndim == 2`` is already last-token
    ``(n_selected, d)``. A list or tuple is one array per selected layer.
    """
    flat = _concat_selected_last(h_last)
    return proj_ds @ (proj_r @ flat)


def _concat_selected_last(h_last):
    if isinstance(h_last, (list, tuple)):
        parts = [_last_vec(jnp.asarray(h)) for h in h_last]
        return jnp.concatenate(parts, axis=0)
    return jnp.ravel(_last_block(jnp.asarray(h_last)))


def _last_block(h):
    if h.ndim >= 3:
        return h[..., -1, :]
    return h


def _last_vec(h):
    if h.ndim >= 2:
        return jnp.ravel(h[-1])
    return jnp.ravel(h)


def _bars(x, dt, A, W_B, x_proj):
    """A_bar = exp(-softplus(dt) * softplus(A)); u = B_bar * x_proj.

    B_bar = softplus(dt) * W_B(x), W_B a matrix ``(d_state, d_in)``.
    Leading axis is time.
    """
    x = jnp.asarray(x)
    dt = jnp.asarray(dt)
    A = jnp.asarray(A)
    x_proj = jnp.asarray(x_proj)
    W_B = jnp.asarray(W_B)
    sp_dt = jax.nn.softplus(dt)
    if sp_dt.ndim == x.ndim - 1:
        sp_dt = jnp.expand_dims(sp_dt, -1)
    A_bar = jnp.exp(-sp_dt * jax.nn.softplus(A))
    B_bar = sp_dt * (x @ jnp.swapaxes(W_B, -1, -2))
    u = B_bar * x_proj
    return jnp.broadcast_to(A_bar, u.shape), u


def _combine(left, right):
    """Compose S = a * S_prev + u. Left segment is earlier in time."""
    a_l, u_l = left
    a_r, u_r = right
    return (a_r * a_l, a_r * u_l + u_r)


def scan_sequential(x, dt, A, W_B, x_proj, S_init):
    """S_i = A_bar_i * S_{i-1} + B_bar_i * x_proj_i, with S_0 = S_init."""
    A_bar, u = _bars(x, dt, A, W_B, x_proj)
    S_init = jnp.asarray(S_init)

    def step(s, inp):
        a_i, u_i = inp
        s_new = a_i * s + u_i
        return s_new, s_new

    _, states = jax.lax.scan(step, S_init, (A_bar, u))
    return states


def scan_parallel(x, dt, A, W_B, x_proj, S_init):
    """Same recurrence as ``scan_sequential``, via ``lax.associative_scan``."""
    A_bar, u = _bars(x, dt, A, W_B, x_proj)
    S_init = jnp.asarray(S_init)
    a_pref, u_pref = jax.lax.associative_scan(_combine, (A_bar, u))
    return a_pref * S_init + u_pref


def drafter_kv_bytes(prefix_len):
    """Drafter keeps no prefix KV cache."""
    del prefix_len
    return 0


def injection_bytes(n_layers, d):
    """Injected last-position payload in bytes. Independent of prefix length."""
    return int(n_layers) * int(d) * 2
