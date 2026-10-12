"""Associative affine scans and block-triangular state construction."""

import jax
import jax.numpy as jnp

from spvi.flow import flow


def affine_star(later, earlier):
    """(a2, b2) star (a1, b1) = (a2 * a1, a2 * b1 + b2).

    `earlier` is applied first. Identity element is (1, 0).
    """
    a2, b2 = later
    a1, b1 = earlier
    return a2 * a1, a2 * b1 + b2


def _scan_combine(earlier, later):
    return affine_star(later, earlier)


def affine_prefix_scan(a, b, axis=0):
    """Inclusive prefix scan of affine maps under star.

    Returns (a_pref, b_pref) with the same shapes as (a, b). Position i is the
    composition of maps 0..i, so z_i = a_pref_i * z0 + b_pref_i.
    """
    return jax.lax.associative_scan(_scan_combine, (a, b), axis=axis)


def _as_batch(z0, event_times, marks, gates, pad_mask):
    unbatched = z0.ndim == 1
    if unbatched:
        z0 = z0[None]
        event_times = event_times[None]
        if marks is not None and marks.ndim >= 1:
            marks = marks[None]
        if gates is not None:
            gates = gates[None]
        if pad_mask is not None:
            pad_mask = pad_mask[None]
    return unbatched, z0, event_times, marks, gates, pad_mask


def block_scan(z0, event_times, marks, gates, gamma, coeff_fn, block_sizes, pad_mask=None):
    """Algorithm 2. Post-event states with Z0 prepended.

    z0: (d,) or (B, d)
    event_times: (L,) or (B, L) absolute times. r0 = 0.
    marks: (L, m) or (B, L, m)
    gates: (L,) or (B, L) in {0, 1} (straight-through values allowed)
    gamma: (d,) positive decay rates
    coeff_fn(j, marks, lower_pre, lower_post) -> (R, B), each (B, L, d_j).
        Block j must be a function of the mark and of lower blocks only.
    block_sizes: contiguous split of the state, summing to d
    pad_mask: True on padding positions, which use the identity map (1, 0)
        and therefore do not change any real prefix. Padding should be a tail
        so inter-event gaps of real events stay equal to consecutive times.

    If L = 0, returns Z0 with a length-1 event axis (the event-free path).
    """
    unbatched, z0, event_times, marks, gates, pad_mask = _as_batch(
        z0, event_times, marks, gates, pad_mask
    )
    batch, length = event_times.shape
    if length == 0:
        states = z0[:, None, :]
        return states[0] if unbatched else states

    if pad_mask is None:
        pad_mask = jnp.zeros((batch, length), dtype=bool)
    gamma = jnp.asarray(gamma)

    prev_t = jnp.concatenate(
        [jnp.zeros((batch, 1), dtype=event_times.dtype), event_times[:, :-1]],
        axis=1,
    )
    delta = event_times - prev_t
    decay = jnp.exp(-gamma * delta[..., None])

    posts = []
    offset = 0
    for j, width in enumerate(block_sizes):
        if j == 0:
            lower_pre = jnp.zeros((batch, length, 0), dtype=z0.dtype)
            lower_post = jnp.zeros((batch, length, 0), dtype=z0.dtype)
        else:
            lower_post = jnp.concatenate(posts, axis=-1)
            prev_post = jnp.concatenate(
                [z0[:, None, :offset], lower_post[:, :-1, :]],
                axis=1,
            )
            lower_pre = decay[..., :offset] * prev_post
        retention, shift = coeff_fn(j, marks, lower_pre, lower_post)
        gate = gates[..., None]
        decay_j = decay[..., offset : offset + width]
        scale = (1.0 + gate * (retention - 1.0)) * decay_j
        bias = gate * shift
        scale = jnp.where(pad_mask[..., None], 1.0, scale)
        bias = jnp.where(pad_mask[..., None], 0.0, bias)
        scale_pref, bias_pref = affine_prefix_scan(scale, bias, axis=1)
        z_j = scale_pref * z0[:, None, offset : offset + width] + bias_pref
        posts.append(z_j)
        offset += width

    post = jnp.concatenate(posts, axis=-1)
    states = jnp.concatenate([z0[:, None, :], post], axis=1)
    return states[0] if unbatched else states


def apply_jump(z_pre, mark, gate, coeff_fn, block_sizes):
    """One block-triangular jump on an already-flowed pre-event state.

    z_pre: (B, d) or (d,)
    mark: (B, m) or (m,)
    gate: (B,) or scalar
    """
    unbatched = z_pre.ndim == 1
    if unbatched:
        z_pre = z_pre[None]
        mark = mark[None]
        gate = jnp.asarray(gate)[None]
    else:
        gate = jnp.asarray(gate)
    parts = []
    offset = 0
    lower_pre_parts = []
    lower_post_parts = []
    for j, width in enumerate(block_sizes):
        if j == 0:
            lower_pre = jnp.zeros((z_pre.shape[0], 1, 0), dtype=z_pre.dtype)
            lower_post = jnp.zeros((z_pre.shape[0], 1, 0), dtype=z_pre.dtype)
        else:
            lower_pre = jnp.concatenate(lower_pre_parts, axis=-1)[:, None, :]
            lower_post = jnp.concatenate(lower_post_parts, axis=-1)[:, None, :]
        retention, shift = coeff_fn(j, mark[:, None, :], lower_pre, lower_post)
        retention = retention[:, 0, :]
        shift = shift[:, 0, :]
        z_j = z_pre[:, offset : offset + width]
        z_post_j = (1.0 + gate[:, None] * (retention - 1.0)) * z_j + gate[:, None] * shift
        parts.append(z_post_j)
        lower_pre_parts.append(z_j)
        lower_post_parts.append(z_post_j)
        offset += width
    out = jnp.concatenate(parts, axis=-1)
    return out[0] if unbatched else out


def query_states(states, event_times, query_times, gamma, valid=None, pre_event_on_tie=False):
    """State at time t: i = max{n : r_n <= t} (r0 = 0), then Phi_{t - r_i}(Z_{r_i}).

    states: (L+1, d) or (B, L+1, d), index 0 is Z0, index n is the post-event
    state at r_n.
    event_times: (L,) or (B, L)
    query_times: (Q,) or (B, Q)
    valid: mask of real (non-padding) events. Padding times are ignored.
    pre_event_on_tie: if t equals a real event time, return the pre-event state
    (left limit). Observation queries leave this false.

    L = 0 returns Phi_t(Z0).
    """
    states = jnp.asarray(states)
    event_times = jnp.asarray(event_times)
    query_times = jnp.asarray(query_times)
    unbatched = states.ndim == 2
    if unbatched:
        states = states[None]
        event_times = event_times[None]
        if query_times.ndim == 1:
            query_times = jnp.broadcast_to(query_times, (1, query_times.shape[0]))
        if valid is not None:
            valid = valid[None]
    elif query_times.ndim == 1:
        query_times = jnp.broadcast_to(query_times, (states.shape[0], query_times.shape[0]))

    batch, length_plus, _dim = states.shape
    length = event_times.shape[1]
    gamma = jnp.asarray(gamma)
    del length_plus

    if length == 0:
        out = flow(states[:, 0, :][:, None, :], query_times, gamma)
        return out[0] if unbatched else out

    if valid is None:
        valid = jnp.ones((batch, length), dtype=bool)

    # i_star = number of valid events at or before t, in 0..L. State index equals i_star.
    at_or_before = valid[:, None, :] & (event_times[:, None, :] <= query_times[..., None])
    count = jnp.sum(at_or_before, axis=-1).astype(jnp.int32)
    batch_idx = jnp.arange(batch)[:, None]

    idx = jnp.clip(count - 1, 0, length - 1)
    r_i = event_times[batch_idx, idx]
    r_i = jnp.where(count > 0, r_i, jnp.zeros((), dtype=event_times.dtype))

    state = states[batch_idx, count]
    dt = query_times - r_i
    out = flow(state, dt, gamma)

    if pre_event_on_tie:
        prev_state = states[batch_idx, jnp.clip(count - 1, 0, length)]
        earlier_idx = jnp.clip(count - 2, 0, length - 1)
        r_prev = event_times[batch_idx, earlier_idx]
        r_prev = jnp.where(count <= 1, jnp.zeros((), dtype=event_times.dtype), r_prev)
        pre = flow(prev_state, r_i - r_prev, gamma)
        on_tie = (count > 0) & (query_times == r_i)
        out = jnp.where(on_tie[..., None], pre, out)

    return out[0] if unbatched else out
