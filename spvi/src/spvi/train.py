"""Algorithm 1 training step, Algorithm 3 forecast, synthetic cyclic batch."""

import functools

import jax
import jax.numpy as jnp
import optax

from spvi.elbo import gaussian_loglik, kappa, kl_standard_normal, objective_j, stratified_times
from spvi.flow import flow, positive_gamma
from spvi.gates import logit, proposal_rate, straight_through_gate
from spvi.model import (
    Config,
    emission_mean,
    encode,
    initial_posterior,
    make_coeff_fn,
    observation_intervals,
    posterior_event_params,
    prior_event_params,
)
from spvi.scan import apply_jump, block_scan, query_states


def make_optimizer(learning_rate=1e-3):
    """AdamW with betas (0.9, 0.99), weight decay 1e-5, global grad clip 1."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, b1=0.9, b2=0.99, weight_decay=1e-5),
    )


def _sample_z0(mu, var, key):
    eps = jax.random.normal(key, mu.shape)
    return mu + jnp.sqrt(var) * eps


def _sample_thinned_candidates(key, lam, mu, var, t_right, horizon, config):
    """Poisson number of uniform candidates, capped at l_max, with straight-through gates.

    Padding is a tail of identity maps. Returns times, marks, gates, pad_mask.
    """
    batch, n_intervals = lam.shape
    length = config.l_max
    mark_dim = mu.shape[-1]
    k_count, k_time, k_eps, k_zeta = jax.random.split(key, 4)
    lam_bar = proposal_rate(lam, config.rho)
    count = jax.random.poisson(k_count, lam_bar * horizon)
    count = jnp.minimum(count, length).astype(jnp.int32)

    raw = jax.random.uniform(k_time, (batch, length)) * horizon[:, None]
    slot = jnp.arange(length)
    valid = slot[None, :] < count[:, None]
    pad_times = horizon[:, None] + 1.0 + slot[None, :].astype(horizon.dtype)
    times = jnp.where(valid, raw, pad_times)
    order = jnp.argsort(times, axis=1)
    times = jnp.take_along_axis(times, order, axis=1)
    valid = jnp.take_along_axis(valid, order, axis=1)

    def interval_index(right, query):
        return jnp.clip(jnp.searchsorted(right, query, side="left"), 0, n_intervals - 1)

    which = jax.vmap(interval_index)(t_right, times)
    lam_i = jnp.take_along_axis(lam, which, axis=1)
    mu_i = jnp.take_along_axis(mu, which[..., None], axis=1)
    var_i = jnp.take_along_axis(var, which[..., None], axis=1)
    accept_prob = lam_i / lam_bar[:, None]
    eps = jax.random.normal(k_eps, (batch, length, mark_dim))
    marks = mu_i + jnp.sqrt(var_i) * eps
    zeta = jax.random.logistic(k_zeta, (batch, length))
    gates = straight_through_gate(logit(accept_prob), zeta, config.tau_st)
    gates = jnp.where(valid, gates, 0.0)
    return times, marks, gates, ~valid


def estimate_j(params, batch, key, config=None):
    """One Monte Carlo trajectory of J per batch element (Algorithm 1, steps 1-8)."""
    if config is None:
        config = Config()
    times = batch["times"]
    y = batch["y"]
    k_enc, k_z, k_ev, k_quad = jax.random.split(key, 4)
    del k_enc  # recognition is deterministic
    embeddings = encode(params, times, y)
    mu0, v0 = initial_posterior(params, embeddings)
    lam, mu, var = posterior_event_params(params, embeddings)
    t_left, t_right, delta, horizon = observation_intervals(times)

    z0 = _sample_z0(mu0, v0, k_z)
    event_times, marks, gates, pad = _sample_thinned_candidates(
        k_ev, lam, mu, var, t_right, horizon, config
    )
    gamma = positive_gamma(params["gamma_raw"])
    coeff = make_coeff_fn(params, config.rho_min)
    states = block_scan(
        z0,
        event_times,
        marks,
        gates,
        gamma,
        coeff,
        config.block_sizes,
        pad_mask=pad,
    )
    valid = ~pad
    z_obs = query_states(
        states, event_times, times, gamma, valid=valid, pre_event_on_tie=False
    )
    mean = emission_mean(z_obs, config.obs_dim)
    loglik = gaussian_loglik(y, mean, config.obs_var)

    uniforms = jax.random.uniform(k_quad, times.shape + (config.n_quad,))
    xi, weights = stratified_times(t_left, t_right, uniforms)
    batch_size, n_intervals, probes = uniforms.shape
    xi_flat = xi.reshape(batch_size, n_intervals * probes)
    z_quad = query_states(
        states,
        event_times,
        xi_flat,
        gamma,
        valid=valid,
        pre_event_on_tie=True,
    )
    z_quad = z_quad.reshape(batch_size, n_intervals, probes, -1)
    lam_theta, mu_prior, v_prior = prior_event_params(params, z_quad)
    integrand = kappa(
        lam[:, :, None],
        lam_theta,
        mu[:, :, None, :],
        var[:, :, None, :],
        mu_prior,
        v_prior,
    )
    k_dyn = jnp.sum(weights * integrand, axis=(-1, -2))
    kl0 = kl_standard_normal(mu0, v0)
    return objective_j(
        loglik,
        kl0 + k_dyn,
        lam,
        delta,
        config.w_kl,
        config.beta_sparse,
    )


def _static_config(params, obs_var, tau_st, rho, rho_min, w_kl, beta_sparse, n_quad, l_max, obs_dim, block_sizes, lambda_bar_theta):
    return Config(
        obs_dim=obs_dim,
        state_dim=int(params["gamma_raw"].shape[-1]),
        mark_dim=int(params["mark_head"]["mu_b"].shape[-1]),
        hidden=int(params["prior_rate"]["w1"].shape[-1]),
        embed_dim=int(params["rate_head"]["w"].shape[0]),
        block_sizes=block_sizes,
        obs_var=obs_var,
        tau_st=tau_st,
        rho=rho,
        rho_min=rho_min,
        l_max=l_max,
        n_quad=n_quad,
        w_kl=w_kl,
        beta_sparse=beta_sparse,
        lambda_bar_theta=lambda_bar_theta,
    )


@functools.partial(jax.jit, static_argnames=("n_quad", "l_max", "obs_dim", "block_sizes"))
def _loss_and_grad(
    params,
    times,
    y,
    key,
    obs_var,
    tau_st,
    rho,
    rho_min,
    w_kl,
    beta_sparse,
    n_quad,
    l_max,
    obs_dim,
    block_sizes,
):
    config = _static_config(
        params,
        obs_var,
        tau_st,
        rho,
        rho_min,
        w_kl,
        beta_sparse,
        n_quad,
        l_max,
        obs_dim,
        block_sizes,
        lambda_bar_theta=10.0,
    )

    def loss_fn(p):
        return -jnp.mean(estimate_j(p, {"times": times, "y": y}, key, config))

    return jax.value_and_grad(loss_fn)(params)


def train_step(params, opt_state, optimizer, batch, key, config=None):
    """Minimize -J. Returns updated params, opt state, loss, and gradients."""
    if config is None:
        config = Config()
    loss, grads = _loss_and_grad(
        params,
        batch["times"],
        batch["y"],
        key,
        config.obs_var,
        config.tau_st,
        config.rho,
        config.rho_min,
        config.w_kl,
        config.beta_sparse,
        config.n_quad,
        config.l_max,
        config.obs_dim,
        config.block_sizes,
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, grads


def _empty_events(batch, mark_dim, dtype):
    times = jnp.zeros((batch, 0), dtype=dtype)
    marks = jnp.zeros((batch, 0, mark_dim), dtype=dtype)
    gates = jnp.zeros((batch, 0), dtype=dtype)
    pad = jnp.zeros((batch, 0), dtype=bool)
    return times, marks, gates, pad


def _sample_context_events(key, lam, mu, var, t_left, t_right, config):
    """Accepted posterior events: Poisson counts, uniform times, Gaussian marks, g = 1."""
    batch, n_intervals = lam.shape
    mark_dim = mu.shape[-1]
    per = 4
    length = config.l_max
    k_count, k_time, k_eps = jax.random.split(key, 3)
    delta = t_right - t_left
    count = jax.random.poisson(k_count, lam * delta)
    count = jnp.minimum(count, per).astype(jnp.int32)
    unit = jax.random.uniform(k_time, (batch, n_intervals, per))
    times = t_left[..., None] + unit * delta[..., None]
    slot = jnp.arange(per)
    valid = slot[None, None, :] < count[..., None]
    eps = jax.random.normal(k_eps, (batch, n_intervals, per, mark_dim))
    marks = mu[..., None, :] + jnp.sqrt(var)[..., None, :] * eps
    flat_n = n_intervals * per
    times = times.reshape(batch, flat_n)
    valid = valid.reshape(batch, flat_n)
    marks = marks.reshape(batch, flat_n, mark_dim)
    horizon = t_right[:, -1]
    pad_times = horizon[:, None] + 1.0 + jnp.arange(flat_n, dtype=times.dtype)[None, :]
    times_sort = jnp.where(valid, times, pad_times)
    order = jnp.argsort(times_sort, axis=1)
    times = jnp.take_along_axis(times_sort, order, axis=1)[:, :length]
    valid = jnp.take_along_axis(valid, order, axis=1)[:, :length]
    marks = jnp.take_along_axis(marks, order[..., None], axis=1)[:, :length]
    gates = valid.astype(times.dtype)
    return times, marks, gates, ~valid


def reconstruct_context(params, times, y, key, config=None, include_events=True):
    """Sample q0 and context events, then BlockScan. Returns states, z_T, z0, gamma."""
    if config is None:
        config = Config()
    k_z, k_ev = jax.random.split(key)
    embeddings = encode(params, times, y)
    mu0, v0 = initial_posterior(params, embeddings)
    lam, mu, var = posterior_event_params(params, embeddings)
    z0 = _sample_z0(mu0, v0, k_z)
    gamma = positive_gamma(params["gamma_raw"])
    coeff = make_coeff_fn(params, config.rho_min)
    if include_events:
        t_left, t_right, _, _ = observation_intervals(times)
        event_times, marks, gates, pad = _sample_context_events(
            k_ev, lam, mu, var, t_left, t_right, config
        )
    else:
        event_times, marks, gates, pad = _empty_events(times.shape[0], config.mark_dim, times.dtype)
    states = block_scan(
        z0, event_times, marks, gates, gamma, coeff, config.block_sizes, pad_mask=pad
    )
    valid = ~pad if include_events else None
    horizon = times[:, -1]
    z_T = query_states(
        states, event_times, horizon[:, None], gamma, valid=valid, pre_event_on_tie=False
    )[:, 0, :]
    return {
        "states": states,
        "event_times": event_times,
        "valid": valid,
        "z_T": z_T,
        "z0": z0,
        "gamma": gamma,
    }


def _simulate_future(z_T, offsets, valid, key, params, config, gamma):
    """Sequential prior simulation from Z_T. offsets are times relative to the origin.

    Empty offsets (M = 0) leave the state at Z_T, so queries are the pure flow.
    """
    batch, n_cand = offsets.shape
    if n_cand == 0:
        return z_T[:, None, :], jnp.zeros((batch, 0), dtype=z_T.dtype), jnp.zeros((batch, 0), dtype=bool)

    k_accept, k_mark = jax.random.split(key)
    accept_u = jax.random.uniform(k_accept, (n_cand, batch))
    mark_eps = jax.random.normal(k_mark, (n_cand, batch, config.mark_dim))
    coeff = make_coeff_fn(params, config.rho_min)
    bar = config.lambda_bar_theta

    def step(carry, xs):
        z, s = carry
        c, is_valid, u_acc, eps = xs
        dt = c - s
        z_pre = flow(z, dt, gamma)
        lam, mu, var = prior_event_params(params, z_pre)
        accept = (u_acc < lam / bar).astype(z.dtype)
        mark = mu + jnp.sqrt(var) * eps
        z_post = apply_jump(z_pre, mark, accept, coeff, config.block_sizes)
        z_new = jnp.where(is_valid[:, None], z_post, z)
        s_new = jnp.where(is_valid, c, s)
        return (z_new, s_new), (s_new, z_new, is_valid)

    xs = (
        jnp.swapaxes(offsets, 0, 1),
        jnp.swapaxes(valid, 0, 1),
        accept_u,
        mark_eps,
    )
    _, (s_hist, z_hist, v_hist) = jax.lax.scan(step, (z_T, jnp.zeros((batch,), dtype=z_T.dtype)), xs)
    states = jnp.concatenate([z_T[:, None, :], jnp.swapaxes(z_hist, 0, 1)], axis=1)
    rel = jnp.swapaxes(s_hist, 0, 1)
    valid_hist = jnp.swapaxes(v_hist, 0, 1)
    return states, rel, valid_hist


def _sample_future_offsets(key, horizon, config):
    length = config.l_max
    batch = horizon.shape[0]
    k_count, k_time = jax.random.split(key)
    count = jax.random.poisson(k_count, config.lambda_bar_theta * horizon)
    count = jnp.minimum(count, length).astype(jnp.int32)
    raw = jax.random.uniform(k_time, (batch, length)) * horizon[:, None]
    slot = jnp.arange(length)
    valid = slot[None, :] < count[:, None]
    pad_times = horizon[:, None] + 1.0 + slot[None, :].astype(horizon.dtype)
    times = jnp.where(valid, raw, pad_times)
    order = jnp.argsort(times, axis=1)
    times = jnp.take_along_axis(times, order, axis=1)
    valid = jnp.take_along_axis(valid, order, axis=1)
    return times, valid


@functools.partial(
    jax.jit,
    static_argnames=(
        "include_context_events",
        "sample_future",
        "l_max",
        "obs_dim",
        "block_sizes",
        "mark_dim",
    ),
)
def _forecast_jit(
    params,
    times,
    y,
    future_times,
    key,
    future_offsets,
    include_context_events,
    sample_future,
    obs_var,
    tau_st,
    rho,
    rho_min,
    w_kl,
    beta_sparse,
    l_max,
    obs_dim,
    block_sizes,
    mark_dim,
    lambda_bar_theta,
):
    config = _static_config(
        params,
        obs_var,
        tau_st,
        rho,
        rho_min,
        w_kl,
        beta_sparse,
        n_quad=1,
        l_max=l_max,
        obs_dim=obs_dim,
        block_sizes=block_sizes,
        lambda_bar_theta=lambda_bar_theta,
    )
    config = Config(
        obs_dim=config.obs_dim,
        state_dim=config.state_dim,
        mark_dim=mark_dim,
        hidden=config.hidden,
        embed_dim=config.embed_dim,
        block_sizes=block_sizes,
        obs_var=obs_var,
        tau_st=tau_st,
        rho=rho,
        rho_min=rho_min,
        l_max=l_max,
        n_quad=1,
        w_kl=w_kl,
        beta_sparse=beta_sparse,
        lambda_bar_theta=lambda_bar_theta,
    )
    k_ctx, k_fut = jax.random.split(key)
    ctx = reconstruct_context(
        params, times, y, k_ctx, config, include_events=include_context_events
    )
    origin = times[:, -1]
    gamma = ctx["gamma"]
    if sample_future:
        span = jnp.maximum(jnp.max(future_times - origin[:, None], axis=-1), 0.0)
        offsets, valid = _sample_future_offsets(k_fut, span, config)
    else:
        offsets = future_offsets
        valid = jnp.ones(offsets.shape[:2], dtype=bool)
    states, rel, rel_valid = _simulate_future(
        ctx["z_T"], offsets, valid, k_fut, params, config, gamma
    )
    relative_query = future_times - origin[:, None]
    use_valid = rel.shape[1] > 0
    future_states = query_states(
        states,
        rel,
        relative_query,
        gamma,
        valid=rel_valid if use_valid else None,
        pre_event_on_tie=False,
    )
    return future_states, ctx["z_T"], ctx["z0"], gamma


def forecast(
    params,
    times,
    y,
    future_times,
    key,
    config=None,
    include_context_events=True,
    future_offsets=None,
):
    """Algorithm 3. Returns latent states at future_times, plus z0 and z_T.

    future_times: (F,) or (B, F), absolute times at or after the context horizon.
    future_offsets: optional relative candidate times (B, M). An empty event axis
    is the no-event forecast (pure flow from Z_T). None draws Poisson candidates
    at the fixed prior cap.
    """
    if config is None:
        config = Config()
    if future_times.ndim == 1:
        future_times = jnp.broadcast_to(future_times, (times.shape[0], future_times.shape[0]))
    sample_future = future_offsets is None
    if sample_future:
        future_offsets = jnp.zeros((times.shape[0], 0), dtype=times.dtype)
    elif future_offsets.ndim == 1:
        future_offsets = jnp.broadcast_to(future_offsets, (times.shape[0], future_offsets.shape[0]))
    future_states, z_T, z0, gamma = _forecast_jit(
        params,
        times,
        y,
        future_times,
        key,
        future_offsets,
        include_context_events,
        sample_future,
        config.obs_var,
        config.tau_st,
        config.rho,
        config.rho_min,
        config.w_kl,
        config.beta_sparse,
        config.l_max,
        config.obs_dim,
        config.block_sizes,
        config.mark_dim,
        config.lambda_bar_theta,
    )
    return {
        "future_states": future_states,
        "z_T": z_T,
        "z0": z0,
        "gamma": gamma,
    }


def synthetic_cyclic_batch(key, batch_size=4, n_times=12, dt=1.0, noise=0.1, config=None):
    """Tiny cyclic-jump window. y = z + noise * eps, 2D. Clock starts at dt > 0."""
    if config is None:
        config = Config()
    times = (jnp.arange(1, n_times + 1, dtype=jnp.float32)) * dt
    angle = 0.7 * times
    decay = jnp.exp(-0.15 * times)
    z = decay[:, None] * jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
    jump_times = jnp.asarray([4.0, 8.0]) * dt
    shifts = jnp.asarray([[0.8, -0.4], [-0.5, 0.7]], dtype=z.dtype)
    for jump_t, shift in zip(jump_times, shifts):
        elapsed = jnp.clip(times - jump_t, 0.0)
        fired = (times >= jump_t).astype(z.dtype)
        z = z + jnp.exp(-0.15 * elapsed)[:, None] * shift * fired[:, None]
    z = jnp.broadcast_to(z, (batch_size, n_times, config.obs_dim))
    eps = jax.random.normal(key, z.shape)
    y = z + noise * eps
    times = jnp.broadcast_to(times, (batch_size, n_times))
    return {"times": times, "y": y, "context": 8, "forecast": 4}
