"""Hierarchical blockwise diffusion sampling with exact scores.

Analytic model (task form of section 4.2 / Appendix B.1, coefficient 1 on the sign):

    theta ~ N(0, 1)
    z_g ~ Rademacher(1/2)
    y_{g,i} | theta, z_g ~ N(theta + z_g, 1)

Two groups, opposite true signs, 16 observations each. Appendix B.1 uses
N(5 z + theta, 1) and a single observation in the closed form; section 4.2 uses
10 observations per group with that coefficient of 5. With coefficient 1 the
opposite-sign posterior is stable at 16 observations per group, which is the
count the task allows. Eight is not used: on some seeds the coefficient-1
posterior is not concentrated on the true signs.

Continuous embedding of the discrete sign
    z lives on {-1, +1}. The sampler diffuses a continuous coordinate. Conditional
    on a sign, theta is Gaussian and z is a point mass. VP noising of that
    two-point posterior is a two-component Gaussian mixture in (theta, z).
    score_phi returns the exact mixture score. This is the two-point score under
    the paper's continuous embedding, not a trained network, so the check does
    not depend on optimization.

Algorithm 1, simplified to the task's sampler
    One score function score_phi(x, t, y) does not take a group index. The
    sampler splits the state into blocks and calls that function once per block
    with that block's observations.
    Each interval is one Euler-Maruyama step of the reverse SDE (M = 1 block
    round, L = 0 Langevin corrections). Group latents are updated from their own
    block drift only. The shared theta is updated from the average of the
    per-block drifts, as specified for this implementation. Re-noising rounds
    are not used.
    The block score is the exact joint score of that group's posterior, so the
    matching forward process is ordinary VP. That is g_1(t)^2 = beta(t), not
    g_n for the observation count: g_n is the coefficient for composing n
    single-observation diffusion scores, which this exact-score sampler does not do.
"""

import jax
import jax.numpy as jnp

from hbds.diffusion import beta_schedule, g_n_squared, signal_scale

N_PER_GROUP = 16
THETA_TRUE = 1.0
TRUE_SIGNS = jnp.array([1.0, -1.0])


def make_grouped_observations(key, n_per_group=N_PER_GROUP, theta=THETA_TRUE, signs=TRUE_SIGNS):
    """Draw two groups. signs has one Rademacher value per group."""
    signs = jnp.asarray(signs, dtype=jnp.float64)
    k = signs.shape[0]
    keys = jax.random.split(key, k)
    noise = jax.vmap(lambda kk: jax.random.normal(kk, (n_per_group,), dtype=jnp.float64))(keys)
    ys = theta + signs[:, None] + noise
    return ys, signs, jnp.asarray(theta, dtype=jnp.float64)


def _group_mixture_params(y):
    """Weights, theta means, theta variance, and signs for one group's posterior."""
    y = jnp.asarray(y, dtype=jnp.float64)
    n = y.shape[0]
    zs = jnp.array([-1.0, 1.0], dtype=jnp.float64)
    resid = y[:, None] - zs[None, :]
    quad = jnp.sum(resid ** 2, axis=0) - (jnp.sum(resid, axis=0) ** 2) / (n + 1.0)
    logw = -0.5 * quad - 0.5 * jnp.log(n + 1.0)
    logw = logw - jnp.max(logw)
    w = jnp.exp(logw)
    w = w / jnp.sum(w)
    v = 1.0 / (n + 1.0)
    m = (jnp.sum(y) - n * zs) / (n + 1.0)
    return w, m, v, zs


def score_phi(x, t, y, var_floor=1e-8):
    """Exact score of the VP-noised group posterior at x = (theta, z).

    y is this context's observations. The function has no group index and no
    hierarchy. t is diffusion time.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    w, m, v, zs = _group_mixture_params(y)
    a = signal_scale(t)
    sig2 = jnp.maximum(1.0 - a * a, var_floor)
    mu = jnp.stack([a * m, a * zs], axis=1)
    var = jnp.array([a * a * v + sig2, sig2], dtype=jnp.float64)
    diff = x[None, :] - mu
    log_n = -0.5 * jnp.sum(diff ** 2 / var[None, :], axis=-1) - 0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi * var)
    )
    log_r = jnp.log(w) + log_n
    log_r = log_r - jnp.max(log_r)
    r = jnp.exp(log_r)
    r = r / jnp.sum(r)
    return jnp.sum(r[:, None] * (mu - x[None, :]) / var[None, :], axis=0)


def _em_step(theta, z, t, dt, observations, key):
    """One reverse-time Euler step. theta is shared, z is (n_samples, K)."""
    n_samples, k = z.shape
    bt = beta_schedule(t)
    g2 = g_n_squared(t, 1)
    scale = jnp.sqrt(g2 * dt)

    s_theta = []
    s_z = []
    for group in range(k):
        obs = observations[group]
        states = jnp.stack([theta, z[:, group]], axis=1)
        sk = jax.vmap(lambda st: score_phi(st, t, obs))(states)
        s_theta.append(sk[:, 0])
        s_z.append(sk[:, 1])
    s_theta = jnp.stack(s_theta, axis=1)
    s_z = jnp.stack(s_z, axis=1)

    # Shared parameter: average of per-block drifts. Latents: own block only.
    score_theta = jnp.mean(s_theta, axis=1)
    drift_theta = 0.5 * bt * theta + g2 * score_theta
    drift_z = 0.5 * bt * z + g2 * s_z
    noise = jax.random.normal(key, (n_samples, 1 + k), dtype=jnp.float64)
    theta = theta + drift_theta * dt + scale * noise[:, 0]
    z = z + drift_z * dt + scale * noise[:, 1:]
    return theta, z


def hbds_sample(observations, n_samples, n_steps, key, t_start=1.0, t_end=0.02):
    """Algorithm 1 loop with averaged shared drifts and per-block latents.

    observations: (K, n_per_group). Returns theta (n_samples,) and z (n_samples, K).
    """
    observations = jnp.asarray(observations, dtype=jnp.float64)
    k = observations.shape[0]
    key, k_th, k_z = jax.random.split(key, 3)
    theta = jax.random.normal(k_th, (n_samples,), dtype=jnp.float64)
    z = jax.random.normal(k_z, (n_samples, k), dtype=jnp.float64)
    ts = jnp.linspace(t_start, t_end, n_steps + 1)

    def step(carry, i):
        theta, z, key = carry
        key, kn = jax.random.split(key)
        t = ts[i]
        dt = ts[i] - ts[i + 1]
        theta, z = _em_step(theta, z, t, dt, observations, kn)
        return (theta, z, key), None

    (theta, z, _), _ = jax.lax.scan(step, (theta, z, key), jnp.arange(n_steps))
    return theta, z


def pooled_sample(observations, n_samples, n_steps, key, t_start=1.0, t_end=0.02):
    """Pooled sampler: one shared sign for every observation.

    observations is a flat vector. The same score_phi is used; the hierarchy is
    removed by tying every group to that one latent.
    """
    observations = jnp.asarray(observations, dtype=jnp.float64).reshape(-1)
    key, k_th, k_z = jax.random.split(key, 3)
    theta = jax.random.normal(k_th, (n_samples,), dtype=jnp.float64)
    z = jax.random.normal(k_z, (n_samples,), dtype=jnp.float64)
    ts = jnp.linspace(t_start, t_end, n_steps + 1)
    packed = observations[None, :]

    def step(carry, i):
        theta, z, key = carry
        key, kn = jax.random.split(key)
        t = ts[i]
        dt = ts[i] - ts[i + 1]
        theta, z_block = _em_step(theta, z[:, None], t, dt, packed, kn)
        return (theta, z_block[:, 0], key), None

    (theta, z, _), _ = jax.lax.scan(step, (theta, z, key), jnp.arange(n_steps))
    return theta, z


def sign_agreement(latent, true_signs):
    """Fraction of draws whose signs match true_signs on every group.

    latent: (n_samples, n_groups). Agreement is computed from the samples; it is
    not a stored constant.
    """
    pred = jnp.sign(latent)
    ok = jnp.all(pred == jnp.asarray(true_signs, dtype=jnp.float64), axis=-1)
    return jnp.mean(ok.astype(jnp.float64))


def grouped_sign_posterior_mass(observations, true_signs):
    """Analytic posterior mass on the true sign pair, theta marginalized.

    Used to confirm a dataset is identifiable before comparing samplers. Not a
    sampler result.
    """
    observations = jnp.asarray(observations, dtype=jnp.float64)
    true_signs = jnp.asarray(true_signs, dtype=jnp.float64)
    options = jnp.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=jnp.float64,
    )

    def log_marg(signs):
        resid = observations - signs[:, None]
        n = resid.size
        quad = jnp.sum(resid ** 2) - (jnp.sum(resid) ** 2) / (n + 1.0)
        return -0.5 * quad - 0.5 * jnp.log(n + 1.0)

    lps = jax.vmap(log_marg)(options)
    lps = lps - jnp.max(lps)
    p = jnp.exp(lps)
    p = p / jnp.sum(p)
    match = jnp.all(options == true_signs, axis=1)
    return jnp.sum(p * match)
