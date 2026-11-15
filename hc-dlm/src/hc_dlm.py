"""Tiny JAX slice of a hierarchical continuous diffusion language model.

Sequence length n, vocab size V including a mask id (mask = V - 1). Latent
x lives in R^{n x d}. Tokens k are integer ids.

The discrete kernel is an absorbing mask kernel with alpha(s) = s. The
encoder, token predictor, and denoiser are position-wise one-hidden-layer
MLPs. This is not the paper's Sudoku, Countdown, or LM1B stack.

Objective (entropy term is the closed form, not a second Monte Carlo):

    J = lambda_recon * mean(-log p_theta(k0 | stopgrad(x0)))
      + lambda_ent * mean(-H(q_psi(x0 | k0)))
      + lambda_cont * mean(omega * ||d_phi(x_t, k_t, s) - stopgrad(x0)||^2)

E[log q_psi(x0 | k0)] = -H for the reparameterized diagonal Gaussian, so the
entropy term matches mean(log q) without extra noise. H is

    H = 0.5 * sum(log(2 * pi * e * var))
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

import optax


def gaussian_entropy(var, axis=None):
    """Closed-form entropy of a diagonal Gaussian.

    H = 0.5 * sum(log(2 * pi * e * var)). `axis=None` sums every element.
    """
    var = jnp.asarray(var)
    terms = jnp.log(2.0 * jnp.pi * jnp.e * var)
    if axis is None:
        return 0.5 * jnp.sum(terms)
    return 0.5 * jnp.sum(terms, axis=axis)


def gaussian_log_density(x, mean, log_std, axis=None):
    """Diagonal-Gaussian log-density, summed over `axis` (default: all axes)."""
    x = jnp.asarray(x)
    mean = jnp.asarray(mean)
    log_std = jnp.asarray(log_std)
    var = jnp.exp(2.0 * log_std)
    log_p = -0.5 * (jnp.log(2.0 * jnp.pi * var) + (x - mean) ** 2 / var)
    if axis is None:
        return jnp.sum(log_p)
    return jnp.sum(log_p, axis=axis)


def reparameterize(mean, log_std, eps):
    """x0 = mean + std * eps, with std = exp(log_std)."""
    return mean + jnp.exp(log_std) * eps


def _scale(key, shape, scale):
    return random.normal(key, shape) * scale


def init_params(key, vocab, d, hidden, embed_dim=None, time_dim=8, init_scale=0.02):
    """Array-only parameter tree. Mask id is vocab - 1, inferred from shapes."""
    if embed_dim is None:
        embed_dim = d
    if vocab < 2:
        raise ValueError("vocab must include at least one real token and the mask")
    k_enc, k_pred, k_den = random.split(key, 3)
    ke = random.split(k_enc, 5)
    kp = random.split(k_pred, 2)
    kd = random.split(k_den, 3)
    real = vocab - 1
    return {
        "encoder": {
            "tok_embed": _scale(ke[0], (vocab, embed_dim), init_scale),
            "w1": _scale(ke[1], (embed_dim, hidden), init_scale),
            "b1": jnp.zeros((hidden,)),
            "w_mean": _scale(ke[2], (hidden, d), init_scale),
            "b_mean": jnp.zeros((d,)),
            "w_logstd": _scale(ke[3], (hidden, d), init_scale),
            "b_logstd": jnp.zeros((d,)),
        },
        "predictor": {
            "w1": _scale(kp[0], (d, hidden), init_scale),
            "b1": jnp.zeros((hidden,)),
            "w2": _scale(kp[1], (hidden, real), init_scale),
            "b2": jnp.zeros((real,)),
        },
        "denoiser": {
            "tok_embed": _scale(kd[0], (vocab, embed_dim), init_scale),
            "w1": _scale(kd[1], (d + embed_dim + time_dim, hidden), init_scale),
            "b1": jnp.zeros((hidden,)),
            "w2": _scale(kd[2], (hidden, d), init_scale),
            "b2": jnp.zeros((d,)),
        },
    }


def mask_id(params):
    return int(params["denoiser"]["tok_embed"].shape[0]) - 1


def _coeff(s, ref):
    """Broadcast a scalar or per-batch time onto `ref`."""
    s = jnp.asarray(s, dtype=ref.dtype)
    if s.ndim == 0:
        return s
    while s.ndim < ref.ndim:
        s = s[..., None]
    return s


def _batch_time(s, batch, dtype):
    s = jnp.asarray(s, dtype=dtype)
    if s.ndim == 0:
        return jnp.full((batch,), s)
    return s


def sinusoidal_time_embed(s, dim):
    """Sinusoidal embedding of scalar times. s is (B,), result is (B, dim)."""
    s = jnp.asarray(s)
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=s.dtype) / jnp.maximum(half, 1))
    ang = s[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.zeros((s.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


def encode(params, k0, eps):
    """Factorized Gaussian encoder q_psi(x0 | k0). Returns x0, mean, log_std."""
    emb = params["encoder"]["tok_embed"][k0]
    hidden = jax.nn.gelu(emb @ params["encoder"]["w1"] + params["encoder"]["b1"])
    mean = hidden @ params["encoder"]["w_mean"] + params["encoder"]["b_mean"]
    # Bound log-std. Minimizing -H pushes variance up with no prior restoring
    # force; an unbounded std makes the x0-prediction target explode under Adam.
    log_std = hidden @ params["encoder"]["w_logstd"] + params["encoder"]["b_logstd"]
    log_std = jnp.clip(log_std, -6.0, 0.0)
    x0 = reparameterize(mean, log_std, eps)
    return x0, mean, log_std


def token_logits(params, x):
    """Logits over the V - 1 real tokens. The mask id is not a class."""
    hidden = jax.nn.gelu(x @ params["predictor"]["w1"] + params["predictor"]["b1"])
    return hidden @ params["predictor"]["w2"] + params["predictor"]["b2"]


def sequence_log_prob(logits, tokens):
    """Joint log-prob of a token row under a factorized categorical. Shape (B,)."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    chosen = jnp.take_along_axis(log_probs, tokens[..., None], axis=-1)[..., 0]
    return jnp.sum(chosen, axis=-1)


def reconstruction_nll(params, k0, x):
    """mean(-log p_theta(k0 | stopgrad(x))). The stop-gradient blocks whoever built x."""
    logits = token_logits(params, jax.lax.stop_gradient(x))
    return jnp.mean(-sequence_log_prob(logits, k0))


def denoise(params, x_t, k_t, s):
    """d_phi(x_t, k_t, s): one-hidden-layer MLP, x0 estimate of shape (B, n, d).

    Input is the concatenation of the noisy latent, a learned embedding of the
    current tokens (the scaffold), and a sinusoidal embedding of time s.
    """
    batch, length, d = x_t.shape
    embed = params["denoiser"]["tok_embed"][k_t]
    time_dim = params["denoiser"]["w1"].shape[0] - d - embed.shape[-1]
    s_b = _batch_time(s, batch, x_t.dtype)
    t_emb = sinusoidal_time_embed(s_b, time_dim)
    t_emb = jnp.broadcast_to(t_emb[:, None, :], (batch, length, time_dim))
    inp = jnp.concatenate([x_t, embed, t_emb], axis=-1)
    hidden = jax.nn.gelu(inp @ params["denoiser"]["w1"] + params["denoiser"]["b1"])
    return hidden @ params["denoiser"]["w2"] + params["denoiser"]["b2"]


def interpolant(x0, eps, s):
    """Equation (29): x_t = (1 - s) * x0 + s * eps, s in [0, 1]."""
    s_b = _coeff(s, x0)
    return (1.0 - s_b) * x0 + s_b * eps


def absorbing_forward(k0, s, key, mask_id):
    """Replace each position by mask with probability alpha(s) = s, else keep k0."""
    u = random.uniform(key, k0.shape)
    thresh = _coeff(s, k0.astype(u.dtype))
    return jnp.where(u < thresh, jnp.asarray(mask_id, dtype=k0.dtype), k0)


def reverse_step(params, x_t, k_t, s, s_prev, key):
    """Absorbing reverse step from s down to s_prev < s.

    Unmasked positions stay. A masked position is unmasked with probability
    (s - s_prev) / s and then drawn from p_theta(. | x0_hat), which has no
    mask class. Otherwise it stays masked.

    x0_hat = d_phi(x_t, k_t, s) is the scaffold the token predictor reads.
    Newly sampled tokens are the scaffold for the next latent update, not for
    this one. This step's latent update is

        x_prev = (1 - s_prev) * x0_hat + s_prev * eps_new.

    Returns x_prev, k_prev, x0_hat.
    """
    key_unmask, key_tok, key_eps = random.split(key, 3)
    x0_hat = denoise(params, x_t, k_t, s)
    logits = token_logits(params, x0_hat)
    sampled = random.categorical(key_tok, logits, axis=-1)
    mid = mask_id(params)
    is_masked = k_t == mid
    s_row = _batch_time(s, k_t.shape[0], x_t.dtype)
    sp_row = _batch_time(s_prev, k_t.shape[0], x_t.dtype)
    prob = (s_row - sp_row) / jnp.maximum(s_row, jnp.asarray(1e-8, dtype=x_t.dtype))
    u = random.uniform(key_unmask, k_t.shape)
    unmask = is_masked & (u < prob[:, None])
    k_prev = jnp.where(unmask, sampled, k_t)
    eps_new = random.normal(key_eps, x_t.shape)
    sp = _coeff(s_prev, x0_hat)
    x_prev = (1.0 - sp) * x0_hat + sp * eps_new
    return x_prev, k_prev, x0_hat


def objective_terms(
    params,
    k0,
    eps_enc,
    eps_flow,
    s,
    mask_key,
    lambda_recon=1.0,
    lambda_ent=0.01,
    lambda_cont=1.0,
    omega=None,
):
    """Per-term objective. Entropy is closed form, so J is fixed given the noise."""
    x0, _mean, log_std = encode(params, k0, eps_enc)
    var = jnp.exp(2.0 * log_std)
    entropy = gaussian_entropy(var, axis=(-1, -2))
    x0_sg = jax.lax.stop_gradient(x0)
    nll = -sequence_log_prob(token_logits(params, x0_sg), k0)
    # Detach the interpolant sample as well as the target. J's value matches
    # (1 - s) * x0 + s * eps; the stop-gradient keeps this an x0-regression
    # for the denoiser instead of a moving-target chase through q_psi.
    x_t = interpolant(x0_sg, eps_flow, s)
    k_t = absorbing_forward(k0, s, mask_key, mask_id(params))
    x0_hat = denoise(params, x_t, k_t, s)
    sq = jnp.sum((x0_hat - x0_sg) ** 2, axis=(-1, -2))
    if omega is None:
        omega = jnp.ones_like(sq)
    else:
        omega = jnp.asarray(omega, dtype=sq.dtype)
        if omega.ndim == 0:
            omega = jnp.full_like(sq, omega)
    recon = jnp.mean(nll)
    ent = jnp.mean(-entropy)
    cont = jnp.mean(omega * sq)
    J = lambda_recon * recon + lambda_ent * ent + lambda_cont * cont
    return {"J": J, "recon": recon, "ent": ent, "cont": cont, "k_t": k_t, "x0": x0}


def objective(
    params,
    k0,
    eps_enc,
    eps_flow,
    s,
    mask_key,
    lambda_recon=1.0,
    lambda_ent=0.01,
    lambda_cont=1.0,
    omega=None,
):
    """Full variational token-and-latent loss J."""
    return objective_terms(
        params,
        k0,
        eps_enc,
        eps_flow,
        s,
        mask_key,
        lambda_recon=lambda_recon,
        lambda_ent=lambda_ent,
        lambda_cont=lambda_cont,
        omega=omega,
    )["J"]


def _sample_noise(key, k0, d):
    k1, k2, k3, k4 = random.split(key, 4)
    batch, length = k0.shape
    eps_enc = random.normal(k1, (batch, length, d))
    eps_flow = random.normal(k2, (batch, length, d))
    s = random.uniform(k3, (batch,), minval=1e-3, maxval=1.0 - 1e-3)
    return eps_enc, eps_flow, s, k4


def train_copy_task(steps=200, seed=0, batch=32, lr=1e-3):
    """Copy-task smoke train. Vocab is 6 real tokens plus mask, length 4, d=8.

    Targets are random token rows. Times are uniform in (0, 1). Returns the
    objective on one fixed noise batch before and after `steps` Adam updates.
    """
    vocab = 7
    length = 4
    d = 8
    hidden = 32
    key = random.PRNGKey(seed)
    key, k_init, k_data, k_eval, k_steps = random.split(key, 5)
    params = init_params(k_init, vocab=vocab, d=d, hidden=hidden, embed_dim=d, time_dim=d)
    k0 = random.randint(k_data, (batch, length), 0, vocab - 1)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_on(p, noise_key):
        eps_enc, eps_flow, s, mask_key = _sample_noise(noise_key, k0, d)
        return objective(
            p,
            k0,
            eps_enc,
            eps_flow,
            s,
            mask_key,
            lambda_recon=1.0,
            lambda_ent=0.01,
            lambda_cont=1.0,
        )

    initial = loss_on(params, k_eval)

    @jax.jit
    def run(params, opt_state, step_keys):
        def body(carry, step_key):
            params, opt_state = carry
            eps_enc, eps_flow, s, mask_key = _sample_noise(step_key, k0, d)

            def loss_fn(p):
                return objective(
                    p,
                    k0,
                    eps_enc,
                    eps_flow,
                    s,
                    mask_key,
                    lambda_recon=1.0,
                    lambda_ent=0.01,
                    lambda_cont=1.0,
                )

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        return jax.lax.scan(body, (params, opt_state), step_keys)

    step_keys = random.split(k_steps, steps)
    (params, _opt_state), step_losses = run(params, opt_state, step_keys)
    final = loss_on(params, k_eval)
    return {
        "initial_loss": float(initial),
        "final_loss": float(final),
        "step_losses": step_losses,
        "params": params,
        "k0": k0,
    }
