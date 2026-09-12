"""Paper identities for Suan (arXiv:2609.08634)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from suan.losses import (
    DEFAULT_BETA,
    DEFAULT_TAU,
    preference_closed_form_dlog,
    preference_gradient_coefficient,
    preference_loss,
    regularizer_closed_form_dlog,
    regularizer_gradient_coefficient,
    regularizer_loss,
    suan_loss,
)
from suan.policy import log_prob, log_softmax_rows, softmax_rows

jax.config.update("jax_enable_x64", True)


def _toy_batch(key=jax.random.PRNGKey(0), batch=8, n_prompts=5, n_actions=6):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    logits = jax.random.normal(k1, (n_prompts, n_actions), dtype=jnp.float64)
    ref_logits = jax.random.normal(k2, (n_prompts, n_actions), dtype=jnp.float64)
    x = jax.random.randint(k3, (batch,), 0, n_prompts)
    y_plus = jax.random.randint(k4, (batch,), 0, n_actions)
    y_minus = (y_plus + 1 + jax.random.randint(k4, (batch,), 0, n_actions - 1)) % n_actions
    return logits, ref_logits, x, y_plus, y_minus


def test_neg_log_sigmoid_derivative_matches_appendix():
    """d/dt [−log σ(t)] = −(1−σ(t)) = −σ(−t)."""
    t = jnp.linspace(-6.0, 6.0, 41)

    def neg_log_sigma(u):
        return jax.nn.softplus(-u)

    grads = jax.vmap(jax.grad(neg_log_sigma))(t)
    expected_sigma_neg = -jax.nn.sigmoid(-t)
    expected_one_minus = -(1.0 - jax.nn.sigmoid(t))
    assert jnp.allclose(grads, expected_sigma_neg, rtol=1e-10, atol=1e-12)
    assert jnp.allclose(grads, expected_one_minus, rtol=1e-10, atol=1e-12)


def test_tanh_sigmoid_identity():
    """tanh(t) = 2σ(2t) − 1."""
    t = jnp.linspace(-5.0, 5.0, 51)
    assert jnp.allclose(jnp.tanh(t), 2.0 * jax.nn.sigmoid(2.0 * t) - 1.0, rtol=1e-10, atol=1e-12)


def test_preference_autodiff_matches_prop_31_logprob_space():
    """jax.grad of L_P matches Proposition 3.1 in log-prob space."""
    logits, _, x, y_plus, y_minus = _toy_batch()
    log_plus = log_prob(logits, x, y_plus)
    log_minus = log_prob(logits, x, y_minus)
    tau = DEFAULT_TAU

    def mean_lp(lp, lm):
        return jnp.mean(preference_loss(lp, lm, tau=tau))

    g_plus, g_minus = jax.grad(mean_lp, argnums=(0, 1))(log_plus, log_minus)
    c_plus, c_minus = preference_closed_form_dlog(log_plus, log_minus, tau=tau)
    assert jnp.allclose(g_plus, c_plus / log_plus.shape[0], rtol=1e-8, atol=1e-10)
    assert jnp.allclose(g_minus, c_minus / log_minus.shape[0], rtol=1e-8, atol=1e-10)

    coeff = preference_gradient_coefficient(log_plus, log_minus, tau=tau)
    assert jnp.allclose(coeff, -jax.nn.sigmoid(log_minus - log_plus - tau))


def test_preference_autodiff_matches_prop_31_param_space():
    """jax.grad of L_P w.r.t. logits matches Eq (6) times (∇logπ+ − ∇logπ−)."""
    logits, _, x, y_plus, y_minus = _toy_batch()
    tau = DEFAULT_TAU
    n_actions = logits.shape[1]
    batch = x.shape[0]

    def mean_lp(theta):
        return jnp.mean(
            preference_loss(log_prob(theta, x, y_plus), log_prob(theta, x, y_minus), tau=tau)
        )

    auto = jax.grad(mean_lp)(logits)

    log_plus = log_prob(logits, x, y_plus)
    log_minus = log_prob(logits, x, y_minus)
    coeff = preference_gradient_coefficient(log_plus, log_minus, tau=tau)
    probs = softmax_rows(logits)
    closed = jnp.zeros_like(logits)
    for i in range(batch):
        xi = int(x[i])
        e_plus = jax.nn.one_hot(y_plus[i], n_actions, dtype=logits.dtype)
        e_minus = jax.nn.one_hot(y_minus[i], n_actions, dtype=logits.dtype)
        dlog_plus = e_plus - probs[xi]
        dlog_minus = e_minus - probs[xi]
        closed = closed.at[xi].add(coeff[i] * (dlog_plus - dlog_minus) / batch)

    assert jnp.allclose(auto, closed, rtol=1e-7, atol=1e-9)


def test_regularizer_autodiff_matches_prop_32_logprob_space():
    """jax.grad of L_R matches Proposition 3.2 in log-prob space."""
    logits, ref_logits, x, y_plus, y_minus = _toy_batch()
    log_plus = log_prob(logits, x, y_plus)
    log_minus = log_prob(logits, x, y_minus)
    ref_plus = log_prob(ref_logits, x, y_plus)
    ref_minus = log_prob(ref_logits, x, y_minus)

    def mean_lr(lp, lm):
        return jnp.mean(regularizer_loss(lp, lm, ref_plus, ref_minus))

    g_plus, g_minus = jax.grad(mean_lr, argnums=(0, 1))(log_plus, log_minus)
    c_plus, c_minus = regularizer_closed_form_dlog(log_plus, log_minus, ref_plus, ref_minus)
    assert jnp.allclose(g_plus, c_plus / log_plus.shape[0], rtol=1e-8, atol=1e-10)
    assert jnp.allclose(g_minus, c_minus / log_minus.shape[0], rtol=1e-8, atol=1e-10)


def test_regularizer_autodiff_matches_prop_32_param_space():
    """jax.grad of L_R w.r.t. logits matches Eq (8)."""
    logits, ref_logits, x, y_plus, y_minus = _toy_batch()
    n_actions = logits.shape[1]
    batch = x.shape[0]
    ref_plus = log_prob(ref_logits, x, y_plus)
    ref_minus = log_prob(ref_logits, x, y_minus)

    def mean_lr(theta):
        return jnp.mean(
            regularizer_loss(
                log_prob(theta, x, y_plus),
                log_prob(theta, x, y_minus),
                ref_plus,
                ref_minus,
            )
        )

    auto = jax.grad(mean_lr)(logits)
    log_plus = log_prob(logits, x, y_plus)
    log_minus = log_prob(logits, x, y_minus)
    c_plus = regularizer_gradient_coefficient(log_plus, ref_plus)
    c_minus = regularizer_gradient_coefficient(log_minus, ref_minus)
    probs = softmax_rows(logits)
    closed = jnp.zeros_like(logits)
    for i in range(batch):
        xi = int(x[i])
        e_plus = jax.nn.one_hot(y_plus[i], n_actions, dtype=logits.dtype)
        e_minus = jax.nn.one_hot(y_minus[i], n_actions, dtype=logits.dtype)
        closed = closed.at[xi].add(
            (c_plus[i] * (e_plus - probs[xi]) + c_minus[i] * (e_minus - probs[xi])) / batch
        )
    assert jnp.allclose(auto, closed, rtol=1e-7, atol=1e-9)


def test_suan_is_lp_plus_beta_lr():
    logits, ref_logits, x, y_plus, y_minus = _toy_batch()
    log_plus = log_prob(logits, x, y_plus)
    log_minus = log_prob(logits, x, y_minus)
    ref_plus = log_prob(ref_logits, x, y_plus)
    ref_minus = log_prob(ref_logits, x, y_minus)
    beta = DEFAULT_BETA
    tau = DEFAULT_TAU
    combined = suan_loss(log_plus, log_minus, ref_plus, ref_minus, tau=tau, beta=beta)
    expected = preference_loss(log_plus, log_minus, tau=tau) + beta * regularizer_loss(
        log_plus, log_minus, ref_plus, ref_minus
    )
    assert jnp.allclose(combined, expected, rtol=1e-12, atol=1e-14)


def test_regularizer_coefficient_is_bounded():
    """σ(log(π_θ/π_ref)) − 1/2 ∈ (−1/2, 1/2) for finite log-ratios."""
    ratios = jnp.array([-20.0, -8.0, -1.0, 0.0, 0.3, 2.0, 12.0, 20.0])
    coeff = regularizer_gradient_coefficient(ratios, jnp.zeros_like(ratios))
    assert jnp.all(coeff > -0.5)
    assert jnp.all(coeff < 0.5)
    logits, ref_logits, x, y_plus, y_minus = _toy_batch()
    log_plus = log_prob(logits, x, y_plus)
    ref_plus = log_prob(ref_logits, x, y_plus)
    batch_coeff = regularizer_gradient_coefficient(log_plus, ref_plus)
    assert jnp.all(batch_coeff > -0.5)
    assert jnp.all(batch_coeff < 0.5)


def test_preference_loss_near_zero_when_ranking_is_correct():
    tau = DEFAULT_TAU
    large_margin = preference_loss(jnp.array(12.0), jnp.array(-12.0), tau=tau)
    reversed_margin = preference_loss(jnp.array(-12.0), jnp.array(12.0), tau=tau)
    tied = preference_loss(jnp.array(0.0), jnp.array(0.0), tau=tau)
    assert large_margin < 1e-4
    assert reversed_margin > tied > large_margin
    assert reversed_margin > 1.0


def test_sgd_increases_preferred_logprob_margin():
    logits, ref_logits, x, y_plus, y_minus = _toy_batch(jax.random.PRNGKey(7), batch=16)
    ref_plus = log_prob(ref_logits, x, y_plus)
    ref_minus = log_prob(ref_logits, x, y_minus)

    def mean_suan(theta):
        return jnp.mean(
            suan_loss(
                log_prob(theta, x, y_plus),
                log_prob(theta, x, y_minus),
                ref_plus,
                ref_minus,
                tau=DEFAULT_TAU,
                beta=DEFAULT_BETA,
            )
        )

    def margin(theta):
        return jnp.mean(log_prob(theta, x, y_plus) - log_prob(theta, x, y_minus))

    before = float(margin(logits))
    step = jax.grad(mean_suan)
    for _ in range(25):
        logits = logits - 0.5 * step(logits)
    after = float(margin(logits))
    assert after > before + 0.05


def test_losses_vectorize_over_batch():
    logits, ref_logits, x, y_plus, y_minus = _toy_batch(batch=10)
    log_plus = log_prob(logits, x, y_plus)
    log_minus = log_prob(logits, x, y_minus)
    ref_plus = log_prob(ref_logits, x, y_plus)
    ref_minus = log_prob(ref_logits, x, y_minus)
    lp = preference_loss(log_plus, log_minus)
    lr = regularizer_loss(log_plus, log_minus, ref_plus, ref_minus)
    ls = suan_loss(log_plus, log_minus, ref_plus, ref_minus)
    assert lp.shape == (10,)
    assert lr.shape == (10,)
    assert ls.shape == (10,)
    per = jnp.stack(
        [
            suan_loss(log_plus[i], log_minus[i], ref_plus[i], ref_minus[i])
            for i in range(10)
        ]
    )
    assert jnp.allclose(ls, per, rtol=1e-10, atol=1e-12)


def test_log_probs_are_normalized():
    logits, _, _, _, _ = _toy_batch()
    log_pi = log_softmax_rows(logits)
    assert jnp.allclose(jnp.exp(jax.nn.logsumexp(log_pi, axis=-1)), 1.0, rtol=1e-10)


def test_defaults_match_paper_section_4():
    assert DEFAULT_TAU == 1.0
    assert DEFAULT_BETA == 0.1
