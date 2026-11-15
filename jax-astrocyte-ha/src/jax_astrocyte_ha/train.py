"""Decoupled N-RNN MSE training and A-RNN A2C+GAE (Sec. II-C)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_astrocyte_ha.dynamics import (
    ARNNParams,
    NRNNParams,
    actor_logits,
    arnn_step,
    context_from_h,
    observation,
    unroll_nrnn,
    value,
)
from jax_astrocyte_ha.env import Config, build_trial, reach_mse, reach_reward


def sgd(params, grads, lr, max_grad_norm=1.0):
    if max_grad_norm is not None:
        leaves = jax.tree.leaves(grads)
        sq = sum(jnp.sum(g * g) for g in leaves)
        scale = jnp.minimum(1.0, max_grad_norm / jnp.sqrt(sq + 1e-8))
        grads = jax.tree.map(lambda g: g * scale, grads)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


def nrnn_mse_loss(params: NRNNParams, batch, tau_x, mask_response=True):
    """MSE of decoded reach vs target. Default: response window only."""

    def one(u, y_tgt, mask, pi):
        _, ys = unroll_nrnn(params, u, pi, tau_x)
        err = (ys - y_tgt) ** 2
        if mask_response:
            w = mask[:, None]
            return jnp.sum(err * w) / jnp.maximum(jnp.sum(w), 1.0)
        return jnp.mean(err)

    per = jax.vmap(one)(batch["u"], batch["y_tgt"], batch["mask"], batch["pi"])
    return jnp.mean(per)


def nrnn_train_step(params: NRNNParams, batch, tau_x, lr):
    def loss_fn(p):
        return nrnn_mse_loss(p, batch, tau_x)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return sgd(params, grads, lr), loss, grads


def gae(rewards, values, gamma=0.99, lam=0.95):
    """Generalized advantage estimation.

    rewards: (K,)  values: (K+1,) including bootstrap V(h_K)
    """
    deltas = rewards + gamma * values[1:] - values[:-1]

    def scan_fn(adv_next, delta):
        adv = delta + gamma * lam * adv_next
        return adv, adv

    _, advs_rev = jax.lax.scan(scan_fn, jnp.zeros((), dtype=rewards.dtype), deltas[::-1])
    advs = advs_rev[::-1]
    returns = advs + values[:-1]
    return advs, returns


def a2c_loss(params: ARNNParams, h0, actions, rewards, tau_h, cfg: Config):
    """A2C+GAE through the A-RNN (Eqs. 2-4) on a trial sequence.

    Action a_k is the context index used on trial k, taken from h_k.
    Observation o_k = (onehot(a_k), r_k) then updates h.
    """
    M = cfg.M

    def step(h, inp):
        a, r = inp
        logits = actor_logits(h, params)
        logp_all = jax.nn.log_softmax(logits)
        logp = logp_all[a]
        probs = jax.nn.softmax(logits)
        ent = -jnp.sum(probs * logp_all)
        v = value(h, params)
        pi = jax.nn.one_hot(a, M, dtype=h.dtype)
        o = observation(pi, r)
        h_next = arnn_step(h, o, params, tau_h, 0.0)
        return h_next, (logp, v, ent)

    h_last, (logps, vs, ents) = jax.lax.scan(step, h0, (actions, rewards))
    v_last = value(h_last, params)
    values = jnp.concatenate([vs, v_last[None]])
    adv, ret = gae(rewards, values, cfg.gamma, cfg.gae_lambda)
    adv = jax.lax.stop_gradient(adv)
    ret = jax.lax.stop_gradient(ret)
    policy_loss = -jnp.mean(logps * adv)
    value_loss = jnp.mean((vs - ret) ** 2)
    entropy = jnp.mean(ents)
    return policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy


def a2c_train_step(params: ARNNParams, h0, actions, rewards, tau_h, cfg: Config, lr):
    def loss_fn(p):
        return a2c_loss(p, h0, actions, rewards, tau_h, cfg)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return sgd(params, grads, lr), loss, grads


def collect_a2c_rollout(
    nrnn: NRNNParams,
    arnn: ARNNParams,
    cfg: Config,
    key,
    contexts,
    thetas,
    sample_actions=True,
):
    """Interact for K trials. Returns actions, rewards, h0, mse."""
    K = int(len(contexts))
    h = jnp.zeros((cfg.Nh,), dtype=nrnn.J_xx.dtype)
    g, pi, _, _ = context_from_h(h, arnn)
    actions = []
    rewards = []
    mses = []
    for k in range(K):
        key, k_act = jax.random.split(key)
        if sample_actions:
            logits = actor_logits(h, arnn)
            g = jax.random.categorical(k_act, logits)
            pi = jax.nn.one_hot(g, cfg.M, dtype=h.dtype)
        u, _, _, target = build_trial(thetas[k], contexts[k], cfg)
        _, ys = unroll_nrnn(nrnn, u, pi, cfg.tau_x)
        y_reach = jnp.mean(ys[cfg.resp_start :], axis=0)
        r = reach_reward(y_reach, target, cfg.reward_threshold)
        mse = reach_mse(y_reach, target)
        actions.append(g)
        rewards.append(r)
        mses.append(mse)
        o = observation(pi, r)
        h = arnn_step(h, o, arnn, cfg.tau_h, 0.0)
        g, pi, _, _ = context_from_h(h, arnn)
    actions = jnp.stack([jnp.asarray(a, dtype=jnp.int32) for a in actions])
    rewards = jnp.stack(rewards)
    mses = jnp.stack(mses)
    h0 = jnp.zeros((cfg.Nh,), dtype=nrnn.J_xx.dtype)
    return h0, actions, rewards, mses


def hybrid_rollout(
    nrnn: NRNNParams,
    arnn: ARNNParams,
    cfg: Config,
    contexts,
    thetas,
    pi_override=None,
):
    """Eval rollout with argmax π (or a hardwired π per trial). No NaNs expected."""
    K = int(len(contexts))
    h = jnp.zeros((cfg.Nh,), dtype=nrnn.J_xx.dtype)
    g, pi, _, _ = context_from_h(h, arnn)
    rewards = []
    mses = []
    pis = []
    for k in range(K):
        if pi_override is not None:
            pi = pi_override[k]
        u, _, _, target = build_trial(thetas[k], contexts[k], cfg)
        _, ys = unroll_nrnn(nrnn, u, pi, cfg.tau_x)
        y_reach = jnp.mean(ys[cfg.resp_start :], axis=0)
        r = reach_reward(y_reach, target, cfg.reward_threshold)
        mse = reach_mse(y_reach, target)
        rewards.append(r)
        mses.append(mse)
        pis.append(pi)
        o = observation(pi, r)
        h = arnn_step(h, o, arnn, cfg.tau_h, 0.0)
        _, pi, _, _ = context_from_h(h, arnn)
    return {
        "rewards": jnp.stack(rewards),
        "mses": jnp.stack(mses),
        "pis": jnp.stack(pis),
        "h": h,
    }


def mse_with_pi(nrnn: NRNNParams, batch, tau_x, pi):
    """Reach MSE using a supplied π (broadcast or per-trial)."""
    if pi.ndim == 1:
        pi = jnp.broadcast_to(pi, (batch["u"].shape[0], pi.shape[0]))

    def one(u, y_tgt, mask, p):
        _, ys = unroll_nrnn(nrnn, u, p, tau_x)
        err = (ys - y_tgt) ** 2
        w = mask[:, None]
        return jnp.sum(err * w) / jnp.maximum(jnp.sum(w), 1.0)

    return jnp.mean(jax.vmap(one)(batch["u"], batch["y_tgt"], batch["mask"], pi))


def teacher_pi_from_context(context, M):
    return jax.nn.one_hot(context, M)


def wrong_pi_from_context(context, M):
    return jax.nn.one_hot((context + 1) % M, M)
