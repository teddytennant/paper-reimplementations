"""MetaQuill-KIP: the hybrid of Sec. III-C.

Four steps, in the paper's numbering:

1. theta_S comes from MetaQuill (Eq. 7), and stays frozen from here on.
2. A KIP support set S_T = {(x~_m, y~_m)} is distilled on the new task by
   minimizing Eq. 9, restricted to the observed pixels when inpainting.
3. That kernel solution is lifted into parameter space, Eq. 10. Since K_{theta_S}
   is the tangent kernel at theta_S, the residual that reproduces the kernel
   prediction to first order is

       delta_theta_KIP = J_{theta_S}(X~)^T (K_{theta_S}(X~, X~) + lambda I)^{-1} Y~

   because J(x) J(X~)^T = K_{theta_S}(x, X~). This is Eq. 15 with the observed
   set replaced by the distilled support.
4. Only delta_theta is refined, non-linearly, on the observed pixels (Eq. 11),
   with theta_S frozen and delta_theta initialized at delta_theta_KIP.
"""

import jax
import jax.numpy as jnp
import optax

from metaquill_kip.kip import ntk_kip, support_features
from metaquill_kip.metaquill import (
    flat_field,
    lift_to_weights,
    nonlinear_predict,
    tangent_predict,
)
from metaquill_kip.ntk import kernel_solve, ntk_matrix


def kip_warm_start(f, params_S, support, labels, ridge, encode_fn, space="coord"):
    """Step 3, Eq. 10: turn the distilled support into a weight-space residual."""
    g, flat0, _ = flat_field(f, params_S)
    Zs = support_features(support, encode_fn, space)
    Kss = ntk_matrix(f, params_S, Zs)
    W = kernel_solve(Kss, labels, ridge)
    return lift_to_weights(g, flat0, Zs, W), W


def refine_delta(
    f,
    params_S,
    delta0,
    Z_obs,
    Y_obs,
    steps=200,
    lr=1e-3,
    l2=1e-6,
    centered=True,
    linear=False,
):
    """Step 4, Eq. 11: gradient descent on delta_theta alone, theta_S frozen.

    linear=True keeps the tangent model of Eq. 13 instead, which is the paper's
    "theta_i tangent-only" baseline in Tables II and III.
    """
    g, flat0, _ = flat_field(f, params_S)
    predict = tangent_predict if linear else nonlinear_predict

    def loss_fn(delta):
        pred = predict(g, flat0, delta, Z_obs, centered)
        return jnp.mean((pred - Y_obs) ** 2) + l2 * jnp.sum(delta**2)

    opt = optax.adam(lr)
    opt_state = opt.init(delta0)

    @jax.jit
    def step(delta, opt_state):
        loss, grad = jax.value_and_grad(loss_fn)(delta)
        updates, opt_state = opt.update(grad, opt_state, delta)
        return optax.apply_updates(delta, updates), opt_state, loss

    delta = delta0
    history = []
    for _ in range(steps):
        delta, opt_state, loss = step(delta, opt_state)
        history.append(float(loss))
    return delta, jnp.asarray(history)


def metaquill_kip_adapt(
    f,
    params_S,
    coords_obs,
    Y_obs,
    support,
    labels,
    encode_fn,
    space="coord",
    ridge=1e-4,
    kip_steps=150,
    kip_lr=1e-2,
    refine_steps=200,
    refine_lr=1e-3,
    l2=1e-6,
    centered=True,
    learn_labels=True,
):
    """Run steps 2 through 4 on one task and return every intermediate.

    coords_obs are raw coordinates; they are encoded here so that the KIP support
    can be optimized in the same space. The returned dict carries the distilled
    support, the Eq. 10 warm start, the refined residual, and both loss traces, so
    a caller can score KIP(init) and Combined(KIP->nonlin) separately the way
    Table II does.
    """
    Z_obs = encode_fn(jnp.asarray(coords_obs, dtype=jnp.float64))
    Y_obs = jnp.asarray(Y_obs, dtype=jnp.float64)

    support, labels, kip_hist = ntk_kip(
        f,
        params_S,
        Z_obs,
        Y_obs,
        support,
        labels,
        ridge=ridge,
        encode_fn=encode_fn,
        space=space,
        steps=kip_steps,
        lr=kip_lr,
        learn_labels=learn_labels,
    )

    delta_kip, W = kip_warm_start(f, params_S, support, labels, ridge, encode_fn, space)
    delta, refine_hist = refine_delta(
        f,
        params_S,
        delta_kip,
        Z_obs,
        Y_obs,
        steps=refine_steps,
        lr=refine_lr,
        l2=l2,
        centered=centered,
    )
    return {
        "support": support,
        "labels": labels,
        "W": W,
        "delta_kip": delta_kip,
        "delta": delta,
        "kip_history": kip_hist,
        "refine_history": refine_hist,
    }


def predict(f, params_S, delta, Z_query, centered=True, linear=False):
    """Reconstruct at query coordinates from a residual."""
    g, flat0, _ = flat_field(f, params_S)
    fn = tangent_predict if linear else nonlinear_predict
    return fn(g, flat0, delta, Z_query, centered)
