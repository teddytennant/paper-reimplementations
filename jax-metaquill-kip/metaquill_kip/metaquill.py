"""MetaQuill: meta-learning a shared neural field through the finite NTK (Sec. III-B).

The model for task i is the centered tangent model of Eq. 12,

    NF_i(x) = delta_theta_i^T grad_{theta_S} f_{theta_S}(x)                 (Eq. 13)

so theta_S supplies the feature map and delta_theta_i is a linear weight on top of
it. Meta-training optimizes both at once, Eq. 14,

    L_i = sum_p 0.5 (delta_theta_i^T grad f(x^p) - y_i^p)^2 + ||delta_theta_i||^2

with no MAML inner loop: the per-task residuals are free variables trained
alongside theta_S rather than produced by an unrolled optimizer.

Adapting to a new task is closed form. The optimal residual is a linear
combination of tangent features at the observed coordinates, Eq. 15,

    delta_theta_i = sum_p w_i^p grad_{theta_S} f_{theta_S}(x^p),  w_i = K_{theta_S}^{-1} y_i

which is a single vjp against the kernel coefficients. The paper writes the solve
without a ridge; we keep lambda explicit because Sec. IV-B reports that exactly
this kernel matrix is ill-conditioned.
"""

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from metaquill_kip.ntk import kernel_solve, ntk_matrix


def flat_field(f, params0):
    """(g, flat0, unravel) with g(flat_theta, Z) -> (n,) scalar outputs."""
    flat0, unravel = ravel_pytree(params0)

    def g(flat_theta, Z):
        out = f(unravel(flat_theta), Z)
        return out[:, 0] if out.ndim == 2 and out.shape[-1] == 1 else out

    return g, flat0, unravel


def tangent_predict(g, flat_theta, delta, Z, centered=True):
    """The linearized prediction around theta_S.

    centered=True is Eq. 12/13, J_{theta_S}(x) delta_theta with the zeroth-order
    term dropped. centered=False adds f_{theta_S}(x) back, i.e. the plain
    first-order Taylor expansion the paper writes just above Eq. 12.
    """
    base, tangent = jax.jvp(lambda t: g(t, Z), (flat_theta,), (delta,))
    return tangent if centered else base + tangent


def nonlinear_predict(g, flat_theta, delta, Z, centered=True):
    """The full non-linear output used in the refinement of Eq. 11.

    Eq. 11 writes f_{theta_S + delta_theta}(x). Under the centered convention of
    Eq. 12 the zeroth-order term is not part of the model, so we subtract
    f_{theta_S}(x); the two agree to first order in delta_theta, and subtracting
    is what makes Eq. 10 the correct warm start for Eq. 11. Pass centered=False
    for the paper's literal form.
    """
    out = g(flat_theta + delta, Z)
    return out - g(flat_theta, Z) if centered else out


def kernel_coefficients(f, params, Z_obs, Y_obs, ridge=1e-6, method="cholesky"):
    """w_i = (K_{theta_S} + lambda I)^{-1} y_i, the kernel-space representation.

    This is Eq. 2 evaluated at the meta-learned theta_S, and it is the vector the
    paper feeds to a diffusion model in the Functa pipeline (Algorithm 1).
    """
    K = ntk_matrix(f, params, Z_obs)
    return kernel_solve(K, Y_obs, ridge, method)


def lift_to_weights(g, flat_theta, Z_obs, w):
    """Eq. 15: delta_theta = sum_p w^p grad_{theta_S} f_{theta_S}(x^p) = J^T w.

    One vjp, no Jacobian. This is the kernel-space to weight-space transform in
    Algorithm 1.
    """
    _, vjp_fun = jax.vjp(lambda t: g(t, Z_obs), flat_theta)
    (delta,) = vjp_fun(w)
    return delta


def metaquill_adapt(f, params_S, Z_obs, Y_obs, ridge=1e-6, method="cholesky"):
    """Closed-form per-task adaptation: Eq. 2 then Eq. 15.

    Returns (delta_theta, w). No gradient steps are taken; the paper's claim is
    that a new task is adapted "straight from the first gradient update", and the
    ridge solve is the zero-step version of that.
    """
    g, flat0, _ = flat_field(f, params_S)
    w = kernel_coefficients(f, params_S, Z_obs, Y_obs, ridge, method)
    return lift_to_weights(g, flat0, Z_obs, w), w


def meta_train(
    f,
    params0,
    Z,
    Y_tasks,
    steps=300,
    lr=3e-3,
    l2=1e-3,
    obs_idx=None,
):
    """Learn theta_S and the per-task residuals jointly, Eq. 14 / Algorithm 1.

    Z is the shared encoded coordinate set (the paper assumes every image is
    sampled at the same points, Appendix E-A2a, which is what lets K_{theta_S} be
    computed once). Y_tasks is (n_tasks, n_points). obs_idx optionally restricts
    the loss to a subset of coordinates per task.

    l2 is the coefficient on the ||delta_theta_i||^2 penalty of Eq. 14, which the
    paper writes with an implicit coefficient of one; it pushes work onto the
    shared feature map instead of the residuals.
    """
    g, flat0, unravel = flat_field(f, params0)
    Y_tasks = jnp.asarray(Y_tasks, dtype=jnp.float64)
    n_tasks = Y_tasks.shape[0]
    idx = jnp.arange(Z.shape[0]) if obs_idx is None else jnp.asarray(obs_idx)
    Z_obs = Z[idx]
    Y_obs = Y_tasks[:, idx]

    state = {
        "theta": flat0,
        "delta": jnp.zeros((n_tasks, flat0.shape[0]), dtype=jnp.float64),
    }

    def loss_fn(state):
        def per_task(delta, y):
            pred = tangent_predict(g, state["theta"], delta, Z_obs)
            return 0.5 * jnp.mean((pred - y) ** 2) + l2 * jnp.sum(delta**2)

        return jnp.mean(jax.vmap(per_task)(state["delta"], Y_obs))

    opt = optax.adam(lr)
    opt_state = opt.init(state)

    @jax.jit
    def step(state, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(state)
        updates, opt_state = opt.update(grads, opt_state, state)
        return optax.apply_updates(state, updates), opt_state, loss

    history = []
    for _ in range(steps):
        state, opt_state, loss = step(state, opt_state)
        history.append(float(loss))
    return unravel(state["theta"]), state["delta"], jnp.asarray(history)
