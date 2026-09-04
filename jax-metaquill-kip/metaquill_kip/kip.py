"""NTK-KIP: kernel inducing points through the finite NTK (Sec. III-A).

The idea is to stop solving the kernel system on every observed pixel and instead
learn a compact support set S_T = {(x~_m, y~_m)}, Eq. 8, whose NTK regression
solve already reproduces the signal. Since the support coordinates are optimized
against the labels, the kernel itself becomes label-dependent and the coefficient
vector W stops being a linear map of Y, which is the whole point of Sec. II-A's
"representational linearity" complaint.

The objective is Eq. 9:

    L_KIP = || K_theta(X, X~) (K_theta(X~, X~) + lambda I)^{-1} Y~ - Y ||^2

with X, Y the observed pixels only when inpainting. theta stays frozen; the
gradient flows through the solve into x~ and y~.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from metaquill_kip.ntk import kernel_predict, kernel_solve, ntk_matrix


def support_features(support, encode_fn, space):
    """Encoded support features X~.

    space="coord" optimizes the raw 2D coordinates and encodes them, which is the
    paper's Raw PE / FPE-over-coordinates variant. space="feature" optimizes the
    encoded vector directly, the FPE-KIP variant of Sec. IV-A where the learned
    support no longer corresponds to any 2D position.
    """
    if space == "coord":
        return encode_fn(support)
    if space == "feature":
        return support
    raise ValueError(f"unknown support space {space!r}")


def kip_loss(f, params, support, labels, Z_obs, Y_obs, ridge, encode_fn, space="coord"):
    """Eq. 9, as a mean squared error over the observed pixels.

    The paper writes a squared norm; using the mean only rescales the loss and
    keeps the number readable next to an MSE.
    """
    Zs = support_features(support, encode_fn, space)
    Kss = ntk_matrix(f, params, Zs)
    Kos = ntk_matrix(f, params, Z_obs, Zs)
    W = kernel_solve(Kss, labels, ridge)
    pred = kernel_predict(Kos, W)
    return jnp.mean((pred - Y_obs) ** 2)


def kip_predict(f, params, support, labels, Z_query, ridge, encode_fn, space="coord"):
    """Reconstruct at query coordinates from the distilled support alone.

    This is Eq. 5 with the observed set replaced by S_T, and it is the predictor
    the paper calls KIP(init) in Tables II and III.
    """
    Zs = support_features(support, encode_fn, space)
    Kss = ntk_matrix(f, params, Zs)
    Kqs = ntk_matrix(f, params, Z_query, Zs)
    W = kernel_solve(Kss, labels, ridge)
    return kernel_predict(Kqs, W), W


def init_support(coords_obs, values_obs, m, seed, space, encode_fn):
    """Start the support at m of the observed pixels with their true values.

    Random observed pixels are the baseline NTK-KIP has to beat, so they are also
    the honest starting point for the optimization.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(coords_obs.shape[0], size=m, replace=False)
    support = jnp.asarray(coords_obs[idx], dtype=jnp.float64)
    if space == "feature":
        support = encode_fn(support)
    labels = jnp.asarray(values_obs[idx], dtype=jnp.float64)
    return support, labels


def ntk_kip(
    f,
    params,
    Z_obs,
    Y_obs,
    support,
    labels,
    ridge=1e-4,
    encode_fn=None,
    space="coord",
    steps=200,
    lr=1e-2,
    learn_labels=True,
):
    """Optimize S_T by gradient descent on Eq. 9. theta is frozen throughout.

    The paper enables label learning in both of its variants ("In both experiments
    we enable labels learning (yields better results)", Appendix E-A1), so
    learn_labels defaults to True. Returns the optimized support, labels and the
    loss trace.
    """
    if encode_fn is None:
        raise ValueError("ntk_kip needs the encoding used for the observed pixels")

    def loss_fn(state):
        lab = state["labels"] if learn_labels else jax.lax.stop_gradient(state["labels"])
        return kip_loss(f, params, state["support"], lab, Z_obs, Y_obs, ridge, encode_fn, space)

    opt = optax.adam(lr)
    state = {"support": support, "labels": labels}
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
    return state["support"], state["labels"], jnp.asarray(history)
