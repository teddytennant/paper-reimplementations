"""Empirical finite NTK and the closed-form kernel regression fit.

Symbols follow the paper: K is the finite-width empirical tangent kernel of Eq. 1,
W is the coefficient vector of Eq. 2 (Eq. 4 in the matrix-completion form), and
the prediction at query coordinates is Eq. 3 (Eq. 5 for matrix completion).

Only scalar-output fields are supported here. The paper's C-channel argument is
that the per-class kernels are identical and can be treated as C independent
copies of one kernel; a scalar field makes that assumption unnecessary rather
than approximate.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def flatten_field(f, params):
    """Rewrite f(params, Z) as g(flat_theta, Z) with a flat parameter vector.

    The NTK is an inner product over all Q parameters (Eq. 1), so it is easier to
    write against a flat theta than against the pytree.
    """
    flat, unravel = ravel_pytree(params)

    def g(flat_theta, Z):
        out = f(unravel(flat_theta), Z)
        if out.ndim == 2 and out.shape[-1] == 1:
            out = out[:, 0]
        if out.ndim != 1:
            raise ValueError("the NTK code here expects a scalar-output field")
        return out

    return g, flat, unravel


def jacobian_matrix(f, params, Z):
    """The tangent feature map phi(x) = grad_theta f(x; theta), stacked.

    Returns (n, Q). This is the explicit form of the gradients whose inner
    product defines K in Eq. 1. It costs O(n * Q) memory, which is fine at the
    sizes used here but is exactly what `ntk_matrix_jvp` avoids.
    """
    g, flat, _ = flatten_field(f, params)
    return jax.jacrev(g)(flat, Z)


def ntk_matrix(f, params, Z1, Z2=None):
    """K(x, x'; theta) = grad_theta f(x)^T grad_theta f(x'), Eq. 1.

    Returns (n1, n2). Built from explicit Jacobians, so K is symmetric to
    machine precision when Z2 is None.
    """
    J1 = jacobian_matrix(f, params, Z1)
    if Z2 is None:
        K = J1 @ J1.T
        return 0.5 * (K + K.T)
    J2 = jacobian_matrix(f, params, Z2)
    return J1 @ J2.T


def ntk_matrix_jvp(f, params, Z1, Z2=None):
    """Same kernel as `ntk_matrix`, Eq. 1, without materializing both Jacobians.

    Column j is J1 u_j where u_j = grad_theta f(x'_j) comes from a vjp against the
    j-th basis cotangent, so only one Jacobian's worth of work is in flight at a
    time. Use this when n is large enough that two (n, Q) blocks do not fit.
    """
    g, flat, _ = flatten_field(f, params)
    target = Z1 if Z2 is None else Z2
    n2 = target.shape[0]
    _, vjp_fun = jax.vjp(lambda t: g(t, target), flat)

    def column(e):
        (u,) = vjp_fun(e)  # u = grad_theta f(x'_j)
        _, jv = jax.jvp(lambda t: g(t, Z1), (flat,), (u,))
        return jv

    KT = jax.vmap(column)(jnp.eye(n2, dtype=flat.dtype))
    K = KT.T
    if Z2 is None:
        K = 0.5 * (K + K.T)
    return K


@partial(jax.jit, static_argnames=("method",))
def kernel_solve(K, Y, ridge=0.0, method="cholesky"):
    """W = (K + lambda I)^{-1} Y, Eq. 2 and Eq. 4 with the ridge term of Eq. 9.

    lambda = 0 recovers the paper's literal W = K^{-1} Y. The paper reports that
    the finite NTK of a trained shared field is ill-conditioned (Sec. IV-B), so
    an explicit ridge is the difference between a usable solve and numerical
    noise.

    method="cholesky" needs K + lambda I positive definite, which holds for any
    lambda > 0 since K is PSD. method="solve" is the general LU path, and
    method="lstsq" is the minimum-norm solution for a singular K at lambda = 0.
    """
    K = jnp.asarray(K)
    Y = jnp.asarray(Y)
    A = K + ridge * jnp.eye(K.shape[0], dtype=K.dtype)
    if method == "cholesky":
        L = jnp.linalg.cholesky(A)
        return jax.scipy.linalg.cho_solve((L, True), Y)
    if method == "solve":
        return jnp.linalg.solve(A, Y)
    if method == "lstsq":
        return jnp.linalg.lstsq(A, Y)[0]
    raise ValueError(f"unknown solve method {method!r}")


def kernel_predict(K_test, W):
    """Yhat = K(x', x) W, Eq. 3 (Eq. 5 in the matrix-completion form)."""
    return K_test @ W


def ntk_regression(
    f,
    params,
    Z_train,
    Y_train,
    Z_query,
    ridge=1e-6,
    method="cholesky",
    kernel_fn=ntk_matrix,
):
    """The whole closed-form fit: Eq. 4 then Eq. 5.

    Z_train are the encoded observed coordinates, Y_train their values, Z_query
    the coordinates to reconstruct (typically the full h*w grid).
    """
    K_train = kernel_fn(f, params, Z_train)
    K_test = kernel_fn(f, params, Z_query, Z_train)
    W = kernel_solve(K_train, Y_train, ridge, method)
    return kernel_predict(K_test, W), W
