"""Occupancy maps, the bicausal OT linear program, and the envelope gradient.

The paper's inner solver is SOMCOT. On an enumerated state space the same
saddle point is an ordinary linear program, solved here with
scipy.optimize.linprog. Imitation learning from observations is not
implemented.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import optax
from scipy.optimize import linprog

jax.config.update("jax_enable_x64", True)


def _check_gamma(gamma) -> float:
    if isinstance(gamma, jax.core.Tracer):
        raise ValueError("gamma must be a concrete value in (0, 1)")
    try:
        value = float(gamma)
    except (TypeError, ValueError) as exc:
        raise ValueError("gamma must be a concrete value in (0, 1)") from exc
    if not (0.0 < value < 1.0):
        raise ValueError(f"gamma must be in (0, 1), got {value}")
    return value


def _as_numpy(array):
    if isinstance(array, jax.core.Tracer):
        return None
    try:
        return np.asarray(array)
    except (TypeError, ValueError, jax.errors.TracerArrayConversionError):
        return None


def _validate_kernel(kernel) -> None:
    """Reject a concrete kernel whose rows are not probabilities."""
    array = _as_numpy(kernel)
    if array is None:
        return
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("kernel must be a square matrix")
    if np.any(array < -1e-8):
        raise ValueError("kernel rows must be probabilities")
    row_sums = array.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-5):
        raise ValueError("kernel rows must be probabilities")


def softmax_kernel(logits):
    """Row-stochastic kernel, Eq. 16.

    P(y'|y) = exp(W[y, y']) / sum_{y''} exp(W[y, y'']).
    """
    logits = jnp.asarray(logits)
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    return weights / jnp.sum(weights, axis=-1, keepdims=True)


def occupancy(kernel, nu0, gamma):
    """Discounted transition occupancy nu(y, y'), Eq. 9 and Eq. 10.

    Solves (I - gamma P^T) rho = (1 - gamma) nu0, then
    nu(y, y') = rho(y) * P(y'|y). The returned matrix sums to 1 when
    nu0 is a probability vector and P is row-stochastic.
    """
    gamma = _check_gamma(gamma)
    kernel = jnp.asarray(kernel)
    if not jnp.issubdtype(kernel.dtype, jnp.floating):
        kernel = kernel.astype(jnp.float64)
    _validate_kernel(kernel)
    nu0 = jnp.asarray(nu0, dtype=kernel.dtype)
    eye = jnp.eye(kernel.shape[0], dtype=kernel.dtype)
    state = jax.scipy.linalg.solve(eye - gamma * kernel.T, (1.0 - gamma) * nu0)
    return state[:, None] * kernel


def _indexers(n_x, n_y):
    n_mu = n_x * n_y * n_x * n_y
    n_lx = n_x * n_y

    def mu_index(x, y, xp, yp):
        return ((x * n_y + y) * n_x + xp) * n_y + yp

    def lx_index(x, y):
        return n_mu + x * n_y + y

    def ly_index(y, x):
        return n_mu + n_lx + y * n_x + x

    return n_mu + n_lx + n_y * n_x, mu_index, lx_index, ly_index


def _failed_lp(n_x, n_y, message):
    return {
        "distance": float("nan"),
        "lam_y": jnp.zeros((n_y, n_x)),
        "alpha_y": jnp.zeros((n_x, n_y, n_y)),
        "success": False,
        "message": message,
    }


def bisimulation_lp(nu_x, nu_y, cost, gamma, x0, y0):
    """Bisimulation distance as a bicausal occupancy LP (section 2.3).

    Variables are mu(x, y, x', y') >= 0, lambda_X(y|x) in the simplex, and
    lambda_Y(x|y) in the simplex. nu_x and nu_y are data, so the products
    with lambda are linear.

    Returns distance, lam_y with shape (n_y, n_x) equal to lambda_Y(x|y),
    alpha_y with shape (n_x, n_y, n_y) equal to the causal-Y multiplier in
    the paper's Lagrangian, and success. On solver failure success is False
    and distance is nan.
    """
    gamma = _check_gamma(gamma)
    nu_x = np.asarray(nu_x, dtype=np.float64)
    nu_y = np.asarray(nu_y, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    if nu_x.ndim != 2 or nu_x.shape[0] != nu_x.shape[1]:
        raise ValueError("nu_x must be square")
    if nu_y.ndim != 2 or nu_y.shape[0] != nu_y.shape[1]:
        raise ValueError("nu_y must be square")
    n_x = int(nu_x.shape[0])
    n_y = int(nu_y.shape[0])
    if cost.shape != (n_x, n_y):
        raise ValueError(f"cost must have shape {(n_x, n_y)}, got {tuple(cost.shape)}")
    x0 = int(x0)
    y0 = int(y0)
    if not 0 <= x0 < n_x:
        raise ValueError(f"x0 must be in [0, {n_x}), got {x0}")
    if not 0 <= y0 < n_y:
        raise ValueError(f"y0 must be in [0, {n_y}), got {y0}")

    n_vars, mu_index, lx_index, ly_index = _indexers(n_x, n_y)
    objective = np.zeros(n_vars, dtype=np.float64)
    for x in range(n_x):
        for y in range(n_y):
            cij = float(cost[x, y])
            for xp in range(n_x):
                for yp in range(n_y):
                    objective[mu_index(x, y, xp, yp)] = cij

    rows = []
    rhs = []

    # Flow: outflow - gamma * inflow = (1 - gamma) * 1[x = x0] * 1[y = y0].
    for x in range(n_x):
        for y in range(n_y):
            row = np.zeros(n_vars, dtype=np.float64)
            for xp in range(n_x):
                for yp in range(n_y):
                    row[mu_index(x, y, xp, yp)] += 1.0
            for xh in range(n_x):
                for yh in range(n_y):
                    row[mu_index(xh, yh, x, y)] -= gamma
            rows.append(row)
            rhs.append((1.0 - gamma) if (x == x0 and y == y0) else 0.0)

    # Causal-X: sum_{y'} mu = nu_X(x, x') * lambda_X(y|x).
    for x in range(n_x):
        for y in range(n_y):
            for xp in range(n_x):
                row = np.zeros(n_vars, dtype=np.float64)
                for yp in range(n_y):
                    row[mu_index(x, y, xp, yp)] = 1.0
                row[lx_index(x, y)] -= nu_x[x, xp]
                rows.append(row)
                rhs.append(0.0)

    causal_y_start = len(rows)
    # Causal-Y: sum_{x'} mu = nu_Y(y, y') * lambda_Y(x|y).
    for x in range(n_x):
        for y in range(n_y):
            for yp in range(n_y):
                row = np.zeros(n_vars, dtype=np.float64)
                for xp in range(n_x):
                    row[mu_index(x, y, xp, yp)] = 1.0
                row[ly_index(y, x)] -= nu_y[y, yp]
                rows.append(row)
                rhs.append(0.0)
    n_causal_y = n_x * n_y * n_y

    for x in range(n_x):
        row = np.zeros(n_vars, dtype=np.float64)
        for y in range(n_y):
            row[lx_index(x, y)] = 1.0
        rows.append(row)
        rhs.append(1.0)
    for y in range(n_y):
        row = np.zeros(n_vars, dtype=np.float64)
        for x in range(n_x):
            row[ly_index(y, x)] = 1.0
        rows.append(row)
        rhs.append(1.0)

    result = linprog(
        objective,
        A_eq=np.vstack(rows),
        b_eq=np.asarray(rhs, dtype=np.float64),
        bounds=(0, None),
        method="highs",
    )
    if not result.success or result.x is None:
        return _failed_lp(n_x, n_y, str(result.message))
    marginals = getattr(result.eqlin, "marginals", None)
    if marginals is None or len(marginals) < causal_y_start + n_causal_y:
        return _failed_lp(n_x, n_y, "linprog returned no equality marginals")

    lam_y = np.zeros((n_y, n_x), dtype=np.float64)
    for y in range(n_y):
        for x in range(n_x):
            lam_y[y, x] = result.x[ly_index(y, x)]

    # scipy linprog eqlin.marginals are partial derivatives of the optimal
    # value with respect to b_eq. The paper's alpha_Y multiplies the residual
    # (sum_{x'} mu - nu_Y lambda_Y) in the Lagrangian (Eq. 3), so it is the
    # negative of that marginal. The linprog marginal sign was aligned to
    # Eq. 7 by this negation, and checked against central finite differences.
    alpha_y = np.zeros((n_x, n_y, n_y), dtype=np.float64)
    cursor = 0
    for x in range(n_x):
        for y in range(n_y):
            for yp in range(n_y):
                alpha_y[x, y, yp] = -float(marginals[causal_y_start + cursor])
                cursor += 1

    return {
        "distance": float(result.fun),
        "lam_y": jnp.asarray(lam_y),
        "alpha_y": jnp.asarray(alpha_y),
        "success": True,
    }


def envelope_grad(logits, lam_y, alpha_y, nu0, gamma):
    """Envelope gradient of the bisimulation distance in the logits, Eq. 7.

    lambda_Y and alpha_Y are stop-gradient constants (Remark 1). The
    occupancy map is differentiated through jax.scipy.linalg.solve, not
    through the linear program.
    """
    _check_gamma(gamma)
    lam_y = jax.lax.stop_gradient(jnp.asarray(lam_y))
    alpha_y = jax.lax.stop_gradient(jnp.asarray(alpha_y))
    nu0 = jnp.asarray(nu0)

    def surrogate(weights):
        transition = occupancy(softmax_kernel(weights), nu0, gamma)
        # sum_{x,y,y'} lambda_Y(x|y) * alpha_Y(x, y, y') * nu(y, y')
        weighted = jnp.einsum("yx,xyk,yk->", lam_y, alpha_y, transition)
        return -weighted

    return jax.grad(surrogate)(jnp.asarray(logits))


def fit_kernel(target_kernel, *, cost, gamma, x0, y0, steps, lr, key):
    """Fit a softmax kernel by alternating an exact LP and a gradient step.

    Model learning case: the model state space has the same size as the
    target. Logits start at zero. key is part of the public signature and
    is not used for initialization. Raises if the first LP fails.
    """
    del key
    gamma = _check_gamma(gamma)
    target_kernel = jnp.asarray(target_kernel)
    if target_kernel.ndim != 2 or target_kernel.shape[0] != target_kernel.shape[1]:
        raise ValueError("target_kernel must be square")
    n = int(target_kernel.shape[0])
    x0 = int(x0)
    y0 = int(y0)
    if not 0 <= x0 < n or not 0 <= y0 < n:
        raise ValueError(f"x0 and y0 must be in [0, {n})")
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be non-negative")

    nu0_x = jnp.zeros(n).at[x0].set(1.0)
    nu0_y = jnp.zeros(n).at[y0].set(1.0)
    nu_x = occupancy(target_kernel, nu0_x, gamma)
    logits = jnp.zeros((n, n))
    optimizer = optax.sgd(learning_rate=lr)
    opt_state = optimizer.init(logits)
    trace = []
    solved = None

    for step in range(steps + 1):
        if solved is None:
            nu_y = occupancy(softmax_kernel(logits), nu0_y, gamma)
            solved = bisimulation_lp(nu_x, nu_y, cost, gamma, x0, y0)
        if not solved["success"]:
            if step == 0:
                raise RuntimeError("bisimulation LP failed on the initial kernel")
            raise RuntimeError(f"bisimulation LP failed at step {step}")
        trace.append(float(solved["distance"]))
        if step == steps:
            break
        grad = envelope_grad(logits, solved["lam_y"], solved["alpha_y"], nu0_y, gamma)
        updates, opt_state = optimizer.update(grad, opt_state)
        logits = optax.apply_updates(logits, updates)
        solved = None

    return {"distance_trace": trace, "logits": logits}
