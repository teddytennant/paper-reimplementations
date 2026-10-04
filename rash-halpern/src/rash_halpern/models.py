"""Appendix B models and analytic worst-case rows.

State order follows the paper. Action 0 is the conservative / curved / safe
action. Action 1 is the risky alternative at the decision state only.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

DTYPE = jnp.float64


def _unit(n: int, index: int) -> jnp.ndarray:
    row = jnp.zeros(n, dtype=DTYPE)
    return row.at[index].set(1.0)


def _as_f64(v) -> jnp.ndarray:
    return jnp.asarray(v, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Model I, boundary leakage. States (x, y, z, w, H).
# ---------------------------------------------------------------------------

def model_i_c_objective(u, v) -> float:
    """p(u)^T v for the curved row at (x, c)."""
    v = _as_f64(v)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    u = float(u)
    return vx + u * (vy - vx) + (u ** 3) * (vz - vx)


def model_i_c_derivative(u, v) -> float:
    """Derivative of p(u)^T v in the curved parameter u."""
    v = _as_f64(v)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    u = float(u)
    return (vy - vx) + 3.0 * (u ** 2) * (vz - vx)


def model_i_c_row(u) -> jnp.ndarray:
    u = float(u)
    u3 = u ** 3
    return jnp.array([1.0 - u - u3, u, u3, 0.0, 0.0], dtype=DTYPE)


def worst_case_u_model_i(v) -> float:
    """Minimizer of p(u)^T v on [0, 1/2].

    Compare endpoints with every root of the cubic derivative in (0, 1/2).
    A root that is a maximum does not win. Ties break toward the smaller u.
    """
    v = _as_f64(v)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    linear = vy - vx
    cubic = vz - vx
    candidates = [0.0, 0.5]
    # phi'(u) = linear + 3 * cubic * u^2. At most one positive root.
    if cubic != 0.0:
        u2 = -linear / (3.0 * cubic)
        if u2 > 0.0:
            u_root = math.sqrt(u2)
            if 0.0 < u_root < 0.5:
                candidates.append(u_root)

    def phi(u: float) -> float:
        return vx + u * linear + (u ** 3) * cubic

    best = min(candidates, key=lambda u: (phi(u), u))
    return best


def worst_case_rho_model_i(v) -> float:
    """Worst-case rho for (x, t).

    Row is (0, 0, 0, 1 - rho, rho). Minimizing continuation puts mass on the
    smaller of v_w and v_H, so rho = 0 when v_w <= v_H and rho = 1 otherwise.
    """
    v = _as_f64(v)
    v_w = float(v[3])
    v_h = float(v[4])
    if v_w <= v_h:
        return 0.0
    return 1.0


def model_i_t_row(rho) -> jnp.ndarray:
    rho = float(rho)
    return jnp.array([0.0, 0.0, 0.0, 1.0 - rho, rho], dtype=DTYPE)


# ---------------------------------------------------------------------------
# Model II, periodic basins. States (x, y, h0, h1, m, ell).
# ---------------------------------------------------------------------------

def model_ii_row(u) -> jnp.ndarray:
    """Curved row p(u) at (x, c). Sums to 1 for every u."""
    u = float(u)
    u2 = u * u
    return jnp.array(
        [
            0.0,
            0.0,
            0.55 + 0.10 * u,
            0.0,
            0.25 - 0.05 * u2,
            0.20 - 0.10 * u + 0.05 * u2,
        ],
        dtype=DTYPE,
    )


def model_ii_c_objective(u, v) -> float:
    row = model_ii_row(u)
    return float(jnp.dot(row, _as_f64(v)))


def _model_ii_quadratic_coeffs(v) -> tuple[float, float, float]:
    """phi(u) = A + B u + C u^2 for p(u)^T v."""
    v = _as_f64(v)
    v_h0 = float(v[2])
    v_m = float(v[4])
    v_ell = float(v[5])
    A = 0.55 * v_h0 + 0.25 * v_m + 0.20 * v_ell
    B = 0.10 * (v_h0 - v_ell)
    C = 0.05 * (v_ell - v_m)
    return A, B, C


def worst_case_u_model_ii(v) -> float:
    """Minimizer of p(u)^T v on [0, 1].

    Compare endpoints with the critical point when it lies in (0, 1).
    A critical point of a concave objective is a maximum and does not win.
    Ties break toward the smaller u.
    """
    A, B, C = _model_ii_quadratic_coeffs(v)
    candidates = [0.0, 1.0]
    if C != 0.0:
        u_star = -B / (2.0 * C)
        # Include only a minimizing critical point (C > 0). Comparing values
        # would drop a maximum anyway; the paper asks for that check.
        if C > 0.0 and 0.0 < u_star < 1.0:
            candidates.append(u_star)

    def phi(u: float) -> float:
        return A + B * u + C * (u ** 2)

    return min(candidates, key=lambda u: (phi(u), u))


def worst_case_rho_model_ii(v) -> float:
    """Worst-case rho for (x, f). rho = 0 if v_m <= v_ell else 1."""
    v = _as_f64(v)
    v_m = float(v[4])
    v_ell = float(v[5])
    if v_m <= v_ell:
        return 0.0
    return 1.0


def model_ii_f_row(rho) -> jnp.ndarray:
    rho = float(rho)
    return jnp.array([0.0, 0.0, 0.0, 0.0, 1.0 - rho, rho], dtype=DTYPE)


# ---------------------------------------------------------------------------
# Model III, safe-risky. States (x, L, H).
# ---------------------------------------------------------------------------

MODEL_III_SAFE_ROW = jnp.array([0.0, 0.6, 0.4], dtype=DTYPE)


def worst_case_p_model_iii(v) -> float:
    """Worst-case risky probability of H.

    Row is (0, 1 - p, p) with p in [0.1, 0.9]. The minimum is at p = 0.1
    when v_H >= v_L, and at p = 0.9 otherwise.
    """
    v = _as_f64(v)
    v_l = float(v[1])
    v_h = float(v[2])
    if v_h >= v_l:
        return 0.1
    return 0.9


def model_iii_risky_row(p) -> jnp.ndarray:
    p = float(p)
    return jnp.array([0.0, 1.0 - p, p], dtype=DTYPE)


# ---------------------------------------------------------------------------
# Gain-face helpers used by the certificate operator.
# A flat objective means every parameter is gain-minimizing, so the bias
# minimum is taken over the whole parameter interval.
# ---------------------------------------------------------------------------

def _minimizing_us_cubic(v, lo: float, hi: float) -> list[float] | None:
    """Minimizing parameters of the Model I cubic, or None if it is flat."""
    v = _as_f64(v)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    linear = vy - vx
    cubic = vz - vx
    if linear == 0.0 and cubic == 0.0:
        return None
    candidates = [lo, hi]
    if cubic != 0.0:
        u2 = -linear / (3.0 * cubic)
        if u2 > 0.0:
            u_root = math.sqrt(u2)
            if lo < u_root < hi:
                candidates.append(u_root)

    def phi(u: float) -> float:
        return vx + u * linear + (u ** 3) * cubic

    best = min(phi(u) for u in candidates)
    tol = 1e-12 * (1.0 + abs(best))
    return [u for u in candidates if phi(u) <= best + tol]


def _minimizing_us_quadratic(v, lo: float, hi: float) -> list[float] | None:
    A, B, C = _model_ii_quadratic_coeffs(v)
    del A
    if B == 0.0 and C == 0.0:
        return None
    candidates = [lo, hi]
    if C > 0.0:
        u_star = -B / (2.0 * C)
        if lo < u_star < hi:
            candidates.append(u_star)
    elif C < 0.0:
        # Critical point is a maximum. Endpoints are the only candidates.
        pass
    elif B == 0.0:
        return None

    def phi(u: float) -> float:
        return B * u + C * (u ** 2)

    best = min(phi(u) for u in candidates)
    tol = 1e-12 * (1.0 + abs(best))
    return [u for u in candidates if phi(u) <= best + tol]


def _min_objective_on_curve(v, kind: str) -> float:
    if kind == "i":
        us = _minimizing_us_cubic(v, 0.0, 0.5)
        if us is None:
            return model_i_c_objective(0.0, v)
        return min(model_i_c_objective(u, v) for u in us)
    if kind == "ii":
        us = _minimizing_us_quadratic(v, 0.0, 1.0)
        if us is None:
            return model_ii_c_objective(0.0, v)
        return min(model_ii_c_objective(u, v) for u in us)
    raise KeyError(kind)


def _face_min_objective(g, h, kind: str) -> float:
    """min p^T h over argmin p^T g, for a curved row set."""
    if kind == "i":
        us = _minimizing_us_cubic(g, 0.0, 0.5)
        if us is None:
            us = _minimizing_us_cubic(h, 0.0, 0.5)
            if us is None:
                return model_i_c_objective(0.0, h)
            return min(model_i_c_objective(u, h) for u in us)
        return min(model_i_c_objective(u, h) for u in us)
    if kind == "ii":
        us = _minimizing_us_quadratic(g, 0.0, 1.0)
        if us is None:
            us = _minimizing_us_quadratic(h, 0.0, 1.0)
            if us is None:
                return model_ii_c_objective(0.0, h)
            return min(model_ii_c_objective(u, h) for u in us)
        return min(model_ii_c_objective(u, h) for u in us)
    raise KeyError(kind)


@dataclass(frozen=True)
class Model:
    """Finite robust average-reward model with an analytic row oracle."""

    name: str
    kind: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    rewards: jnp.ndarray
    mask: jnp.ndarray
    g_star: jnp.ndarray
    h: jnp.ndarray
    prefer: tuple[int, ...]
    optimal_action_at_x: int

    @property
    def n_states(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.rewards.shape[1])

    def nominal_parameter(self, action: int) -> float:
        if self.kind == "i":
            return 0.25 if action == 0 else 0.5
        if self.kind == "ii":
            return 0.5
        if self.kind == "iii":
            return 0.5
        raise KeyError(self.kind)

    def row(self, state: int, action: int, v, nominal: bool) -> jnp.ndarray:
        """Transition row. Robust mode uses the analytic worst-case row."""
        v = _as_f64(v)
        n = self.n_states
        if self.kind == "i":
            if state == 0 and action == 0:
                u = 0.25 if nominal else worst_case_u_model_i(v)
                return model_i_c_row(u)
            if state == 0 and action == 1:
                rho = 0.5 if nominal else worst_case_rho_model_i(v)
                return model_i_t_row(rho)
            # y -> x, and z, w, H absorbing.
            target = {1: 0, 2: 2, 3: 3, 4: 4}[state]
            return _unit(n, target)
        if self.kind == "ii":
            if state == 0 and action == 0:
                u = 0.5 if nominal else worst_case_u_model_ii(v)
                return model_ii_row(u)
            if state == 0 and action == 1:
                rho = 0.5 if nominal else worst_case_rho_model_ii(v)
                return model_ii_f_row(rho)
            # y -> x, h0 -> h1, h1 -> h0, m and ell absorbing.
            target = {1: 0, 2: 3, 3: 2, 4: 4, 5: 5}[state]
            return _unit(n, target)
        if self.kind == "iii":
            if state == 0 and action == 0:
                return MODEL_III_SAFE_ROW
            if state == 0 and action == 1:
                p = 0.5 if nominal else worst_case_p_model_iii(v)
                return model_iii_risky_row(p)
            target = {1: 1, 2: 2}[state]
            return _unit(n, target)
        raise KeyError(self.kind)

    def continuation_gain(self, state: int, action: int, g) -> float:
        """m_ia(g) = min_{p in U_ia} p^T g."""
        if not bool(self.mask[state, action]):
            return -math.inf
        if state != 0:
            row = self.row(state, action, g, nominal=False)
            return float(jnp.dot(row, _as_f64(g)))
        if self.kind == "i" and action == 0:
            return _min_objective_on_curve(g, "i")
        if self.kind == "i" and action == 1:
            g = _as_f64(g)
            return min(float(g[3]), float(g[4]))
        if self.kind == "ii" and action == 0:
            return _min_objective_on_curve(g, "ii")
        if self.kind == "ii" and action == 1:
            g = _as_f64(g)
            return min(float(g[4]), float(g[5]))
        if self.kind == "iii" and action == 0:
            return float(jnp.dot(MODEL_III_SAFE_ROW, _as_f64(g)))
        if self.kind == "iii" and action == 1:
            g = _as_f64(g)
            v_l, v_h = float(g[1]), float(g[2])
            # p in [0.1, 0.9], objective v_L + p (v_H - v_L)
            if v_h >= v_l:
                p = 0.1
            else:
                p = 0.9
            return (1.0 - p) * v_l + p * v_h
        raise KeyError((self.kind, state, action))

    def face_bias_score(self, state: int, action: int, g, h) -> float:
        """r_ia + min_{p in F_ia(g)} p^T h."""
        reward = float(self.rewards[state, action])
        if state != 0:
            row = self.row(state, action, h, nominal=False)
            return reward + float(jnp.dot(row, _as_f64(h)))
        if self.kind == "i" and action == 0:
            return reward + _face_min_objective(g, h, "i")
        if self.kind == "i" and action == 1:
            return reward + _endpoint_face_bias(g, h, 3, 4)
        if self.kind == "ii" and action == 0:
            return reward + _face_min_objective(g, h, "ii")
        if self.kind == "ii" and action == 1:
            return reward + _endpoint_face_bias(g, h, 4, 5)
        if self.kind == "iii" and action == 0:
            return reward + float(jnp.dot(MODEL_III_SAFE_ROW, _as_f64(h)))
        if self.kind == "iii" and action == 1:
            return reward + _interval_face_bias(g, h, 1, 2, 0.1, 0.9)
        raise KeyError((self.kind, state, action))


def _endpoint_face_bias(g, h, i_low: int, i_high: int) -> float:
    """Affine row (mass on i_low or i_high). Face bias is min h on the gain face."""
    g = _as_f64(g)
    h = _as_f64(h)
    g_low, g_high = float(g[i_low]), float(g[i_high])
    h_low, h_high = float(h[i_low]), float(h[i_high])
    if g_low < g_high:
        return h_low
    if g_high < g_low:
        return h_high
    return min(h_low, h_high)


def _interval_face_bias(g, h, i_low: int, i_high: int, p_lo: float, p_hi: float) -> float:
    """Row (0, 1-p, p) with p in [p_lo, p_hi], states ordered so index 0 is unused."""
    g = _as_f64(g)
    h = _as_f64(h)
    g_low, g_high = float(g[i_low]), float(g[i_high])
    h_low, h_high = float(h[i_low]), float(h[i_high])
    if g_high > g_low:
        p = p_lo
    elif g_high < g_low:
        p = p_hi
    else:
        # Whole interval is the face. Bias is affine, so an endpoint wins.
        p = p_lo if h_high >= h_low else p_hi
    return (1.0 - p) * h_low + p * h_high


def _mask(n_states: int, n_actions: int) -> jnp.ndarray:
    mask = jnp.zeros((n_states, n_actions), dtype=bool)
    mask = mask.at[:, 0].set(True)
    mask = mask.at[0, 1].set(True)
    return mask


def _model_i() -> Model:
    rewards = jnp.array(
        [
            [0.0, 10.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-2.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=DTYPE,
    )
    g_star = jnp.array([0.0, 0.0, 1.0, -2.0, 5.0], dtype=DTYPE)
    h = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    return Model(
        name="boundary leakage",
        kind="i",
        state_names=("x", "y", "z", "w", "H"),
        action_names=("c", "t"),
        rewards=rewards,
        mask=_mask(5, 2),
        g_star=g_star,
        h=h,
        prefer=(0, 0, 0, 0, 0),
        optimal_action_at_x=0,
    )


def _model_ii() -> Model:
    rewards = jnp.array(
        [
            [0.0, 0.0],
            [-0.3, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=DTYPE,
    )
    g_star = jnp.array([-3.0 / 40.0, -3.0 / 40.0, 0.0, 0.0, -1.0, 1.0], dtype=DTYPE)
    h = jnp.array([27.0 / 40.0, 9.0 / 20.0, 1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    return Model(
        name="periodic basins",
        kind="ii",
        state_names=("x", "y", "h0", "h1", "m", "ell"),
        action_names=("c", "f"),
        rewards=rewards,
        mask=_mask(6, 2),
        g_star=g_star,
        h=h,
        prefer=(0, 0, 0, 0, 0, 0),
        optimal_action_at_x=0,
    )


def _model_iii() -> Model:
    rewards = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=DTYPE,
    )
    g_star = jnp.array([0.4, 0.0, 1.0], dtype=DTYPE)
    h = jnp.array([-0.4, 0.0, 0.0], dtype=DTYPE)
    # Tie-break at x favors risky (action 1). They do not tie at the optimum.
    return Model(
        name="safe-risky",
        kind="iii",
        state_names=("x", "L", "H"),
        action_names=("safe", "risky"),
        rewards=rewards,
        mask=_mask(3, 2),
        g_star=g_star,
        h=h,
        prefer=(1, 0, 0),
        optimal_action_at_x=0,
    )


model_i = _model_i()
model_ii = _model_ii()
model_iii = _model_iii()
MODELS = (model_i, model_ii, model_iii)

# Robust gain of the forced risky controller in Model III. Not used by Algorithm 1.
MODEL_III_RISKY_GAIN = jnp.array([0.1, 0.0, 1.0], dtype=DTYPE)
