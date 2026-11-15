"""Self-consistency and recurrent-depth refinement, Section 3 and Theorem 2.

Two things live here.

The first is the decoder-optimal representation of Section 5,

    Z_j(X) = argmin_{||Z|| <= 4} L(p_j(X); p_hat_Ehat(Z)),

and the normalized distance D_Z(l) between the l-th intermediate representation
of the recurrent encoder and Z. Figure 3 plots D_Z(l) falling geometrically as
the same block is applied again.

The second is a finite instantiation of the self-consistency operator itself. A
representation map is self-consistent when Theta_j(X) = g_Theta(Theta_{j,c}(X)),
and Section 3.2 studies the iteration Psi^{l+1} = T_{g}(Psi^l). Here a
representation map is a vector per token type, and

    [T_g(Psi)](v) = g( E[Psi(x_{j-3}) | x_j = v],
                       E[Psi(x_{j-2}) | x_j = v],
                       E[Psi(x_{j-1}) | x_j = v] ),

the smooth recovery of a token's representation from the representations of the
contexts it occurs in, under the paper's own p_v(X_{j,c}|v). The metric is the
paper's D(Psi, Phi) = {E_{X,j} ||Psi_j - Phi_j||^2}^{1/2}, which here is the
p(v)-weighted mean. Under this pairing the conditioning matrices are averaging
operators and so are nonexpansive in that metric, and the Lipschitz condition
Section 3.2 states, ||g(y) - g(y')||^2 <= (L_g/m) sum_j ||y_j - y_j'||^2, gives
a contraction with factor kappa_Theta = sqrt(L_g).

A residual transformer block is not a contraction in this metric, so Theorem 2's
inequality is checked on this operator rather than on the encoder, and the
encoder is checked against Figure 3's empirical geometric decay instead.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tokengeom import model as M
from tokengeom.train import Enumeration

jax.config.update("jax_enable_x64", True)


# -- decoder-optimal representation ------------------------------------------


def ideal_representation(pair, n_steps: int = 3000, learning_rate: float = 0.05,
                         norm: float = M.REPRESENTATION_NORM) -> jnp.ndarray:
    """Z_j(X) for the fitted head, by projected gradient on a convex problem.

    L(p; p_hat_E(Z)) is convex in Z and the constraint ||Z|| <= 4 is convex, so
    the projected iteration has a unique limit. Appendix B.2 uses the same
    radius the encoder's readout enforces.
    """
    embed, bias, enum = pair.embed, pair.bias, pair.enum
    target = enum.conditional
    weight = enum.context_prob

    def project(z):
        n = jnp.linalg.norm(z, axis=1, keepdims=True)
        return jnp.where(n > norm, z * norm / jnp.maximum(n, 1e-12), z)

    def loss(z):
        logq = jax.nn.log_softmax(z @ embed.T + bias, axis=-1)
        return -jnp.sum(weight[:, None] * target * logq)

    z = project(jnp.zeros_like(pair.theta))
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(z)

    def step(carry, _):
        z, state = carry
        value, grad = jax.value_and_grad(loss)(z)
        updates, state = optimizer.update(grad, state, z)
        return (project(optax.apply_updates(z, updates)), state), value

    (z, _), _ = jax.lax.scan(jax.jit(step), (z, state), None, length=n_steps)
    return z


def weighted_mean_square(x, weight) -> float:
    return float(jnp.sum(weight * jnp.sum(x**2, axis=-1)))


def normalized_distance_to_ideal(layer_reps, ideal, weight) -> np.ndarray:
    """D_Z(l) for l = 0, ..., L, the criterion plotted in Figures 1b and 3."""
    centered = ideal - jnp.sum(weight[:, None] * ideal, axis=0)[None, :]
    scale = weighted_mean_square(centered, weight)
    return np.asarray(
        [weighted_mean_square(psi - ideal, weight) / scale for psi in layer_reps]
    )


def encoder_layer_representations(params, enum: Enumeration, config: M.EncoderConfig,
                                  chunk: int = 4096) -> jnp.ndarray:
    """Psi^l_j(X) for l = 0, ..., L on every enumerated context."""
    out = []
    for start in range(0, enum.contexts.shape[0], chunk):
        out.append(
            M.layer_representations(params, enum.contexts[start : start + chunk], config)
        )
    return jnp.concatenate(out, axis=1)


# -- the self-consistency operator -------------------------------------------


def context_conditionals(sim) -> jnp.ndarray:
    """P(x_{j-4+i} = v' | x_j = v) for i = 1, 2, 3, shape (3, d, d).

    Marginals of the paper's conditional context distribution p_v(X_{j,c} | v).
    Because they are conditional expectations of a coordinate, each is an
    averaging operator, hence nonexpansive in the p(v)-weighted L2 metric.
    """
    d, A = sim.config.vocab_size, sim.config.n_groups
    joint = sim.state_distribution[:, None] * sim.group_conditional  # (A^3, A)
    out = np.zeros((3, d, d))
    for i in range(3):
        group_of_position = sim.context_groups[:, i]
        block = np.zeros((A, A))
        np.add.at(block, (group_of_position, slice(None)), joint)
        block = block.T  # (next group, context group at position i)
        block = block / block.sum(axis=1, keepdims=True)
        token_block = (
            block[np.ix_(sim.token_group, sim.token_group)]
            * sim.variant_weights[sim.token_variant][None, :]
        )
        out[i] = token_block
    return jnp.asarray(out)


@dataclasses.dataclass
class SmoothBlock:
    """g(y) = c * tanh(sum_i y_i W_i) with the Section 3.2 Lipschitz property.

    Each W_i has spectral norm one and tanh is nonexpansive, so
    ||g(y) - g(y')||^2 <= 3 c^2 sum_i ||y_i - y_i'||^2, which is the paper's
    condition with L_g = 9 c^2 at m = 3 context positions. Composed with the
    three nonexpansive averaging operators this gives a contraction with the
    certified factor kappa_Theta = sqrt(L_g) = 3c.
    """

    weights: jnp.ndarray  # (3, s, s)
    scale: float
    offset: jnp.ndarray  # the perturbation that realizes delta_Theta

    @property
    def lipschitz(self) -> float:
        return 9.0 * self.scale**2

    @property
    def kappa(self) -> float:
        return float(np.sqrt(self.lipschitz))

    def __call__(self, y):
        return self.scale * jnp.tanh(jnp.einsum("ink,ikl->nl", y, self.weights)) + self.offset


def make_block(seed: int, width: int, kappa: float, delta: float = 0.0,
               n_tokens: int | None = None, token_weight=None) -> SmoothBlock:
    """A block with certified contraction factor `kappa` and offset norm `delta`.

    `delta` is realized as a fixed field whose D-norm is exactly delta, so
    sup_Psi D(T_{g_Theta}(Psi), T_g(Psi)) = delta and Theorem 2's second term is
    known rather than estimated.
    """
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(3):
        w = rng.standard_normal((width, width))
        weights.append(w / np.linalg.norm(w, 2))
    offset = jnp.zeros((n_tokens or 1, width))
    if delta > 0.0:
        raw = rng.standard_normal((n_tokens, width))
        weight = np.asarray(token_weight)
        norm = np.sqrt(float((weight * (raw**2).sum(axis=1)).sum()))
        offset = jnp.asarray(raw * delta / norm)
    return SmoothBlock(jnp.asarray(np.stack(weights)), kappa / 3.0, offset)


def apply_operator(block: SmoothBlock, gathers: jnp.ndarray, psi: jnp.ndarray):
    """T_g(Psi), the operator of Section 3.1 on token-level representation maps."""
    return block(jnp.einsum("idv,vk->idk", gathers, psi))


def map_distance(psi, phi, weight) -> float:
    """D(Psi, Phi) = {E_{X,j} ||Psi_j - Phi_j||^2}^{1/2}."""
    return float(jnp.sqrt(jnp.sum(weight * jnp.sum((psi - phi) ** 2, axis=-1))))


def iterate_operator(block: SmoothBlock, gathers, psi0, n_iter: int):
    """Psi^0, ..., Psi^L under repeated application of one shared block."""
    out = [psi0]
    psi = psi0
    for _ in range(n_iter):
        psi = apply_operator(block, gathers, psi)
        out.append(psi)
    return out


def fixed_point(block: SmoothBlock, gathers, psi0, weight, tol: float = 1e-14,
                max_iter: int = 20_000):
    """The self-consistent representation Theta = T_{g_Theta}(Theta)."""
    psi = psi0
    for _ in range(max_iter):
        nxt = apply_operator(block, gathers, psi)
        if map_distance(nxt, psi, weight) < tol:
            return nxt
        psi = nxt
    raise RuntimeError("operator iteration did not reach its fixed point")


def theorem2_bound(kappa: float, delta: float, depth: int, initial_error: float) -> float:
    """kappa^L D_0 + (1 - kappa^L)/(1 - kappa) delta, the bound of Theorem 2."""
    return kappa**depth * initial_error + (1 - kappa**depth) / (1 - kappa) * delta


def empirical_lipschitz(block: SmoothBlock, gathers, weight, seed: int,
                        n_trials: int = 40, radius: float = 1.0) -> float:
    """The largest observed D(T(Psi), T(Phi)) / D(Psi, Phi) over random pairs."""
    rng = np.random.default_rng(seed)
    shape = (gathers.shape[1], block.weights.shape[-1])
    worst = 0.0
    for _ in range(n_trials):
        psi = jnp.asarray(rng.standard_normal(shape) * radius)
        phi = jnp.asarray(rng.standard_normal(shape) * radius)
        num = map_distance(
            apply_operator(block, gathers, psi), apply_operator(block, gathers, phi), weight
        )
        worst = max(worst, num / map_distance(psi, phi, weight))
    return worst
