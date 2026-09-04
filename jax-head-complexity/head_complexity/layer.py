"""The one-layer attention-only model of Section 2, and the scalar normal form.

Definition 1 presents x in {0,1}^n as the (n+1)-token string x_1 x_2 ... x_n ?,
embeds it into rows X(x) in R^{(n+1) x d}, and reads the query position:

    MHA(x) = sum_j Softmax(v_? A_j X^T) X V_j,   Z_s(x) = <w_s, MHA(x)>.

Section 2.3 folds the two logits into a single gap Delta(x) = Z_0(x) - Z_1(x),
which is a sum of k ratios s_x^j / d_x^j, and Definition 2 gives the affine
scalar alternative net(x) = 1{<omega, MHA(x)> - theta > 0}. Clearing the strictly
positive denominators turns either decision into the sign of one polynomial F_x,
which is the object both the lower bound (Theorem 1) and the counting bound
(equation (2)) work with.
"""

from __future__ import annotations

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray


def _f64(a) -> Array:
    return jnp.asarray(a, dtype=jnp.float64)


def all_inputs(n: int) -> Array:
    """The 2^n rows of {0,1}^n in lexicographic order, as an int array."""
    return jnp.asarray(list(itertools.product([0, 1], repeat=n)), dtype=jnp.int32)


# --------------------------------------------------------------------------
# Embeddings (Section 2.1)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class JointEmbedding:
    """The general regime: v_{b,i} = Phi(b, i) is unrestricted.

    `v` has shape (2, n, d) and `v_query` shape (d,).
    """

    v: Array
    v_query: Array

    @property
    def n(self) -> int:
        return int(self.v.shape[1])

    @property
    def d(self) -> int:
        return int(self.v.shape[2])

    def table(self) -> Array:
        return self.v

    def query(self) -> Array:
        return self.v_query

    def augment_constant(self) -> "JointEmbedding":
        """Append a coordinate equal to 1 on every token (Theorem 8, part 2, step 1)."""
        ones = jnp.ones(self.v.shape[:2] + (1,), dtype=self.v.dtype)
        return JointEmbedding(
            jnp.concatenate([self.v, ones], axis=-1),
            jnp.concatenate([self.v_query, jnp.ones((1,), dtype=self.v_query.dtype)]),
        )


@dataclasses.dataclass(frozen=True)
class AdditiveEmbedding:
    """The restricted regime of Section 2.1: v_{b,i} = tau_b + pi_i.

    `tau` has shape (2, d), `pi` shape (n, d). The query token Phi(?) stays
    unrestricted, as in the paper; only the input tokens are tied. Every
    additive model is a joint model, so `table()` just expands the sum.
    """

    tau: Array
    pi: Array
    v_query: Array

    @property
    def n(self) -> int:
        return int(self.pi.shape[0])

    @property
    def d(self) -> int:
        return int(self.tau.shape[1])

    def table(self) -> Array:
        return self.tau[:, None, :] + self.pi[None, :, :]

    def query(self) -> Array:
        return self.v_query

    def augment_constant(self) -> "AdditiveEmbedding":
        """Append a constant-1 coordinate while keeping the additive tie."""
        return AdditiveEmbedding(
            jnp.concatenate([self.tau, jnp.ones((2, 1), dtype=self.tau.dtype)], axis=-1),
            jnp.concatenate([self.pi, jnp.zeros((self.n, 1), dtype=self.pi.dtype)], axis=-1),
            jnp.concatenate([self.v_query, jnp.ones((1,), dtype=self.v_query.dtype)]),
        )


Embedding = JointEmbedding | AdditiveEmbedding


def rows(emb: Embedding, xs: Array) -> Array:
    """X(x) in R^{(n+1) x d} for a batch of inputs xs of shape (B, n)."""
    xs = jnp.asarray(xs, dtype=jnp.int32)
    if xs.ndim == 1:
        xs = xs[None, :]
    table = emb.table()
    tokens = table[xs, jnp.arange(emb.n)]  # (B, n, d)
    query = jnp.broadcast_to(emb.query(), (xs.shape[0], 1, emb.d))
    return jnp.concatenate([tokens, query], axis=1)


# --------------------------------------------------------------------------
# The attention core and the two read-outs (Definitions 1 and 2)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AttentionCore:
    """The k-head attention core of Definition 1, without a read-out.

    `A` has shape (k, d, d) and `V` shape (k, d, d_out).
    """

    emb: Embedding
    A: Array
    V: Array

    @property
    def k(self) -> int:
        return int(self.A.shape[0])

    @property
    def d(self) -> int:
        return int(self.A.shape[1])

    @property
    def d_out(self) -> int:
        return int(self.V.shape[2])

    @property
    def n(self) -> int:
        return self.emb.n

    def xi(self) -> Array:
        """The effective queries xi_j = v_? A_j of Section 4, shape (k, d)."""
        return jnp.einsum("d,jde->je", self.emb.query(), self.A)

    def token_scores(self) -> Array:
        """xi_j . v_t over the 2n+1 token types, shape (k, 2n+1).

        Input-independent: an input only selects which of these scores appear.
        """
        table = self.emb.table().reshape(-1, self.d)
        types = jnp.concatenate([table, self.emb.query()[None, :]], axis=0)
        return jnp.einsum("jd,td->jt", self.xi(), types)


def scores(core: AttentionCore, xs: Array) -> Array:
    """v_? A_j X(x)^T for every head, shape (B, k, n+1)."""
    return jnp.einsum("jd,bnd->bjn", core.xi(), rows(core.emb, xs))


def mha(core: AttentionCore, xs: Array) -> Array:
    """MHA(x) of Definition 1 at the query position, shape (B, d_out)."""
    X = rows(core.emb, xs)
    weights = jax.nn.softmax(jnp.einsum("jd,bnd->bjn", core.xi(), X), axis=-1)
    context = jnp.einsum("bjn,bnd->bjd", weights, X)
    return jnp.einsum("bjd,jdo->bo", context, core.V)


def head_stats(core: AttentionCore, direction: Array, xs: Array) -> tuple[Array, Array]:
    """The scalar normal form of Section 2.3: (d_x^j, s_x^j), each of shape (B, k).

    `direction` is the vector in R^{d_out} the read-out contracts the values
    against: w_0 - w_1 for the arg-max read-out, omega for the affine one. With
    delta_j = V_j direction, u^j_{b,i} = v_{b,i} . delta_j and r^j_{b,i} the
    exponentiated scores,

        d_x^j = r_?^j + sum_i r^j_{x_i,i},   s_x^j = r_?^j u_?^j + sum_i r^j u^j.

    The r's are the raw exp(xi_j . v_t) of Section 2.3, not softmax weights, so
    the numbers here are the ones the proofs quote: head j of Theorem 2 really
    has d_x^j = c + j. Nothing rescales them, which is safe as long as the token
    scores stay in float64 range; every construction and generator in this
    package keeps |xi_j . v_t| below about 100.
    """
    X = rows(core.emb, xs)
    r = jnp.exp(jnp.einsum("jd,bnd->bjn", core.xi(), X))
    delta = jnp.einsum("jdo,o->jd", core.V, _f64(direction))
    u = jnp.einsum("bnd,jd->bjn", X, delta)
    return r.sum(axis=-1), (r * u).sum(axis=-1)


def decision_polynomial(d_x: Array, s_x: Array, theta: float = 0.0) -> Array:
    """F_x, the decision polynomial obtained by clearing the softmax denominators.

    Theorem 1 (arg-max, theta = 0) and equation (2) (affine, general theta):

        F_x = sum_j s_x^j prod_{j' != j} d_x^{j'}  -  theta prod_j d_x^j.

    Every d_x^j is strictly positive, so sign(F_x) = sign(sum_j s_x^j/d_x^j - theta).
    The leave-one-out products are formed explicitly rather than as
    (prod_j d_x^j) * (sum_j s_x^j/d_x^j), so the object under test is the
    polynomial the proof writes down.
    """
    k = d_x.shape[-1]
    total = jnp.zeros(d_x.shape[:-1], dtype=d_x.dtype)
    for j in range(k):
        others = jnp.concatenate([d_x[..., :j], d_x[..., j + 1 :]], axis=-1)
        total = total + s_x[..., j] * jnp.prod(others, axis=-1)
    if theta != 0.0:
        total = total - theta * jnp.prod(d_x, axis=-1)
    return total


@dataclasses.dataclass(frozen=True)
class ArgmaxLayer:
    """Definition 1 in full: the core plus W_O in R^{2 x d_out} with rows w_0, w_1.

    out(x) = argmax_s Z_s(x) with ties resolved toward 0, equivalently
    out(x) = 1 exactly when Z_1(x) > Z_0(x), equivalently Delta(x) < 0.
    """

    core: AttentionCore
    W_O: Array

    def logits(self, xs: Array) -> Array:
        return jnp.einsum("bo,so->bs", mha(self.core, xs), self.W_O)

    def delta(self, xs: Array) -> Array:
        """Delta(x) = Z_0(x) - Z_1(x), equation (1)."""
        z = self.logits(xs)
        return z[:, 0] - z[:, 1]

    def stats(self, xs: Array) -> tuple[Array, Array]:
        return head_stats(self.core, self.W_O[0] - self.W_O[1], xs)

    def F(self, xs: Array) -> Array:
        d_x, s_x = self.stats(xs)
        return decision_polynomial(d_x, s_x)

    def decide(self, xs: Array) -> Array:
        z = self.logits(xs)
        return (z[:, 1] > z[:, 0]).astype(jnp.int32)


@dataclasses.dataclass(frozen=True)
class AffineLayer:
    """Definition 2: net(x) = 1{<omega, MHA(x)> - theta > 0}."""

    core: AttentionCore
    omega: Array
    theta: float

    def G(self, xs: Array) -> Array:
        """G(x) = sum_j s_x^j/d_x^j - theta, the statistic of Section 5.2.1."""
        return jnp.einsum("bo,o->b", mha(self.core, xs), _f64(self.omega)) - self.theta

    def stats(self, xs: Array) -> tuple[Array, Array]:
        return head_stats(self.core, self.omega, xs)

    def F(self, xs: Array) -> Array:
        d_x, s_x = self.stats(xs)
        return decision_polynomial(d_x, s_x, theta=float(self.theta))

    def decide(self, xs: Array) -> Array:
        return (self.G(xs) > 0).astype(jnp.int32)


# --------------------------------------------------------------------------
# Theorem 8: the two read-outs are equivalent on Boolean tasks
# --------------------------------------------------------------------------


def to_affine(layer: ArgmaxLayer) -> AffineLayer:
    """Theorem 8, part 1. Take omega = w_1 - w_0 and theta = 0; no head is added.

    The arg-max boundary is homogeneous, which is why the threshold comes out 0.
    """
    return AffineLayer(layer.core, layer.W_O[1] - layer.W_O[0], 0.0)


def to_argmax(layer: AffineLayer) -> ArgmaxLayer:
    """Theorem 8, part 2. One extra embedding coordinate and one extra value
    coordinate, no extra head.

    The new embedding coordinate is 1 on every token and the new row and column
    of A_j^+ are zero, so no score moves. Every head averages a column of 1s, so
    the new value coordinate of MHA^+(x) equals the head count C = k on every
    input, and w_1^+ = (omega, -theta/C) turns that constant into the threshold.
    """
    core, k = layer.core, layer.core.k
    d, d_out = core.d, core.d_out

    A_plus = jnp.pad(core.A, ((0, 0), (0, 1), (0, 1)))
    V_plus = jnp.pad(core.V, ((0, 0), (0, 1), (0, 1)))
    V_plus = V_plus.at[:, d, d_out].set(1.0)

    W_O = jnp.zeros((2, d_out + 1), dtype=jnp.float64)
    W_O = W_O.at[1, :d_out].set(_f64(layer.omega))
    W_O = W_O.at[1, d_out].set(-float(layer.theta) / k)

    return ArgmaxLayer(AttentionCore(core.emb.augment_constant(), A_plus, V_plus), W_O)


def truth_table(layer, n: int) -> np.ndarray:
    """The full input-output behaviour on {0,1}^n, as a numpy int array."""
    return np.asarray(layer.decide(all_inputs(n)))
