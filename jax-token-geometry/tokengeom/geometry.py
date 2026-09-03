"""Embedding geometry: the exponential family of Section 2.3 and Theorem 1.

With the softmax head, the model-implied conditional context distributions

    p_hat_{E,v}(X_{j,c} | v) = exp(E_v^T Theta_j(X) - A(E_v)) ptilde(X_{j,c})

form an exponential family in the natural parameter E_v, where

    A(e) = log sum_{X,j} exp(e^T Theta_j(X)) ptilde(X_{j,c}),
    ptilde(X_{j,c}) = p(X_{j,c}) / sum_v exp(E_v^T Theta_j(X)),

and Section 2.3 derives the identity

    H^2(p_hat_{E,v1}, p_hat_{E,v2}) = 1 - exp(-J_A(E_v1, E_v2)),
    J_A(e1, e2) = (A(e1) + A(e2))/2 - A((e1 + e2)/2).

The fitted head carries a token bias b_v, which Appendix B.2 constrains jointly
with E_v. The bias cancels out of the family: it multiplies both
p_hat_E(v|Theta) and the induced marginal p_hat(v) by the same e^{b_v}, so the
natural parameter stays E_v and the bias is absorbed into ptilde. Distances
between token types are therefore taken on E alone, as Section 5 writes them.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from tokengeom import model as M
from tokengeom.train import Enumeration

jax.config.update("jax_enable_x64", True)


@dataclasses.dataclass
class FittedPair:
    """A fitted (E, Theta) pair with every population quantity enumerated.

    `theta` holds Theta_j(X) for each enumerated context, so both the recurrent
    encoder of Section 3 and the unrestricted representation of Section 2.2 are
    handled by the same code.
    """

    theta: jnp.ndarray  # (n_contexts, s)
    embed: jnp.ndarray  # (d, s)
    bias: jnp.ndarray  # (d,)
    enum: Enumeration

    @classmethod
    def from_encoder(cls, params, enum: Enumeration, config: M.EncoderConfig,
                     chunk: int = 8192) -> "FittedPair":
        out = []
        for start in range(0, enum.contexts.shape[0], chunk):
            out.append(
                M.representation(params, enum.contexts[start : start + chunk], config)
            )
        return cls(jnp.concatenate(out, 0), params["embed"], params["head_bias"], enum)

    @classmethod
    def from_unrestricted(cls, params, enum: Enumeration) -> "FittedPair":
        return cls(params["z"], params["embed"], params["head_bias"], enum)

    @classmethod
    def from_arrays(cls, theta, embed, bias, enum: Enumeration) -> "FittedPair":
        return cls(jnp.asarray(theta), jnp.asarray(embed), jnp.asarray(bias), enum)

    # -- basic quantities ----------------------------------------------------

    def embeddings(self) -> jnp.ndarray:
        """E, the natural parameter of the family."""
        return self.embed

    def representations(self) -> jnp.ndarray:
        """Theta_j(X) on every enumerated context."""
        return self.theta

    @functools.cached_property
    def logits(self) -> jnp.ndarray:
        return self.theta @ self.embed.T + self.bias

    @functools.cached_property
    def log_predictions(self) -> jnp.ndarray:
        """log p_hat_E(v | Theta_j(X))."""
        return jax.nn.log_softmax(self.logits, axis=-1)

    @functools.cached_property
    def epsilon_p(self) -> float:
        """The population prediction error of Assumption 1.1."""
        e = self.enum
        return float(
            -jnp.sum(e.context_prob[:, None] * e.conditional * self.log_predictions)
            + e.entropy
        )

    @functools.cached_property
    def token_marginal(self) -> jnp.ndarray:
        """p_hat(v), the marginal induced by the fitted conditionals."""
        return jnp.exp(self.log_predictions).T @ self.enum.context_prob

    # -- exponential family --------------------------------------------------

    @functools.cached_property
    def log_base_measure(self) -> jnp.ndarray:
        """log ptilde(X_{j,c})."""
        return jnp.log(self.enum.context_prob) - jax.scipy.special.logsumexp(
            self.logits, axis=1
        )

    def log_normalizer(self, e) -> jnp.ndarray:
        """A(e) for one natural parameter or a stack of them."""
        theta = self.representations()
        flat = jnp.atleast_2d(e)
        out = jax.scipy.special.logsumexp(flat @ theta.T + self.log_base_measure, axis=-1)
        return out if jnp.ndim(e) > 1 else out[0]

    def jensen_gap(self, e1, e2) -> float:
        """J_A(e1, e2)."""
        a = self.log_normalizer(jnp.stack([e1, e2, 0.5 * (e1 + e2)]))
        return float(0.5 * (a[0] + a[1]) - a[2])

    def hessian_A(self, e) -> jnp.ndarray:
        """grad^2 A(e), the covariance of Theta under the tilted base measure."""
        theta = self.representations()
        w = jax.nn.softmax(theta @ e + self.log_base_measure)
        mean = w @ theta
        return jnp.einsum("n,ni,nj->ij", w, theta, theta) - jnp.outer(mean, mean)

    def model_context_distributions(self) -> jnp.ndarray:
        """p_hat_{E,v}(X_{j,c} | v), shape (d, n_contexts)."""
        joint = self.enum.context_prob[:, None] * jnp.exp(self.log_predictions)
        return (joint / joint.sum(axis=0)[None, :]).T

    def curvature_bounds(self, n_grid: int = 9, subset=None) -> tuple[float, float]:
        """b_Theta^2 and B_Theta^2 of Assumption 1.3, measured not assumed.

        The assumption asks for a sandwich on etilde^T grad^2 A(e) etilde over
        e in conv{E_v} and etilde in span{E_v1 - E_v2}. Theorem 1 uses it only
        through the Jensen gap of a pair, and by the integral form of that gap
        the e that matters always lies on the segment between the two rows. So
        the grid is the union of those segments; on each grid point the extreme
        Rayleigh quotients over the difference span are exact eigenvalues.

        `subset` restricts the token types, which is what Assumption 5.3 asks
        for when the curvature is only needed inside one preferred set W_k.
        """
        rows = self.embeddings()
        if subset is not None:
            rows = rows[jnp.asarray(subset)]
        basis = difference_span(rows)
        ts = np.linspace(0.0, 1.0, n_grid)
        lo, hi = np.inf, -np.inf
        n = rows.shape[0]
        for i in range(n):
            for j in range(i, n):
                for t in ts:
                    h = basis.T @ self.hessian_A((1.0 - t) * rows[i] + t * rows[j]) @ basis
                    w = jnp.linalg.eigvalsh(0.5 * (h + h.T))
                    lo = min(lo, float(w[0]))
                    hi = max(hi, float(w[-1]))
        return max(lo, 0.0), hi

    # -- Section 5 summaries -------------------------------------------------

    def squared_embedding_distances(self) -> np.ndarray:
        e = np.asarray(self.embeddings())
        return ((e[:, None, :] - e[None, :, :]) ** 2).sum(axis=-1)

    def group_embedding_distances(self, sim) -> np.ndarray:
        """D_g(a1, a2), the squared distance between semantic group centroids."""
        e = np.asarray(self.embeddings())
        centers = np.stack(
            [e[sim.token_group == a].mean(axis=0) for a in range(sim.config.n_groups)]
        )
        return ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)

    def synonym_distance_ratio(self, sim) -> float:
        """D_s: mean squared distance inside semantic groups over the median
        squared distance between them."""
        d = self.squared_embedding_distances()
        same = sim.token_group[:, None] == sim.token_group[None, :]
        return float(d[same].sum() / len(d) / np.median(d[~same]))


def difference_span(e_rows, tol: float = 1e-10) -> jnp.ndarray:
    """Orthonormal basis of span{E_v1 - E_v2}, the set of Assumption 1.3."""
    centered = np.asarray(e_rows) - np.asarray(e_rows).mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered.T, full_matrices=False)
    keep = s > tol * max(1.0, float(s[0]))
    return jnp.asarray(u[:, keep])


def theorem1_upper(embedding_distance, curvature_hi, epsilon_p, p_min):
    """The upper bound of Theorem 1 on sqrt(Gamma_h(v1, v2))."""
    d2 = np.asarray(embedding_distance)
    return np.sqrt(1.0 - np.exp(-curvature_hi / 8.0 * d2)) + np.sqrt(epsilon_p / p_min)


def theorem1_lower(embedding_distance, curvature_lo, epsilon_p, p_min):
    """The lower bound of Theorem 1 on sqrt(Gamma_h(v1, v2))."""
    d2 = np.asarray(embedding_distance)
    return np.maximum(
        np.sqrt(1.0 - np.exp(-curvature_lo / 8.0 * d2)) - np.sqrt(epsilon_p / p_min), 0.0
    )


def population_context_distributions(sim) -> np.ndarray:
    """p_v(X_{j,c} | v) over observed-token contexts, from the ground truth."""
    d = sim.config.vocab_size
    idx = np.arange(d**3)
    contexts = np.stack([idx // d**2, (idx // d) % d, idx % d], axis=1)
    joint = sim.context_distribution_tokens[:, None] * sim.token_conditional[
        sim.context_index(contexts)
    ]
    return (joint / joint.sum(axis=0)[None, :]).T


def spearman(x, y) -> float:
    """Spearman rank correlation, the statistic reported in Figures 1, 2 and 5."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=np.float64)
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    rx = rank(x) - rank(x).mean()
    ry = rank(y) - rank(y).mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def upper_triangle(matrix) -> np.ndarray:
    m = np.asarray(matrix)
    i, j = np.triu_indices(m.shape[0], k=1)
    return m[i, j]


def coordinate_change(pair: FittedPair, matrix) -> FittedPair:
    """The reparameterization of Section 3.1: Theta -> A Theta, E -> A^{-T} E.

    Leaves every kernel score, and so the token-prediction loss, unchanged.
    """
    a = jnp.asarray(matrix)
    inv_t = jnp.linalg.inv(a).T
    return FittedPair(pair.theta @ a.T, pair.embed @ inv_t.T, pair.bias, pair.enum)


def best_permutation_accuracy(predicted, truth, n_labels: int) -> float:
    """1 - Phi(chat, c), maximized over relabelings, as in Section 4.2."""
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    best = 0
    for perm in itertools.permutations(range(n_labels)):
        best = max(best, int((predicted == np.asarray(perm)[truth]).sum()))
    return best / len(truth)


def whiten_coordinates(pair: FittedPair, ridge: float = 1e-10) -> FittedPair:
    """Pick the representative of the equivalence class that Theorem 1 likes.

    Section 3.1: the token-prediction loss cannot separate (Theta, E) from
    (A Theta, A^{-T} E), but the curvature constants of Assumption 1.3 differ
    across the class, and so does how much the Euclidean geometry of E says
    about Gamma_h. Under the change of coordinates the normalizer transforms as
    Atilde(e) = A(A^T e), so grad^2 Atilde = A grad^2 A A^T. Taking A to be the
    inverse square root of the Hessian at the embedding centroid sets that
    curvature to the identity there, which is the best-conditioned choice.
    """
    center = pair.embeddings().mean(axis=0)
    hessian = np.asarray(pair.hessian_A(center))
    hessian = 0.5 * (hessian + hessian.T)
    values, vectors = np.linalg.eigh(hessian)
    values = np.maximum(values, ridge * max(float(values[-1]), 1e-30))
    return coordinate_change(pair, vectors @ np.diag(values**-0.5) @ vectors.T)
