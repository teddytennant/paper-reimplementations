"""Synthetic FCCO problems with known g_i oracles."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class QuadraticFCCO:
    """Linear-inner quadratic-outer FCCO.

    g_i(w) = A_i w          with A_i of shape (k, d)
    f_i(u) = 0.5 ||u - b_i||^2
    F(w)   = (1 / (2m)) sum_i ||A_i w - b_i||^2

    Stochastic oracle: g_i(w; xi) = A_i w + noise, Jacobian A_i.
    Finite-sum variant: g_i(w; xi_ij) = A_i w + c_ij with mean_j c_ij = 0,
    so the exact inner map remains A_i w.
    """

    def __init__(self, A, b, sigma=0.0, C=None):
        self.A = jnp.asarray(A)
        self.b = jnp.asarray(b)
        self.sigma = float(sigma)
        self.C = None if C is None else jnp.asarray(C)
        self.m, self.k, self.d = map(int, self.A.shape)
        self.n = 0 if self.C is None else int(self.C.shape[1])

    def g_exact(self, w):
        return jnp.einsum("mkd,d->mk", self.A, w)

    def jac_exact(self, w):
        del w
        return self.A

    def f(self, u):
        return 0.5 * jnp.sum((u - self.b) ** 2, axis=-1)

    def F(self, w):
        return jnp.mean(self.f(self.g_exact(w)))

    def grad_f(self, u):
        return u - self.b

    def grad_F(self, w):
        residual = self.g_exact(w) - self.b
        return jnp.einsum("mkd,mk->d", self.A, residual) / self.m

    def sample_blocks(self, key, B1):
        return jax.random.choice(key, self.m, shape=(int(B1),), replace=False)

    def probe(self, w, idx, key):
        """Stochastic (g, Jacobian) for blocks `idx` using randomness `key`.

        The same key at two different w is the two-point probe of one xi.
        """
        g = jnp.einsum("bkd,d->bk", self.A[idx], w)
        jac = self.A[idx]
        if self.C is not None:
            j = jax.random.randint(key, shape=idx.shape, minval=0, maxval=self.n)
            g = g + self.C[idx, j]
        elif self.sigma > 0.0:
            g = g + self.sigma * jax.random.normal(key, g.shape, dtype=g.dtype)
        return g, jac


def make_quadratic(m, d, k, key, sigma=0.0, n=None, scale=0.3):
    """Random quadratic FCCO whose minimizer is near a drawn w_star."""
    k1, k2, k3 = jax.random.split(key, 3)
    A = scale * jax.random.normal(k1, (m, k, d))
    w_star = jax.random.normal(k2, (d,))
    b = jnp.einsum("mkd,d->mk", A, w_star)
    C = None
    if n is not None:
        C = 0.1 * jax.random.normal(k3, (m, n, k))
        C = C - C.mean(axis=1, keepdims=True)
    return QuadraticFCCO(A, b, sigma=sigma, C=C), w_star


class TinyAUC(NamedTuple):
    """Tiny AUC-style compositional problem (optional smoke / extra check).

    g_i(w) = mean_{x in D+_i} w·x - mean_{x in D-_i} w·x
    f_i(g) = 0.5 * relu(margin - g)^2
    """

    pos: jnp.ndarray  # (m, n_pos, d)
    neg: jnp.ndarray  # (m, n_neg, d)
    margin: float = 1.0

    @property
    def m(self):
        return int(self.pos.shape[0])

    @property
    def k(self):
        return 1

    @property
    def d(self):
        return int(self.pos.shape[-1])

    def g_exact(self, w):
        pos = jnp.einsum("ind,d->in", self.pos, w).mean(axis=1)
        neg = jnp.einsum("ind,d->in", self.neg, w).mean(axis=1)
        return (pos - neg)[:, None]

    def jac_exact(self, w):
        del w
        pos_mean = self.pos.mean(axis=1)
        neg_mean = self.neg.mean(axis=1)
        return (pos_mean - neg_mean)[:, None, :]

    def f(self, u):
        g = u.reshape(u.shape[0])
        return 0.5 * jnp.square(jax.nn.relu(self.margin - g))

    def F(self, w):
        return jnp.mean(self.f(self.g_exact(w)))

    def grad_f(self, u):
        g = u.reshape(u.shape[0])
        gap = self.margin - g
        return (-jax.nn.relu(gap))[:, None]

    def sample_blocks(self, key, B1):
        return jax.random.choice(key, self.m, shape=(int(B1),), replace=False)

    def probe(self, w, idx, key):
        k_pos, k_neg = jax.random.split(key)
        n_pos = self.pos.shape[1]
        n_neg = self.neg.shape[1]
        j_pos = jax.random.randint(k_pos, shape=idx.shape, minval=0, maxval=n_pos)
        j_neg = jax.random.randint(k_neg, shape=idx.shape, minval=0, maxval=n_neg)
        x_pos = self.pos[idx, j_pos]
        x_neg = self.neg[idx, j_neg]
        g = jnp.sum((x_pos - x_neg) * w, axis=-1, keepdims=True)
        jac = (x_pos - x_neg)[:, None, :]
        return g, jac
