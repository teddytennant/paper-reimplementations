"""The controlled-skewness generator of Section 4.1, shrunk to CPU-sized dimensions.

The paper uses p = 20, (r_1, r_2) = (5, 2), (K_1, K_2) = (2, 2), balanced layer
proportions, first-layer locations at plus and minus 2.5 on the first three
coordinates, Kronecker-structured loadings, Psi^(1) = 0.5 I and Psi^(2) = 0.3 I,
and skewness delta_1^(1) = delta_2^(1) = kappa * v along a fixed unit direction v
with the active set A = {1}. Because E(W) = nu/(nu-2), it recentres the
first-layer locations as

    eta_a^(1)(nu, kappa) = eta_a,0^(1) - (nu/(nu-2)) kappa v,

so changing kappa changes asymmetry without moving the unconditional centre.

Everything here keeps that mechanism, including the recentring, but at p = 6 and
(r_1, r_2) = (3, 2) so the tests run on a CPU in seconds. The absolute numbers
are not the paper's.
"""

import jax.numpy as jnp

from dstmm.model import DStMM


def paper_style_truth(p=6, r1=3, r2=2, nu=6.0, kappa=1.0, pi1=0.5, pi2=0.5):
    """A two-layer DStMM built the way Section 4.1 builds its data-generating model."""
    sep = 2.5
    eta1_0 = jnp.stack(
        [
            -sep * (jnp.arange(p) < 3).astype(jnp.float64),
            sep * (jnp.arange(p) < 3).astype(jnp.float64),
        ]
    )
    v = jnp.zeros(p).at[: min(4, p)].set(0.5)
    v = v / jnp.linalg.norm(v)
    delta1 = jnp.stack([kappa * v, kappa * v])
    eta1 = eta1_0 - (nu / (nu - 2.0)) * kappa * v

    base = jnp.array([1.0, 0.9, 1.1])[:, None] * jnp.ones((1, r1))
    a_mat = jnp.zeros((p, r1))
    for i in range(p):
        a_mat = a_mat.at[i, i % r1].set(float(base[i % 3, 0]))
    d2 = jnp.diag(jnp.array([1.10, 0.90, 1.05])[:r1])
    lam1 = jnp.stack([a_mat, a_mat @ d2])

    l21 = jnp.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])[:r1, :r2]
    l22 = jnp.array([[0.9, 0.1], [0.7, -0.2], [0.1, 0.9]])[:r1, :r2]
    lam2 = jnp.stack([l21, l22])
    eta2 = jnp.stack(
        [1.25 * jnp.eye(r1)[0], -1.25 * jnp.eye(r1)[0]]
    )

    return DStMM(
        dims=(p, r1, r2),
        ks=(2, 2),
        log_pi=(
            jnp.log(jnp.array([pi1, 1.0 - pi1])),
            jnp.log(jnp.array([pi2, 1.0 - pi2])),
        ),
        eta=(eta1, eta2),
        lam=(lam1, lam2),
        psi_diag=(jnp.full((2, p), 0.5), jnp.full((2, r1), 0.3)),
        delta=(delta1, jnp.zeros((2, r1))),
        nu=jnp.full((4,), float(nu)),
    )
