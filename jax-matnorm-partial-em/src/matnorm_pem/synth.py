"""The simulation designs of Section 3.5 and Section 4.3, at small dimensions.

Section 3.5 generates from a matrix normal with a sinusoidal mean,

    M = matrix[sin(1/5), sin(2/5), ..., sin(pq/5)] / 3,

AR(1) row and column factors with rho_1 = 0.55 and rho_2 = 0.45, and
sigma^2 = 1.2, then deletes cells at random (10%, 25%, 50%, 75%) or as a
rectangular block. Section 4.3 builds component means from a shared sinusoidal
surface plus a smooth component pattern P_g and a rectangular shift L_g, with
rho_1g = 0.35 + 0.04g, rho_2g = 0.30 + 0.04g and sigma_g^2 = 0.85 + 0.12g.

The paper runs these at 15 x 20 with N = 1000 to 3000. Here p and q are single
digits, which changes one thing that matters: at 15 x 20 the mixture components
differ across 300 cells and are almost perfectly separable, while at 3 x 4 the
same coefficients leave them overlapping. `mixture_params` therefore takes a
`separation` multiplier on the component-specific part of the mean, default 2.5,
so clustering is a real but solvable problem. Everything else follows the paper.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from .params import MatNormParams, normalize_scale


def ar1(rho, d):
    """AR(1) correlation matrix, (Sigma)_{ab} = rho^{|a - b|}."""
    i = jnp.arange(d)
    return jnp.asarray(rho, dtype=jnp.float64) ** jnp.abs(i[:, None] - i[None, :])


def sinusoidal_mean(p, q, scale=1.0 / 3.0):
    """The Section 3.5 mean, folded column first."""
    v = scale * jnp.sin(jnp.arange(1, p * q + 1, dtype=jnp.float64) / 5.0)
    return v.reshape(q, p).T


def paper_params(p, q, rho1=0.55, rho2=0.45, sigma2=1.2):
    """Section 3.5 truth, normalized to |S1| = |S2| = 1 as Section 2.4 requires."""
    return normalize_scale(sinusoidal_mean(p, q), ar1(rho1, p), ar1(rho2, q), sigma2)


def mcar_mask(key, n, p, q, rate):
    """Cells missing completely at random, True where missing."""
    return jr.bernoulli(key, rate, (n, p, q))


def rect_indicators(key, n, p, q, nr, nc):
    """Random nr x nc missing blocks, returned as row and column indicators.

    Indicators rather than index lists so observations with blocks in different
    places batch together; `block_mask` turns a pair into the p x q mask.
    """
    k1, k2 = jr.split(key)
    r0 = jr.randint(k1, (n,), 0, p - nr + 1)
    c0 = jr.randint(k2, (n,), 0, q - nc + 1)
    ar = ((jnp.arange(p)[None] >= r0[:, None]) & (jnp.arange(p)[None] < (r0 + nr)[:, None]))
    ac = ((jnp.arange(q)[None] >= c0[:, None]) & (jnp.arange(q)[None] < (c0 + nc)[:, None]))
    return ar.astype(jnp.float64), ac.astype(jnp.float64)


def component_params(p, q, g, G, separation=2.5):
    """Component g of the Section 4.3 mixture, g counted from 1."""
    gbar = (G + 1) / 2.0
    Mo = sinusoidal_mean(p, q, scale=0.25)
    r = jnp.linspace(-1.0, 1.0, p)[:, None]
    c = jnp.linspace(-1.0, 1.0, q)[None, :]
    P = jnp.sin((g + 0.5) * jnp.pi * (r + 1) / 2) * jnp.cos((g + 0.25) * jnp.pi * (c + 1) / 2)
    L = jnp.zeros((p, q)).at[: max(p // 2, 1), : max(q // 2, 1)].set(0.55 * (g - gbar))
    M = Mo + separation * (0.85 * (g - gbar) * P + L)
    return normalize_scale(M, ar1(0.35 + 0.04 * g, p), ar1(0.30 + 0.04 * g, q), 0.85 + 0.12 * g)


def mixture_params(p, q, G, separation=2.5):
    """Stacked parameters of a G-component Section 4.3 mixture, equal weights."""
    from .mixture import MixtureParams

    comps = [component_params(p, q, g + 1, G, separation) for g in range(G)]
    stacked = MatNormParams(
        *[jnp.stack([getattr(c, f) for c in comps]) for f in MatNormParams._fields]
    )
    return MixtureParams(jnp.full((G,), 1.0 / G, dtype=jnp.float64), stacked)


def sample_mixture(key, params, n):
    """Draw from (13); returns the matrices and their true component labels."""
    from .density import sample_matnorm

    G = params.G
    klab, kdraw = jr.split(key)
    labels = jr.categorical(klab, jnp.log(params.pi), shape=(n,))
    keys = jr.split(kdraw, G)
    draws = jnp.stack(
        [sample_matnorm(keys[g], MatNormParams(*[l[g] for l in params.comps]), n) for g in range(G)]
    )
    Y = jnp.take_along_axis(draws, labels[None, :, None, None], axis=0)[0]
    return Y, labels
