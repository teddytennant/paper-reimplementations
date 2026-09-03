"""Proposition 1 and Corollary 3: the pathway collapse and the nested-model reductions.

The paper's main structural claim is that a complete latent pathway, after the
Gaussian latent states are marginalised, is exactly GHSt_p(mu_s, Sigma_s,
alpha_s, nu_s) with the recursions (5)-(7). These tests generate from the
hierarchy itself, layer by layer, and compare against that closed form.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import multivariate_t

from dstmm import (
    DStMM,
    collapse_paths,
    ghst_cov,
    ghst_logpdf,
    ghst_mean,
    path_log_densities,
    sample_dstmm,
    student_t_logpdf,
)


def toy_model(nu=12.0, skew=1.0, seed=0, p=4, r1=3, r2=2):
    k = jax.random.split(jax.random.key(seed), 4)
    lam1 = jax.random.normal(k[0], (2, p, r1), dtype=jnp.float64) * 0.6
    lam2 = jax.random.normal(k[1], (2, r1, r2), dtype=jnp.float64) * 0.6
    eta1 = jax.random.normal(k[2], (2, p), dtype=jnp.float64) * 1.5
    eta2 = jax.random.normal(k[3], (2, r1), dtype=jnp.float64) * 0.8
    d1 = skew * jnp.stack(
        [jnp.array([0.5, 0.5, 0.0, 0.0])[:p], jnp.array([0.0, -0.5, -0.5, 0.0])[:p]]
    )
    return DStMM(
        dims=(p, r1, r2),
        ks=(2, 2),
        log_pi=(jnp.log(jnp.array([0.6, 0.4])), jnp.log(jnp.array([0.45, 0.55]))),
        eta=(eta1, eta2),
        lam=(lam1, lam2),
        psi_diag=(jnp.full((2, p), 0.5), jnp.full((2, r1), 0.3)),
        delta=(d1, jnp.zeros((2, r1))),
        nu=jnp.full((4,), float(nu)),
    )


def _pin_to_path(model, path):
    """Force every observation down one complete path by making the layer weights degenerate."""
    log_pi = tuple(
        jnp.log(jnp.eye(model.ks[l])[path[l]]) for l in range(model.depth)
    )
    return model.replace(log_pi=log_pi)


def test_pathway_marginal_moments():
    """Sample the hierarchy down a fixed pathway and check the first two moments
    against E(X) = mu_s + E(W) alpha_s and Cov(X) = E(W) Sigma_s + Var(W) alpha_s alpha_s'
    with mu_s, alpha_s, Sigma_s from the recursions (5)-(7)."""
    model = toy_model(nu=12.0)
    path = (1, 0)
    idx = model.paths.index(path)
    mu_seq, alpha_seq, sigma_seq, _ = collapse_paths(model)
    mu, alpha, sigma = mu_seq[0][idx], alpha_seq[0][idx], sigma_seq[0][idx]
    nu = float(model.nu[idx])

    n = 200_000
    y, path_index, _, _ = sample_dstmm(jax.random.key(7), _pin_to_path(model, path), n)
    assert int(path_index.min()) == idx and int(path_index.max()) == idx
    y = np.asarray(y)

    mean_ref = np.asarray(ghst_mean(mu, alpha, nu))
    se = y.std(axis=0) / np.sqrt(n)
    assert np.all(np.abs(y.mean(axis=0) - mean_ref) < 5.0 * se)

    cov_ref = np.asarray(ghst_cov(sigma, alpha, nu))
    c = y - y.mean(axis=0)
    prod = c[:, :, None] * c[:, None, :]
    se_cov = prod.std(axis=0) / np.sqrt(n)
    assert np.all(np.abs(prod.mean(axis=0) - cov_ref) < 5.0 * se_cov)


def test_pathway_marginal_distribution_via_projection():
    """The distributional form of the claim, not just two moments.

    If Y is GHSt_p(mu, Sigma, alpha, nu) then a'Y is GHSt_1(a'mu, a'Sigma a, a'alpha, nu),
    because the representation (2) survives any linear map. So the closed-form
    one-dimensional density can be integrated to a CDF and compared with the
    empirical CDF of a'Y over hierarchy draws. The band is 2.5/sqrt(n), roughly a
    Kolmogorov p-value of 1e-7 under the null.
    """
    model = toy_model(nu=7.0, skew=1.4)
    n = 60_000
    for path in [(1, 0), (0, 1)]:
        idx = model.paths.index(path)
        mu_seq, alpha_seq, sigma_seq, _ = collapse_paths(model)
        mu, alpha, sigma = mu_seq[0][idx], alpha_seq[0][idx], sigma_seq[0][idx]
        nu = float(model.nu[idx])

        y, _, _, _ = sample_dstmm(jax.random.key(11), _pin_to_path(model, path), n)
        a = jnp.array([0.5, -0.7, 0.3, 0.9])[: model.dims[0]]
        a = a / jnp.linalg.norm(a)
        x = np.asarray(y @ a)

        m1 = jnp.array([a @ mu])
        s1 = jnp.array([[a @ sigma @ a]])
        a1 = jnp.array([a @ alpha])

        grid = np.linspace(x.min() - 5.0, x.max() + 5.0, 20_001)
        dens = np.asarray(jnp.exp(ghst_logpdf(jnp.asarray(grid)[:, None], m1, s1, a1, nu)))
        cdf = cumulative_trapezoid(dens, grid, initial=0.0)
        assert abs(cdf[-1] - 1.0) < 1e-4

        probs = np.linspace(0.01, 0.99, 99)
        emp_q = np.quantile(x, probs)
        assert np.abs(np.interp(emp_q, grid, cdf) - probs).max() < 2.5 / np.sqrt(n)


def test_conditional_gaussian_at_every_layer():
    """Equation (8): given the path and W = w, layer l is N(mu_s^(l) + w alpha_s^(l), w Sigma_s^(l)).

    Checked at the observation layer and at layer 1, with w held fixed so the
    statement is the conditional one rather than the mixture.
    """
    model = toy_model(nu=9.0, skew=1.2)
    path = (0, 1)
    idx = model.paths.index(path)
    mu_seq, alpha_seq, sigma_seq, _ = collapse_paths(model)

    # W is held at a constant instead of being drawn, so the layers are simulated
    # here rather than through sample_dstmm.
    w_fixed = 1.7
    pinned = _pin_to_path(model, path)
    n = 150_000
    key = jax.random.split(jax.random.key(13), 3)
    z = jnp.sqrt(w_fixed) * jax.random.normal(key[0], (n, model.dims[2]), dtype=jnp.float64)
    for l, kk in ((2, key[1]), (1, key[2])):
        a = path[l - 1]
        mean = (
            pinned.eta[l - 1][a]
            + z @ pinned.lam[l - 1][a].T
            + w_fixed * pinned.delta[l - 1][a]
        )
        eps = jax.random.normal(kk, mean.shape, dtype=jnp.float64)
        z_next = mean + np.sqrt(w_fixed) * jnp.sqrt(pinned.psi_diag[l - 1][a]) * eps
        level = l - 1
        ref_mean = np.asarray(mu_seq[level][idx] + w_fixed * alpha_seq[level][idx])
        ref_cov = w_fixed * np.asarray(sigma_seq[level][idx])
        arr = np.asarray(z_next)
        se = arr.std(axis=0) / np.sqrt(n)
        assert np.all(np.abs(arr.mean(axis=0) - ref_mean) < 5.0 * se)
        emp_cov = np.cov(arr.T)
        # Gaussian at fixed w, so Var(S_ij) = (C_ii C_jj + C_ij^2)/n exactly.
        se_cov = np.sqrt(
            (np.outer(np.diag(ref_cov), np.diag(ref_cov)) + ref_cov**2) / n
        )
        assert np.all(np.abs(emp_cov - ref_cov) < 5.0 * se_cov)
        z = z_next


def test_single_layer_reduction():
    """Corollary 3 part 3: with h = 1 the recursions give mu_s = eta_s^(1),
    alpha_s = delta_s^(1), Sigma_s = Psi_s^(1) + Lambda_s^(1) Lambda_s^(1)'."""
    k = jax.random.split(jax.random.key(4), 3)
    p, r1 = 5, 2
    eta = jax.random.normal(k[0], (3, p), dtype=jnp.float64)
    lam = jax.random.normal(k[1], (3, p, r1), dtype=jnp.float64)
    delta = jax.random.normal(k[2], (3, p), dtype=jnp.float64) * 0.4
    psi = jnp.abs(jax.random.normal(jax.random.key(5), (3, p), dtype=jnp.float64)) + 0.2
    model = DStMM(
        dims=(p, r1),
        ks=(3,),
        log_pi=(jnp.log(jnp.array([0.2, 0.5, 0.3])),),
        eta=(eta,),
        lam=(lam,),
        psi_diag=(psi,),
        delta=(delta,),
        nu=jnp.full((3,), 8.0),
    )
    mu_seq, alpha_seq, sigma_seq, _ = collapse_paths(model)
    for a in range(3):
        assert np.array_equal(np.asarray(mu_seq[0][a]), np.asarray(eta[a]))
        assert np.array_equal(np.asarray(alpha_seq[0][a]), np.asarray(delta[a]))
        ref = np.diag(np.asarray(psi[a])) + np.asarray(lam[a] @ lam[a].T)
        assert np.abs(np.asarray(sigma_seq[0][a]) - ref).max() < 1e-14


def test_symmetric_reduction_is_a_deep_t_mixture():
    """Corollary 3 part 1: delta = 0 at every layer forces alpha_s = 0 on every path
    through the recursion (6), and the observed mixture becomes a Student-t deep
    mixture. Compared against scipy's multivariate_t, component by component."""
    model = toy_model(nu=7.0, skew=0.0)
    mu_seq, alpha_seq, sigma_seq, log_pi = collapse_paths(model)
    assert np.abs(np.asarray(alpha_seq[0])).max() == 0.0

    y = jax.random.normal(jax.random.key(6), (25, model.dims[0]), dtype=jnp.float64) * 2.0
    got = np.asarray(path_log_densities(model, y))
    for i in range(len(model.paths)):
        ref = multivariate_t.logpdf(
            np.asarray(y),
            loc=np.asarray(mu_seq[0][i]),
            shape=np.asarray(sigma_seq[0][i]),
            df=float(model.nu[i]),
        ) + float(log_pi[i])
        assert np.abs(got[:, i] - ref).max() < 1e-12
        mine = np.asarray(
            student_t_logpdf(y, mu_seq[0][i], sigma_seq[0][i], model.nu[i])
        ) + float(log_pi[i])
        # Same quantity by two routes through the package, so this is float64
        # rounding rather than a modelling gap. It is not bit-exact: XLA fuses
        # the two differently on a GPU and the log-densities land ~4e-15 apart
        # at magnitude 13, which is a couple of ULP.
        assert np.abs(got[:, i] - mine).max() < 1e-13
