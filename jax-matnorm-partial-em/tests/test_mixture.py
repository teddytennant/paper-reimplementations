"""Section 4: clustering and imputation at the same time."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import matnorm_pem as mp
from matnorm_pem import synth
from matnorm_pem.mixture import (
    _component,
    adjusted_rand_index,
    fit_mixture,
    imputed_data,
    mixture_estep,
    mixture_loglik,
)


def _data(p, q, G, n, rate, seed, separation=2.5):
    truth = synth.mixture_params(p, q, G, separation)
    Y, labels = synth.sample_mixture(jr.key(seed), truth, n)
    mask = synth.mcar_mask(jr.key(seed + 700), n, p, q, rate)
    return truth, Y, labels, mask


@pytest.mark.parametrize("method", ["exact", "partial"])
@pytest.mark.parametrize("seed", [0, 1])
def test_mixture_em_never_decreases_the_loglik(method, seed):
    """Monotonicity of (13) under the E-step of Section 4.1 and the M-step (14) to (16)."""
    _, Y, _, mask = _data(3, 4, 2, 150, 0.25, seed)
    _, _, hist = fit_mixture(Y, mask, 2, jr.key(seed + 3), method=method, n_iter=25)
    ll = np.asarray(hist["loglik"])
    assert np.min(np.diff(ll)) > -1e-9 * max(1.0, abs(ll[-1]))


def test_partial_mixture_matches_exact_mixture():
    """Same likelihood and the same clustering, which is the Table 1 claim."""
    n = 200
    _, Y, labels, mask = _data(3, 4, 2, n, 0.25, 4)
    p_exact, r_exact, h_exact = fit_mixture(Y, mask, 2, jr.key(1), method="exact", n_iter=25)
    p_part, r_part, h_part = fit_mixture(Y, mask, 2, jr.key(1), method="partial", n_iter=25)
    assert abs(h_exact["loglik"][-1] - h_part["loglik"][-1]) / n < 1e-3
    hard_e = np.asarray(jnp.argmax(r_exact, axis=1))
    hard_p = np.asarray(jnp.argmax(r_part, axis=1))
    assert adjusted_rand_index(hard_e, hard_p) > 0.98
    assert float(jnp.max(jnp.abs(p_exact.pi - p_part.pi))) < 5e-3


def test_mixture_recovers_the_clustering_and_the_parameters():
    """Two components, a quarter of the cells missing, five seeds.

    Clustering is scored by ARI against the true labels and averaged over seeds,
    because any single fit can land in a bad local optimum and a per-seed
    threshold would be a coin flip. Component parameters are compared after
    matching fitted components to true ones by mean distance.
    """
    p, q, G, n = 3, 4, 2, 250
    aris = []
    mean_errs = []
    sigma_errs = []
    for seed in range(5):
        truth, Y, labels, mask = _data(p, q, G, n, 0.25, seed + 20)
        est, resp, _ = fit_mixture(Y, mask, G, jr.key(seed), method="partial", n_iter=30)
        hard = np.asarray(jnp.argmax(resp, axis=1))
        aris.append(adjusted_rand_index(np.asarray(labels), hard))
        d = np.array([[float(jnp.max(jnp.abs(est.comps.M[a] - truth.comps.M[b])))
                       for b in range(G)] for a in range(G)])
        perm = [int(np.argmin(d[a])) for a in range(G)]
        if len(set(perm)) == G:
            mean_errs.append(max(d[a][perm[a]] for a in range(G)))
            sigma_errs.append(max(abs(float(est.comps.sigma2[a] - truth.comps.sigma2[perm[a]]))
                                  for a in range(G)))
    assert np.mean(aris) > 0.7
    assert min(aris) > 0.4
    assert np.mean(mean_errs) < 0.6
    assert np.mean(sigma_errs) < 0.25


def test_responsibilities_use_the_observed_density():
    """Section 4.1 scores u-hat_ig on the observed cells, not on the imputed matrix.

    With no missing cells the two coincide and the component score has to equal
    the plain matrix normal density, which pins the determinant identity
    log|Sigma_x| = log|Sigma| + log|Xi_z| at its boundary case.
    """
    p, q, G, n = 3, 4, 2, 20
    truth, Y, _, _ = _data(p, q, G, n, 0.0, 11)
    mask = jnp.zeros((n, p, q), dtype=bool)
    direct = float(sum(mp.matnorm_logpdf(Y[i], _component(truth, 0)) for i in range(n)))
    assert abs(float(mp.observed_loglik(Y, mask, _component(truth, 0))) - direct) < 1e-9
    got = mixture_loglik(Y, mask, truth)
    manual = 0.0
    for i in range(n):
        lg = [float(jnp.log(truth.pi[g]) + mp.matnorm_logpdf(Y[i], _component(truth, g)))
              for g in range(G)]
        manual += float(np.logaddexp(lg[0], lg[1]))
    assert abs(got - manual) < 1e-9


def test_pooled_imputation_is_a_responsibility_weighted_average():
    """Y-hat_i = sum_g u-hat_ig Y-hat_ig, and observed cells come back untouched."""
    p, q, G, n = 3, 4, 2, 40
    truth, Y, _, mask = _data(p, q, G, n, 0.3, 13)
    states = [mp.init_estep_state(Y, mask, _component(truth, g)) for g in range(G)]
    states, resp = mixture_estep(Y, mask, truth, states, method="exact")
    pooled = imputed_data(states, resp)
    obs = ~mask
    assert float(jnp.max(jnp.abs(jnp.where(obs, pooled - Y, 0.0)))) < 1e-10
    assert float(jnp.max(jnp.abs(jnp.sum(resp, axis=1) - 1.0))) < 1e-12
