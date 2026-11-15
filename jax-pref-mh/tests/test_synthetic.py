"""The Appx. C.1 instance itself, at full size.

Section 4.1 runs a 241-state chain; the kernel for that is a 241x241 matrix,
so the exactness claim can be checked on the paper's own setup rather than a
stand-in. Nothing here samples.
"""

import numpy as np
import pytest

from pref_mh.finite import (
    plugin_kernel,
    pref_mh_kernel,
    stationary,
    total_variation,
)
from pref_mh.synthetic import N_STATES, mixture_proposal, synthetic_model


def test_proposal_matches_appendix_c1():
    """gamma = 0.88 global uniform plus a lazy nearest-neighbour move, with
    boundary mass kept in place, and symmetric so the Hastings ratio cancels."""
    q = np.exp(mixture_proposal())
    assert q.shape == (N_STATES, N_STATES)
    assert np.allclose(q.sum(axis=0), 1.0)
    assert np.allclose(q, q.T)
    gamma, n = 0.88, N_STATES
    assert np.isclose(q[0, 0], gamma / n + 0.12 * 0.75)
    assert np.isclose(q[1, 0], gamma / n + 0.12 * 0.25)
    assert np.isclose(q[5, 5], gamma / n + 0.12 * 0.5)
    assert np.isclose(q[240, 240], gamma / n + 0.12 * 0.75)


def test_r0_is_the_base_density_ratio():
    """With q symmetric, r_0(k, k') = p_0(k')/p_0(k), as Appx. C.1 says."""
    model = synthetic_model()
    lp = np.asarray(model.log_p0)
    assert np.max(np.abs(np.asarray(model.log_r0()) - (lp[None, :] - lp[:, None]))) < 1e-12


@pytest.mark.parametrize("n_votes", [1, 2, 4])
def test_exact_on_the_paper_instance(n_votes):
    """pi @ P = pi and detailed balance on the full 241-state target."""
    model = synthetic_model()
    p = pref_mh_kernel(model, n_votes)
    pi = model.target()
    assert float(np.max(np.abs(np.asarray(p).sum(axis=1) - 1.0))) < 1e-12
    assert float(np.max(np.abs(np.asarray(pi @ p - pi)))) < 1e-14
    flow = np.asarray(pi)[:, None] * np.asarray(p)
    assert np.max(np.abs(flow - flow.T)) < 1e-15


@pytest.mark.parametrize("n_votes", [1, 2, 4])
def test_plugin_plateau_on_the_paper_instance(n_votes):
    """Figure 2's separation, read off the stationary distributions instead of
    a TV trace: Pref-MH sits on pi, the plug-in does not."""
    model = synthetic_model()
    pi = np.asarray(model.target())
    assert total_variation(stationary(pref_mh_kernel(model, n_votes)), pi) < 1e-10
    assert total_variation(stationary(plugin_kernel(model, n_votes)), pi) > 0.1
