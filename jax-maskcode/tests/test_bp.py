"""Sum-product identities: equation (4), cycle-free APP, arctanh cap."""

import jax
import jax.numpy as jnp
import numpy as np

from maskcode import bp, codes, ops


def test_stable_arctanh_caps_at_one_minus_two_eps():
    x = jnp.array(1.0)
    got = float(ops.stable_arctanh(x, 1e-7))
    expected = float(np.arctanh(1.0 - 2e-7))
    assert np.isfinite(got)
    assert abs(got - expected) < 1e-10


def test_check_to_variable_matches_equation_4_on_one_check():
    """One check of degree 3: mu_cv[i] = 2 arctanh(prod_{k != i} tanh(mu_vc[k]/2))."""
    mu_vc = jnp.array([[0.4, -1.2, 0.8, 0.0]], dtype=jnp.float64)
    mask = jnp.array([[True, True, True, False]])
    got = np.asarray(bp.check_to_variable(mu_vc, mask, eps=1e-12))
    t = np.tanh(np.array([0.4, -1.2, 0.8]) / 2.0)
    expected = np.zeros(4)
    for i in range(3):
        prod = np.prod([t[k] for k in range(3) if k != i])
        expected[i] = 2.0 * np.arctanh(prod)
    assert np.max(np.abs(got[0] - expected)) < 1e-12
    assert got[0, 3] == 0.0


def test_bp_on_a_tree_matches_exhaustive_app():
    """Cycle-free Tanner graph: sum-product after diameter iterations is bitwise MAP."""
    code = codes.tree_code()
    rng = np.random.default_rng(2)
    llr = jnp.asarray(rng.normal(scale=1.5, size=(16, code.n)))
    app = bp.app_decode_llr(llr, code.codewords())
    # Diameter of this tree is 6 edges; 8 iterations is enough.
    decoded = bp.decode(llr, code.h, n_iter=8)
    assert np.max(np.abs(np.asarray(decoded - app))) < 1e-5


def test_bp_on_spc_matches_exhaustive_app():
    code = codes.single_parity_check(4)
    rng = np.random.default_rng(3)
    llr = jnp.asarray(rng.normal(scale=1.2, size=(32, code.n)))
    app = bp.app_decode_llr(llr, code.codewords())
    decoded = bp.decode(llr, code.h, n_iter=4)
    assert np.max(np.abs(np.asarray(decoded - app))) < 1e-5


def test_arctanh_derivative_explodes_at_the_paper_clip():
    """Remark 1: d/dx arctanh((1-2eps)x) reaches ~2.5e6 at |x|=1 for eps=1e-7.

    That is the term the paper blames for gradient explosion when BP is unrolled.
    On the cycle-free codes here the messages saturate in one iteration, so the
    composed Jacobian of decode(., N_BP) does not keep growing with N_BP; the
    local derivative of the check-node map still does.
    """
    d = jax.grad(lambda z: ops.stable_arctanh(z, 1e-7))
    at_half = float(d(jnp.array(0.5)))
    # Interior: clip is the identity, so the derivative is the closed form.
    x = 0.999999
    got = float(d(jnp.array(x)))
    scale = 1.0 - 2e-7
    expected = scale / (1.0 - (scale * x) ** 2)
    assert abs(got - expected) / expected < 1e-8
    assert got > 1e5
    assert got / at_half > 1e5
    assert np.isfinite(float(ops.stable_arctanh(jnp.array(1.0), 1e-7)))
