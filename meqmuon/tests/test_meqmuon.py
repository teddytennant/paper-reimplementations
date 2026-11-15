"""Checks for the MeqMuon formulas, not just shapes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import meqmuon
from meqmuon import (
    coefficient_of_variation,
    column_normalize,
    column_rms,
    global_rms_normalize,
    init_momentum,
    meqmuon_step,
    newton_schulz,
    one_sided_normalize,
    row_normalize,
    row_rms,
    step_pytree,
)


def test_public_names():
    names = [
        "newton_schulz",
        "row_rms",
        "column_rms",
        "coefficient_of_variation",
        "row_normalize",
        "column_normalize",
        "global_rms_normalize",
        "one_sided_normalize",
        "meqmuon_step",
        "init_momentum",
        "step_pytree",
    ]
    for name in names:
        assert hasattr(meqmuon, name)


def test_zeros_short_circuit():
    h = jnp.zeros((3, 4))
    out = newton_schulz(h)
    assert out.shape == h.shape
    assert jnp.all(out == 0)

    v = jnp.zeros((5,))
    got = global_rms_normalize(v)
    assert got.shape == v.shape
    assert jnp.all(got == 0)

    x = jnp.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    y = row_normalize(x)
    assert jnp.all(y[0] == 0)
    assert not jnp.all(y[1] == 0)


def test_hand_computed_row_rms_and_cv():
    # Row 1 is exact zeros and must be excluded by delta.
    # r = [sqrt(1/3), 0, sqrt(25/3)]
    # positive mean = sqrt(3), population second moment = 4/3, CV = 2/3.
    x = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ]
    )
    expected_rms = np.array(
        [np.sqrt(1.0 / 3.0), 0.0, np.sqrt(25.0 / 3.0)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(row_rms(x), dtype=np.float64), expected_rms, rtol=1e-5, atol=1e-6)

    gamma = coefficient_of_variation(row_rms(x), delta=1e-7)
    np.testing.assert_allclose(float(gamma), 2.0 / 3.0, rtol=1e-5, atol=1e-6)

    # A row at the delta boundary is excluded. CV of the remaining single row is 0.
    tiny = jnp.array([[1e-7, 0.0], [2.0, 0.0]])
    assert float(coefficient_of_variation(row_rms(tiny), delta=1e-7)) == pytest.approx(0.0, abs=1e-6)


def test_normalize_sets_unit_rms():
    x = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.1],
            [1e-8, 0.0, 0.0],
        ]
    )
    original_rows = row_rms(x)
    y = row_normalize(x)
    new_rows = row_rms(y)
    for i in range(x.shape[0]):
        if float(original_rows[i]) > 1e-7:
            np.testing.assert_allclose(float(new_rows[i]), 1.0, rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_allclose(np.asarray(y[i]), np.asarray(x[i]), atol=1e-12)

    original_cols = column_rms(x)
    z = column_normalize(x)
    new_cols = column_rms(z)
    for j in range(x.shape[1]):
        if float(original_cols[j]) > 1e-7:
            np.testing.assert_allclose(float(new_cols[j]), 1.0, rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_allclose(np.asarray(z[:, j]), np.asarray(x[:, j]), atol=1e-12)


def test_one_sided_selection_and_tie():
    row_heavy = jnp.array(
        [
            [1.0, 10.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    g_row = float(coefficient_of_variation(row_rms(row_heavy)))
    g_col = float(coefficient_of_variation(column_rms(row_heavy)))
    assert g_row > g_col
    got = one_sided_normalize(row_heavy)
    assert jnp.allclose(got, row_normalize(row_heavy))
    assert not jnp.allclose(row_normalize(row_heavy), column_normalize(row_heavy))

    col_heavy = jnp.swapaxes(row_heavy, 0, 1)
    g_row = float(coefficient_of_variation(row_rms(col_heavy)))
    g_col = float(coefficient_of_variation(column_rms(col_heavy)))
    assert g_col > g_row
    got = one_sided_normalize(col_heavy)
    assert jnp.allclose(got, column_normalize(col_heavy))
    assert not jnp.allclose(got, row_normalize(col_heavy))

    # Equal CV takes the >= branch: row_normalize. The two normalizations differ.
    tie = jnp.array([[1.0, 1.0], [0.0, 0.0]])
    g_row = float(coefficient_of_variation(row_rms(tie)))
    g_col = float(coefficient_of_variation(column_rms(tie)))
    assert g_row == pytest.approx(g_col)
    assert g_row == pytest.approx(0.0)
    assert jnp.allclose(one_sided_normalize(tie), row_normalize(tie))
    assert not jnp.allclose(row_normalize(tie), column_normalize(tie))


def test_matrix_kind_is_row_then_column():
    grad = jnp.array([[1.0, 2.0], [3.0, 0.5]])
    row_then_col = column_normalize(row_normalize(grad))
    col_then_row = row_normalize(column_normalize(grad))
    assert not jnp.allclose(row_then_col, col_then_row)

    lr = 0.03
    rho = 0.2
    param = jnp.array([[0.4, -0.2], [0.1, 0.7]])
    new_param, new_momentum = meqmuon_step(
        param,
        grad,
        jnp.zeros_like(param),
        lr,
        "matrix",
        mu=0.0,
        rho=rho,
        weight_decay=0.0,
    )
    assert jnp.allclose(new_momentum, grad)
    expected = param - rho * lr * row_then_col
    assert jnp.allclose(new_param, expected)
    opposite = param - rho * lr * col_then_row
    assert not jnp.allclose(new_param, opposite)


def test_momentum_accumulation():
    g1 = jnp.array([1.0, 2.0, 3.0])
    g2 = jnp.array([0.5, -1.0, 0.25])
    mu = 0.95
    w = jnp.zeros((3,))
    m = jnp.zeros((3,))
    w1, m1 = meqmuon_step(w, g1, m, 0.01, "vector", mu=mu, weight_decay=0.0)
    _, m2 = meqmuon_step(w1, g2, m1, 0.01, "vector", mu=mu, weight_decay=0.0)
    expected = np.float32(mu) * np.asarray(g1) + np.asarray(g2)
    np.testing.assert_allclose(np.asarray(m1), np.asarray(g1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(m2), expected, rtol=1e-6, atol=1e-6)


def test_weight_decay_with_zero_grad():
    eta = 0.1
    lam = 0.1
    cases = {
        "hidden": jnp.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]]),
        "matrix": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "vector": jnp.array([1.0, -2.0, 0.5, 4.0]),
    }
    for kind, param in cases.items():
        new_param, new_momentum = meqmuon_step(
            param,
            jnp.zeros_like(param),
            jnp.zeros_like(param),
            eta,
            kind,
            weight_decay=lam,
        )
        np.testing.assert_allclose(
            np.asarray(new_param),
            (1.0 - eta * lam) * np.asarray(param),
            rtol=1e-6,
            atol=1e-6,
        )
        assert jnp.all(new_momentum == 0)


def test_hidden_uses_orthogonalization():
    grad = jnp.array(
        [
            [0.4, -0.2, 0.1],
            [0.0, 0.8, -0.3],
            [0.5, 0.1, 0.2],
            [-0.2, 0.3, 0.7],
        ]
    )
    rho = 0.2
    lr = 0.05
    param = jnp.zeros_like(grad)
    new_param, new_momentum = meqmuon_step(
        param,
        grad,
        jnp.zeros_like(grad),
        lr,
        "hidden",
        mu=0.0,
        rho=rho,
        weight_decay=0.0,
        nesterov=False,
    )
    assert jnp.allclose(new_momentum, grad)
    got = -new_param / (rho * lr)
    ortho = one_sided_normalize(newton_schulz(grad))
    raw = one_sided_normalize(grad)
    assert jnp.allclose(got, ortho, rtol=1e-5, atol=1e-5)
    assert not jnp.allclose(ortho, raw, rtol=1e-3, atol=1e-3)


def test_nesterov_hidden_only():
    momentum = jnp.array([[0.2, -0.1], [0.5, 0.3], [0.0, 0.4]])
    grad = jnp.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.2]])
    mu = 0.95
    rho = 0.2
    lr = 0.05
    param = jnp.zeros_like(grad)

    new_nest, mom_nest = meqmuon_step(
        param,
        grad,
        momentum,
        lr,
        "hidden",
        mu=mu,
        rho=rho,
        weight_decay=0.0,
        nesterov=True,
    )
    new_plain, mom_plain = meqmuon_step(
        param,
        grad,
        momentum,
        lr,
        "hidden",
        mu=mu,
        rho=rho,
        weight_decay=0.0,
        nesterov=False,
    )
    b_new = mu * momentum + grad
    assert jnp.allclose(mom_nest, b_new)
    assert jnp.allclose(mom_plain, b_new)

    h_nest = grad + mu * b_new
    h_plain = b_new
    dir_nest = one_sided_normalize(newton_schulz(h_nest))
    dir_plain = one_sided_normalize(newton_schulz(h_plain))
    assert jnp.allclose(-new_nest / (rho * lr), dir_nest, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(-new_plain / (rho * lr), dir_plain, rtol=1e-5, atol=1e-5)
    assert not jnp.allclose(dir_nest, dir_plain, rtol=1e-3, atol=1e-3)


def _numpy_newton_schulz(h, steps=5, coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7):
    """Independent recurrence: Frobenius scale, smaller-side Gram, then transpose back."""
    a, b, c = coefficients
    h = np.asarray(h, dtype=np.float64)
    fro = np.linalg.norm(h, ord="fro")
    if fro == 0.0:
        return np.zeros_like(h)
    x = h / (fro + eps)
    transposed = h.shape[0] > h.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x


def test_rectangular_and_square_newton_schulz():
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(0)
    k_tall, k_wide, k_square = jax.random.split(key, 3)

    tall = jax.random.normal(k_tall, (16, 4), dtype=jnp.float64)
    wide = jax.random.normal(k_wide, (4, 16), dtype=jnp.float64)
    square = jax.random.normal(k_square, (8, 8), dtype=jnp.float64)

    u_tall = newton_schulz(tall)
    u_wide = newton_schulz(wide)
    u_sq = newton_schulz(square)
    assert u_tall.dtype == jnp.float64
    assert u_wide.shape == (4, 16)
    assert u_sq.shape == (8, 8)

    # Default coefficients are the Muon quintic. Five steps must match that
    # recurrence, including the transpose that puts the Gram on the smaller side.
    np.testing.assert_allclose(np.asarray(u_tall), _numpy_newton_schulz(tall), atol=1e-10)
    np.testing.assert_allclose(np.asarray(u_wide), _numpy_newton_schulz(wide), atol=1e-10)
    np.testing.assert_allclose(np.asarray(u_sq), _numpy_newton_schulz(square), atol=1e-10)

    # Those coefficients keep singular values in a band around 1. They do not
    # finish at the polar factor, so the column Gram is not I at 1e-2.
    for mat, side in (
        (u_tall.T @ u_tall, 4),
        (u_wide @ u_wide.T, 4),
        (u_sq.T @ u_sq, 8),
    ):
        singular = np.linalg.svd(np.asarray(mat), compute_uv=False)
        assert np.all(singular > 0.5**2)
        assert np.all(singular < 1.5**2)
        assert np.max(np.abs(np.asarray(mat) - np.eye(side))) > 1e-2

    # The transpose-back is what selects the side. A convergent cubic, run long
    # enough to reach the polar factor, makes a tall matrix column-orthonormal
    # and a wide matrix row-orthonormal. Square hits both sides.
    cubic = (1.5, -0.5, 0.0)
    tall_orth = newton_schulz(tall, steps=8, coefficients=cubic)
    wide_orth = newton_schulz(wide, steps=8, coefficients=cubic)
    square_orth = newton_schulz(square, steps=20, coefficients=cubic)
    eye4 = np.eye(4)
    eye8 = np.eye(8)
    np.testing.assert_allclose(np.asarray(tall_orth.T @ tall_orth), eye4, atol=1e-2)
    np.testing.assert_allclose(np.asarray(wide_orth @ wide_orth.T), eye4, atol=1e-2)
    np.testing.assert_allclose(np.asarray(square_orth.T @ square_orth), eye8, atol=1e-2)
    np.testing.assert_allclose(np.asarray(square_orth @ square_orth.T), eye8, atol=1e-2)


def test_state_is_only_momentum():
    params = {
        "w": jnp.ones((2, 3)),
        "b": jnp.ones((4,)),
    }
    momentum = init_momentum(params)
    assert set(momentum) == {"w", "b"}
    assert momentum["w"].shape == (2, 3)
    assert momentum["b"].shape == (4,)
    assert jnp.all(momentum["w"] == 0)
    assert jnp.all(momentum["b"] == 0)

    grads = {
        "w": jnp.ones((2, 3)),
        "b": jnp.arange(4.0),
    }
    kinds = {"w": "matrix", "b": "vector"}
    new_params, new_momentum = step_pytree(params, grads, momentum, kinds, 0.01)
    assert set(new_params) == {"w", "b"}
    assert set(new_momentum) == {"w", "b"}
    assert new_params["w"].shape == (2, 3)
    assert new_momentum["b"].shape == (4,)

    single = jnp.ones((3, 3))
    single_m = init_momentum(single)
    new_single, new_single_m = step_pytree(single, jnp.ones((3, 3)), single_m, "hidden", 0.01)
    assert new_single.shape == (3, 3)
    assert new_single_m.shape == (3, 3)


def test_wrong_kind_or_ndim():
    matrix = jnp.ones((2, 2))
    vector = jnp.ones((3,))
    cube = jnp.ones((2, 2, 2))
    with pytest.raises(ValueError):
        meqmuon_step(matrix, matrix, matrix, 0.01, "embedding")
    with pytest.raises(ValueError):
        meqmuon_step(vector, vector, vector, 0.01, "hidden")
    with pytest.raises(ValueError):
        meqmuon_step(vector, vector, vector, 0.01, "matrix")
    with pytest.raises(ValueError):
        meqmuon_step(matrix, matrix, matrix, 0.01, "vector")
    with pytest.raises(ValueError):
        meqmuon_step(cube, cube, cube, 0.01, "hidden")
    with pytest.raises(ValueError):
        meqmuon_step(cube, cube, cube, 0.01, "vector")


def _run_steps(param, loss_fn, kind, steps=30):
    momentum = jnp.zeros_like(param)
    initial = loss_fn(param)
    for _ in range(steps):
        grad = jax.grad(loss_fn)(param)
        param, momentum = meqmuon_step(
            param,
            grad,
            momentum,
            0.05,
            kind,
            weight_decay=0.0,
            rho=0.2,
            mu=0.95,
        )
    final = loss_fn(param)
    return float(initial), float(final)


def test_least_squares_loss_decreases():
    key = jax.random.PRNGKey(1)
    kx, kw, kv, ky = jax.random.split(key, 4)
    x_mat = jax.random.normal(kx, (16, 4))
    w_true = jax.random.normal(kw, (4, 3))
    y_mat = x_mat @ w_true

    def loss_matrix(w):
        residual = x_mat @ w - y_mat
        return jnp.sum(residual * residual)

    initial, final = _run_steps(jnp.zeros((4, 3)), loss_matrix, "hidden")
    assert final < initial, f"hidden loss {initial} -> {final}"

    initial, final = _run_steps(jnp.zeros((4, 3)), loss_matrix, "matrix")
    assert final < initial, f"matrix loss {initial} -> {final}"

    x_vec = jax.random.normal(kv, (16, 4))
    w_vec = jax.random.normal(ky, (4,))
    y_vec = x_vec @ w_vec

    def loss_vector(w):
        residual = x_vec @ w - y_vec
        return jnp.sum(residual * residual)

    initial, final = _run_steps(jnp.zeros((4,)), loss_vector, "vector")
    assert final < initial, f"vector loss {initial} -> {final}"


def test_meqmuon_step_is_jittable():
    fn = jax.jit(meqmuon_step, static_argnames=("kind",))
    hidden_w = jnp.ones((3, 2))
    hidden_g = jnp.array([[0.2, -0.4], [0.1, 0.3], [-0.5, 0.2]])
    hidden_m = jnp.zeros((3, 2))
    jit_w, jit_m = fn(hidden_w, hidden_g, hidden_m, 0.01, "hidden")
    raw_w, raw_m = meqmuon_step(hidden_w, hidden_g, hidden_m, 0.01, "hidden")
    assert jit_w.shape == hidden_w.shape
    assert jit_m.shape == hidden_m.shape
    assert jnp.allclose(jit_w, raw_w)
    assert jnp.allclose(jit_m, raw_m)

    mat = jnp.ones((2, 3))
    jit_w, jit_m = fn(mat, mat, jnp.zeros_like(mat), 0.02, "matrix")
    assert jit_w.shape == mat.shape

    vec = jnp.arange(4.0)
    jit_w, jit_m = fn(vec, vec, jnp.zeros_like(vec), 0.02, "vector")
    assert jit_w.shape == vec.shape
    assert jnp.all(jnp.isfinite(jit_w))
