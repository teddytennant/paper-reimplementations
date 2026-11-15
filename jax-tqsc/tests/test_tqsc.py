import jax
import jax.numpy as jnp
import numpy as np

from jax_tqsc.data import make_synthetic_dataset
from jax_tqsc.model import TQSCConfig, count_params, forward, init_tqsc
from jax_tqsc.physics import cholesky_rho, fidelity, frobenius_mse, maximally_mixed, rho_from_chol
from jax_tqsc.train import evaluate, train_tqsc


def _assert_physical(rho, atol=1e-5):
    herm = rho - jnp.conjugate(jnp.moveaxis(rho, -1, -2))
    assert float(jnp.max(jnp.abs(herm))) < atol
    tr = jnp.real(jnp.trace(rho, axis1=-2, axis2=-1))
    assert jnp.allclose(tr, 1.0, atol=1e-4)
    evals = jnp.linalg.eigvalsh(rho)
    assert float(jnp.min(evals)) >= -1e-5


def test_cholesky_reconstruction_matches_s51():
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.normal(size=(32, 4)).astype(np.float32))
    rho_c = rho_from_chol(r)
    rho_s = cholesky_rho(r)
    assert rho_c.shape == (32, 2, 2)
    assert jnp.issubdtype(rho_c.dtype, jnp.complexfloating)
    assert jnp.allclose(rho_c, rho_s, atol=1e-5)
    _assert_physical(rho_c)


def test_fidelity_identity_and_depolarizing():
    rng = np.random.default_rng(1)
    z = rng.normal(size=2) + 1j * rng.normal(size=2)
    psi = z / np.linalg.norm(z)
    pure = jnp.asarray((np.outer(psi, np.conjugate(psi))).astype(np.complex64))
    f_id = fidelity(pure, pure)
    assert jnp.allclose(f_id, 1.0, atol=1e-5)

    r = jnp.asarray(rng.normal(size=(8, 4)).astype(np.float32))
    rho = rho_from_chol(r)
    assert jnp.allclose(fidelity(rho, rho), 1.0, atol=1e-4)

    eye = jnp.eye(2, dtype=pure.dtype) * 0.5
    ps = jnp.array([0.0, 0.25, 0.5, 0.8], dtype=jnp.float32)
    mixed = (1.0 - ps)[:, None, None] * pure + ps[:, None, None] * eye
    f = fidelity(jnp.broadcast_to(pure, mixed.shape), mixed)
    assert jnp.all(f[1:] < f[:-1])
    assert jnp.allclose(f[0], 1.0, atol=1e-5)


def test_forward_shapes_and_attention():
    key = jax.random.PRNGKey(2)
    params = init_tqsc(key)
    m = jax.random.normal(jax.random.PRNGKey(3), (7, 13))
    rho, attn = forward(params, m, return_attn=True)
    assert rho.shape == (7, 2, 2)
    assert jnp.issubdtype(rho.dtype, jnp.complexfloating)
    assert attn.shape == (7, 4, 13, 13)
    assert jnp.allclose(attn.sum(axis=-1), 1.0, atol=1e-5)


def test_random_init_is_physical():
    key = jax.random.PRNGKey(4)
    params = init_tqsc(key)
    m = jax.random.normal(jax.random.PRNGKey(5), (16, 13))
    rho = forward(params, m)
    _assert_physical(rho)


def test_param_count_in_paper_range():
    params = init_tqsc(jax.random.PRNGKey(6), TQSCConfig())
    n = count_params(params)
    assert 50_000 <= n <= 250_000, n
    assert abs(n - 152_292) / 152_292 < 0.20


def test_tiny_training_beats_mixed_and_init():
    train = make_synthetic_dataset(192, seed=10)
    test = make_synthetic_dataset(64, seed=11)
    key = jax.random.PRNGKey(7)
    params0 = init_tqsc(key)
    init_stats = evaluate(params0, test["M"], test["rho"])
    params, log = train_tqsc(
        jax.random.PRNGKey(8),
        train["M"],
        train["rho"],
        steps=250,
        batch_size=32,
        lr=1e-3,
        params=params0,
    )
    after = evaluate(params, test["M"], test["rho"])
    mixed_f = after["mixed_fidelity"]
    assert after["fidelity"] > mixed_f
    assert after["fidelity"] > init_stats["fidelity"] or after["mse"] < init_stats["mse"]
    assert log["loss"][-1] < log["loss"][0]
    assert after["fidelity"] > 0.55
