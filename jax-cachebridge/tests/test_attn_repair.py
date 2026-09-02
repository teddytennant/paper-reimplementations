import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cachebridge import attn_repair as ar


def _qkv(key, g=3, m=24, d=16):
    kq, kk, kv = jax.random.split(key, 3)
    return (jax.random.normal(kq, (g, d)),
            jax.random.normal(kk, (m, d)),
            jax.random.normal(kv, (m, d)))


def test_eq14_linearization_error_is_second_order():
    """Eq 14 is the exact first-order term, so its residual must fall like eps^2."""
    key = jax.random.PRNGKey(0)
    q, k, v = _qkv(key)
    dk = jax.random.normal(jax.random.PRNGKey(1), k.shape)
    dv = jax.random.normal(jax.random.PRNGKey(2), v.shape)
    _, o = ar.attention(q, k, v)
    lin_unit = ar.first_order_delta(q, k, v, dk, dv)

    eps = np.array([2e-1, 1e-1, 5e-2, 2.5e-2])
    true_norm, err_norm = [], []
    for e in eps:
        _, o_pert = ar.attention(q, k + e * dk, v + e * dv)
        true = o_pert - o
        true_norm.append(float(jnp.linalg.norm(true)))
        err_norm.append(float(jnp.linalg.norm(true - e * lin_unit)))

    order_true = np.log2(np.array(true_norm[:-1]) / np.array(true_norm[1:]))
    order_err = np.log2(np.array(err_norm[:-1]) / np.array(err_norm[1:]))
    assert np.all(np.abs(order_true - 1.0) < 0.1), order_true
    assert np.all(np.abs(order_err - 2.0) < 0.15), order_err
    assert err_norm[0] < 0.2 * true_norm[0]


def test_eq14_matches_the_jacobian_vector_product():
    q, k, v = _qkv(jax.random.PRNGKey(3))
    dk = jax.random.normal(jax.random.PRNGKey(4), k.shape)
    dv = jax.random.normal(jax.random.PRNGKey(5), v.shape)

    def out(kk, vv):
        return ar.attention(q, kk, vv)[1]

    _, jvp = jax.jvp(out, (k, v), (dk, dv))
    assert jnp.max(jnp.abs(jvp - ar.first_order_delta(q, k, v, dk, dv))) < 1e-10


def test_sensitivities_match_their_definitions():
    """Eq 15 r^V = sum_u a_ui^2 and eq 16 r^K, checked against an explicit loop."""
    q, k, v = _qkv(jax.random.PRNGKey(6), g=4, m=11, d=8)
    a, o = ar.attention(q, k, v)
    r_k, r_v = ar.raw_sensitivities(q, k, v)
    d = q.shape[-1]
    for i in range(k.shape[0]):
        ref_v = sum(float(a[u, i]) ** 2 for u in range(q.shape[0]))
        ref_k = sum(float(a[u, i]) ** 2
                    * float(jnp.sum((v[i] - o[u]) ** 2))
                    * float(jnp.sum(q[u] ** 2)) / d
                    for u in range(q.shape[0]))
        assert abs(float(r_v[i]) - ref_v) < 1e-12
        assert abs(float(r_k[i]) - ref_k) < 1e-11
    assert jnp.all(r_k >= 0) and jnp.all(r_v >= 0)


def test_masked_positions_get_no_attention_and_no_sensitivity():
    q, k, v = _qkv(jax.random.PRNGKey(7), g=2, m=10, d=8)
    mask = jnp.arange(10) < 6
    a, _ = ar.attention(q, k, v, mask)
    r_k, r_v = ar.raw_sensitivities(q, k, v, mask)
    assert jnp.max(jnp.abs(a[:, 6:])) == 0.0
    assert jnp.allclose(a.sum(-1), 1.0, atol=1e-12)
    assert jnp.max(r_k[6:]) == 0.0 and jnp.max(r_v[6:]) == 0.0


def test_kish_closed_form_matches_eq18():
    r = ar.mean_normalize(jnp.abs(jax.random.normal(jax.random.PRNGKey(8), (200,))) + 0.05)
    c = ar.cv2(r)
    for alpha in (0.0, 0.25, 0.5, 1.0):
        w = ar.shrink(r, jnp.asarray(alpha))
        assert abs(float(ar.kish_neff(w)) - float(200 / (1 + alpha**2 * c))) < 1e-8


def test_neff_decreases_with_alpha():
    r = ar.mean_normalize(jnp.exp(jax.random.normal(jax.random.PRNGKey(9), (150,))))
    grid = jnp.linspace(0.0, 1.0, 21)
    neff = jnp.array([ar.kish_neff(ar.shrink(r, a)) for a in grid])
    assert jnp.all(jnp.diff(neff) < 0)
    assert abs(float(neff[0]) - 150.0) < 1e-9


def test_alpha_is_one_for_uniform_sensitivities():
    """Eq 19 first branch: CV^2(r) = 0 leaves the weights unshrunk."""
    r = jnp.ones(64)
    alpha = ar.optimal_alpha(r, ar.floor_tau(64, 8))
    assert float(alpha) == 1.0
    w, _ = ar.shrunk_weights(r, ar.floor_tau(64, 8))
    assert jnp.max(jnp.abs(w - 1.0)) < 1e-12


def test_alpha_is_one_when_the_floor_already_holds_at_full_weight():
    r = ar.mean_normalize(1.0 + 0.05 * jax.random.normal(jax.random.PRNGKey(10), (400,)))
    tau = ar.floor_tau(400, 20)   # 40, far below n / (1 + CV^2)
    assert float(ar.kish_neff(ar.shrink(r, jnp.asarray(1.0)))) >= tau
    assert float(ar.optimal_alpha(r, tau)) == 1.0


@pytest.mark.parametrize("p_local", [4, 30, 60, 200])
def test_floor_is_respected_for_concentrated_sensitivities(p_local):
    """The chosen alpha* keeps n_eff at or above tau_l = min(n, 2 p_l)."""
    n = 128
    r = ar.mean_normalize(jnp.exp(3.0 * jax.random.normal(jax.random.PRNGKey(11), (n,))))
    tau = ar.floor_tau(n, p_local)
    alpha = ar.optimal_alpha(r, tau)
    neff = float(ar.kish_neff(ar.shrink(r, alpha)))
    assert neff >= tau - 1e-8
    if float(alpha) < 1.0:
        assert abs(neff - tau) < 1e-8            # the constraint is active
        bigger = ar.shrink(r, jnp.minimum(alpha + 0.05, 1.0))
        assert float(ar.kish_neff(bigger)) < tau


def test_weights_are_nonnegative_and_mean_one():
    n = 256
    r = ar.mean_normalize(jnp.exp(2.0 * jax.random.normal(jax.random.PRNGKey(12), (n,))))
    w, alpha = ar.shrunk_weights(r, ar.floor_tau(n, 16))
    assert jnp.all(w >= 0)
    assert abs(float(jnp.mean(w)) - 1.0) < 1e-10
