import jax.numpy as jnp
import numpy as np

from refinemix.schedule import EPS, alpha, alpha_ideal, alpha_prime, nelbo_weight, sample_time
import jax


def test_alpha_endpoints_and_constant_derivative():
    assert abs(float(alpha(0.0)) - 1.0) < 1e-6
    assert abs(float(alpha(1.0)) - EPS) < 1e-6
    ts = jnp.linspace(0.0, 1.0, 9)
    deriv = alpha_prime(ts)
    assert np.allclose(np.asarray(deriv), -(1.0 - EPS), atol=1e-6)
    assert float(jnp.max(deriv) - jnp.min(deriv)) == 0.0 or np.allclose(deriv, deriv[0])


def test_weight_equals_negative_inverse_t():
    # Away from t = 0, where the weight is singular.
    ts = np.array([1e-3, 0.01, 0.2, 0.5, 0.8, 1.0], dtype=np.float64)
    # Algebraic identity for the log-linear schedule: the (1 - eps) factors cancel.
    eps = EPS
    alpha_t = 1.0 - (1.0 - eps) * ts
    alpha_p = -(1.0 - eps)
    assert np.allclose(alpha_p / (1.0 - alpha_t), -1.0 / ts, atol=1e-12)
    # Idealized alpha(t) = 1 - t has the same closed form.
    ideal_alpha = 1.0 - ts
    assert np.allclose(-1.0 / (1.0 - ideal_alpha), -1.0 / ts, atol=1e-12)
    weight = np.asarray(nelbo_weight(jnp.asarray(ts)), dtype=np.float64)
    assert np.allclose(weight, -1.0 / ts, rtol=1e-5, atol=1e-4)
    assert np.all(weight < 0.0)


def test_ideal_schedule_is_distinct_from_experimental():
    assert float(alpha(1.0, ideal=True)) == 0.0
    assert float(alpha_ideal(1.0)) == 0.0
    assert float(alpha_ideal(0.0)) == 1.0
    assert abs(float(alpha(1.0)) - EPS) < 1e-6
    assert float(alpha_prime(0.4, ideal=True)) == -1.0


def test_time_samplers_cover_both_intervals():
    key = jax.random.PRNGKey(0)
    experimental = np.asarray(sample_time(key, (10000,), mode="experimental"))
    algorithm = np.asarray(sample_time(key, (10000,), mode="algorithm"))
    assert experimental.min() >= EPS - 1e-7
    assert experimental.max() <= 1.0
    assert algorithm.min() >= 0.0
    assert algorithm.max() <= 1.0
    assert algorithm.min() < EPS
