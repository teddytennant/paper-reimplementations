import jax
import jax.numpy as jnp

from adamx.opt import apply_updates, init, update


def test_lam0_first_step_equals_bias_corrected_amsgrad():
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grads = {"w": jnp.array([0.5, -1.0, 0.25])}
    lr, b1, b2, eps = 1e-3, 0.9, 0.999, 1e-8
    state = init(params, lr=lr, b1=b1, b2=b2, eps=eps, lam=0.0)
    updates, new_state = update(grads, state)
    t = 1
    m = (1.0 - b1) * grads["w"]
    v = (1.0 - b2) * grads["w"] ** 2
    m_hat = m / (1.0 - b1 ** t)
    v_hat = v / (1.0 - b2 ** t)
    expected = -lr * m_hat / (jnp.sqrt(v_hat) + eps)
    assert jnp.allclose(updates["w"], expected, atol=1e-7)
    assert jnp.allclose(new_state["v_tilde"]["w"], v_hat, atol=1e-7)


def test_first_step_gamma_is_one_even_if_lam_nonzero():
    params = jnp.array([0.0, 0.0, 0.0])
    grads = jnp.array([1.0, -2.0, 0.5])
    s0 = init(params, lr=0.01, lam=0.0)
    s1 = init(params, lr=0.01, lam=0.8)
    u0, n0 = update(grads, s0)
    u1, n1 = update(grads, s1)
    assert int(n0["count"]) == 1
    assert int(n1["count"]) == 1
    assert jnp.allclose(u0, u1, atol=1e-7)


def test_v_tilde_coordinatewise_nondecreasing():
    params = jnp.zeros((5,))
    state = init(params, lam=0.2)
    key = jax.random.PRNGKey(1)
    prev = state["v_tilde"]
    for _ in range(8):
        key, k = jax.random.split(key)
        g = jax.random.normal(k, (5,))
        _, state = update(g, state)
        assert jnp.all(state["v_tilde"] + 1e-12 >= prev)
        prev = state["v_tilde"]


def test_aligned_g_prev_has_larger_update_than_anti_aligned():
    g = jnp.array([1.0, 2.0, 3.0])
    params = jnp.zeros_like(g)
    base = init(params, lr=0.1, lam=0.5)
    aligned = {**base, "count": 1, "g_prev": g}
    anti = {**base, "count": 1, "g_prev": -g}
    u_al, _ = update(g, aligned)
    u_anti, _ = update(g, anti)
    assert jnp.linalg.norm(u_al) > jnp.linalg.norm(u_anti)


def test_analytic_b1_b2_zero():
    g = jnp.array([3.0, 4.0])
    params = jnp.zeros_like(g)
    state = init(params, lr=0.1, b1=0.0, b2=0.0, eps=1e-8, lam=0.0)
    updates, new_state = update(g, state)
    assert jnp.allclose(new_state["m"], g, atol=1e-7)
    assert jnp.allclose(new_state["v"], g ** 2, atol=1e-7)
    assert jnp.allclose(new_state["v_tilde"], g ** 2, atol=1e-7)
    expected = -0.1 * g / (jnp.abs(g) + 1e-8)
    assert jnp.allclose(updates, expected, atol=1e-7)


def test_quadratic_loss_drops_over_50_steps():
    target = jnp.array([1.0, -2.0, 0.5])
    params = jnp.zeros_like(target)
    state = init(params, lr=0.1, lam=0.1)

    def loss(x):
        return 0.5 * jnp.sum((x - target) ** 2)

    before = float(loss(params))
    for _ in range(50):
        grads = params - target
        updates, state = update(grads, state)
        params = apply_updates(params, updates)
    after = float(loss(params))
    assert after < before
