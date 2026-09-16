import jax
import jax.numpy as jnp
import numpy as np

from coop_agents.data import linearly_separable_batch
from coop_agents.losses import combined_loss, confidence, group_loss, isolated_loss
from coop_agents.models import apply_mlp, count_params, mlp, softmax
from coop_agents.consensus import weighted_average
from coop_agents.train import init_opt_state, step


def test_isolated_step_drops_loss_on_linearly_separable_batch():
    x, y = linearly_separable_batch(n=32, dim=2)
    init, _ = mlp([2, 8, 2])
    params = init(jax.random.PRNGKey(3))
    agents = [params]
    opt = init_opt_state(agents, lr=0.5, kind="sgd", key=jax.random.PRNGKey(4))
    logits0 = apply_mlp(params, x)
    isol0 = float(jnp.mean(isolated_loss(softmax(logits0), y)))
    agents1, _, metrics = step(agents, (x, y), "isolated", 1.0, opt)
    isol1 = float(jnp.mean(metrics["isolated"]))
    logits1 = apply_mlp(agents1[0], x)
    isol1_direct = float(jnp.mean(isolated_loss(softmax(logits1), y)))
    assert isol1_direct < isol0
    assert isol1 <= isol0 + 1e-12


def test_stop_gradient_group_loss_does_not_flow_into_other_agents():
    init, apply = mlp([2, 4, 3])
    k0, k1, k2, kx = jax.random.split(jax.random.PRNGKey(7), 4)
    p0, p1, p2 = init(k0), init(k1), init(k2)
    x = jax.random.normal(kx, (8, 2))
    y = jax.nn.one_hot(jnp.zeros((8,), dtype=jnp.int32), 3)

    def loss_a0_wrt_a1(other):
        logits = jnp.stack(
            [apply(p0, x), apply(other, x), apply(p2, x)],
            axis=0,
        )
        probs = softmax(logits)
        w = confidence(probs)
        cons = weighted_average(jax.lax.stop_gradient(probs), jax.lax.stop_gradient(w))
        isol = jnp.mean(isolated_loss(probs[0], y))
        grp = jnp.mean(group_loss(probs[0], cons))
        return combined_loss(isol, grp, 1.0)

    grads = jax.grad(loss_a0_wrt_a1)(p1)
    for g in jax.tree.leaves(grads):
        assert jnp.allclose(g, 0.0), g

    def loss_without_stopgrad(other):
        logits = jnp.stack(
            [apply(p0, x), apply(other, x), apply(p2, x)],
            axis=0,
        )
        probs = softmax(logits)
        w = confidence(probs)
        cons = weighted_average(probs, w)
        isol = jnp.mean(isolated_loss(probs[0], y))
        grp = jnp.mean(group_loss(probs[0], cons))
        return combined_loss(isol, grp, 1.0)

    grads_flow = jax.grad(loss_without_stopgrad)(p1)
    max_abs = max(float(jnp.max(jnp.abs(g))) for g in jax.tree.leaves(grads_flow))
    assert max_abs > 1e-10


def test_three_small_mlps_have_fewer_params_than_one_large():
    init_s, _ = mlp([2, 8, 4])
    init_l, _ = mlp([2, 32, 4])
    small = init_s(jax.random.PRNGKey(0))
    large = init_l(jax.random.PRNGKey(1))
    n_small = 3 * count_params(small)
    n_large = count_params(large)
    assert n_small < n_large
    assert n_small > 0


def test_step_weighted_average_runs_and_returns_metrics():
    init, _ = mlp([2, 8, 4])
    keys = jax.random.split(jax.random.PRNGKey(11), 4)
    agents = [init(k) for k in keys[:3]]
    x = jax.random.normal(keys[3], (16, 2))
    y = jax.nn.one_hot(jnp.arange(16) % 4, 4)
    opt = init_opt_state(agents, lr=0.05, kind="sgd")
    new_agents, new_opt, metrics = step(agents, (x, y), "weighted_average", 1.0, opt)
    assert len(new_agents) == 3
    assert metrics["isolated"].shape == (3,)
    assert new_opt["t"] == 1
