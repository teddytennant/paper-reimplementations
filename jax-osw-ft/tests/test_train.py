"""Algorithm 1 end to end on the toy task.

The behavioural claims here are the ones that survive a seed change. There is no
"OSW-FT reaches a lower loss than SFT" assertion; comparing two stochastic runs
on one seed measures the seed.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from osw_ft import train as train_module
from osw_ft.train import TrainConfig, compute_weights, evaluate, train

SEEDS = [0, 1, 2]


def _small_config(**kwargs):
    base = dict(steps=250, batch_size=6, K=4, lr=3e-3, n_digits=3, eval_every=50)
    base.update(kwargs)
    return TrainConfig(**base)


@pytest.fixture(scope="module")
def runs():
    """One OSW-FT run per seed, shared by the assertions below."""
    out = {}
    for seed in SEEDS:
        cfg = _small_config(seed=seed)
        out[seed] = (cfg, train(cfg))
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_weight_decays_on_mastered_queries(seed, runs):
    """The weight is the point: once the model can solve a query, w -> 0.

    p_hat -> 1 on a mastered query, so w = 1 - p_hat collapses and OSW-FT stops
    spending gradient on it. Nine queries, 250 steps, and the model memorizes
    them, so this holds on every seed rather than on a lucky one.
    """
    _cfg, history = runs[seed]
    assert history["mean_weight"][0] >= 0.9
    assert history["mean_weight"][-1] <= 0.15
    assert history["mean_weight"][-1] < history["mean_weight"][0]


@pytest.mark.parametrize("seed", SEEDS)
def test_success_rate_rises_and_the_task_is_learned(seed, runs):
    """The exact pass rate over the whole dataset, not a rollout estimate."""
    cfg, history = runs[seed]
    assert history["mean_p_s"][0] < 0.2
    assert history["mean_p_s"][-1] >= 0.8
    assert evaluate(history["params"], cfg) >= 0.8


@pytest.mark.parametrize("seed", SEEDS)
def test_second_moment_stays_below_sft(seed, runs):
    """eq (11): Sigma^2 < Sigma^2_SFT at every logged step.

    Measured from the true p_s and the real ||g_0||^2 of the batch, so this is
    the paper's inequality on actual gradients rather than on a chosen p grid.
    Softmax gives p_s > 0 everywhere, which is exactly the positive-measure
    condition Corollary 1 asks for.
    """
    _cfg, history = runs[seed]
    for sigma2, sigma2_sft in zip(history["sigma2"], history["sigma2_sft"]):
        assert sigma2 < sigma2_sft


def test_weights_refresh_every_s_steps(monkeypatch):
    """Theorem 2's S: rollouts happen once per outer iteration, not once per step.

    Weights are held fixed for the S inner steps, which is what Definition 1's
    surrogate J^(t) freezes and what the B * eps_p term in eq (10) pays for.
    """
    calls = []
    original = train_module.compute_weights

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "compute_weights", counted)
    train(_small_config(steps=20, refresh_every=5, eval_every=0))
    assert len(calls) == 4

    calls.clear()
    train(_small_config(steps=20, refresh_every=1, eval_every=0))
    assert len(calls) == 20


def test_sft_mode_uses_uniform_weights():
    """eq (1): the SFT baseline is the same loop with w identically 1."""
    cfg = _small_config()
    params = train_module.init_params(jr.PRNGKey(0), cfg.model)
    batch = jnp.asarray(train_module.build_dataset(3))
    w = compute_weights(params, cfg.model, jr.PRNGKey(1), batch, cfg.K, 1.0, "sft")
    assert jnp.array_equal(w, jnp.ones((batch.shape[0],)))


@pytest.mark.parametrize("K", [1, 2, 4])
def test_rollout_weights_take_only_the_k_plus_one_allowed_values(K):
    """w = 1 - m/K for integer m, so the granularity of the weight is set by K.

    Remark 1's point in code: K = 1 gives a binary weight, K = 2 is the first
    value that can express 0.5.
    """
    cfg = _small_config(K=K)
    params = train_module.init_params(jr.PRNGKey(0), cfg.model)
    batch = jnp.asarray(train_module.build_dataset(3))
    w = compute_weights(params, cfg.model, jr.PRNGKey(1), batch, K, 1.0, "osw")
    allowed = jnp.asarray([1.0 - m / K for m in range(K + 1)])
    assert bool(jnp.all(jnp.any(jnp.isclose(w[:, None], allowed[None, :]), axis=1)))


@pytest.mark.parametrize("mode", ["hard", "random", "sft"])
def test_ablation_modes_run(mode):
    """The section 4.4 controls go through the same loop."""
    history = train(_small_config(steps=20, weight_mode=mode, eval_every=10))
    assert len(history["loss"]) >= 2
    assert all(0.0 <= w <= 1.0 for w in history["mean_weight"])


def test_unknown_weight_mode_is_rejected():
    cfg = _small_config()
    params = train_module.init_params(jr.PRNGKey(0), cfg.model)
    batch = jnp.asarray(train_module.build_dataset(3))
    with pytest.raises(ValueError, match="unknown weight_mode"):
        compute_weights(params, cfg.model, jr.PRNGKey(0), batch, 2, 1.0, "nonsense")
