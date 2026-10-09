import jax
import jax.numpy as jnp
import numpy as np

from refinemix.algorithm import training_step
from refinemix.schedule import EPS


def _apply(params, tokens, t):
    batch, length = tokens.shape
    return jnp.zeros((batch, length, 5), dtype=jnp.float32)


def test_training_step_respects_caller_sampler_and_cutoff():
    batch, length = 16, 4
    target = jnp.zeros((batch, length), dtype=jnp.int32)
    source = jnp.ones((batch, length), dtype=jnp.int32)

    def sampler(key, shape):
        del key
        half = shape[0] // 2
        return jnp.concatenate(
            [jnp.full((half,), 0.2), jnp.full((shape[0] - half,), 0.9)]
        )

    out = training_step(
        {"bias": jnp.zeros(())},
        target,
        source,
        jax.random.PRNGKey(0),
        tmax=0.5,
        lam=0.4,
        apply_fn=_apply,
        mask_id=4,
        time_sampler=sampler,
    )
    t = np.asarray(out["t"])
    domain = np.asarray(out["domain"])
    x0 = np.asarray(out["x0"])
    assert np.allclose(t[:8], 0.2)
    assert np.allclose(t[8:], 0.9)
    high = t > 0.5
    assert np.any(high)
    assert np.all(domain[high] == 0)
    assert np.all(x0[high] == np.asarray(target)[high])
    assert np.all(x0[domain == 1] == np.asarray(source)[domain == 1])
    assert np.all(x0[domain == 0] == np.asarray(target)[domain == 0])
    assert set(out["domain_name"][high]).issubset({"target"})


def test_default_sampler_is_experimental():
    batch, length = 64, 3
    target = jnp.zeros((batch, length), dtype=jnp.int32)
    source = jnp.ones((batch, length), dtype=jnp.int32)
    out = training_step(
        {"bias": jnp.zeros(())},
        target,
        source,
        jax.random.PRNGKey(4),
        tmax=0.5,
        lam=0.5,
        apply_fn=_apply,
        mask_id=4,
    )
    t = np.asarray(out["t"])
    assert np.all(t >= EPS - 1e-7)
    assert np.all(t < 1.0)
    high = t > 0.5
    assert np.any(high)
    assert np.all(np.asarray(out["domain"])[high] == 0)
    assert np.all(np.asarray(out["x0"])[high] == 0)
