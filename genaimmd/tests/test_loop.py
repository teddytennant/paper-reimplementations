"""One tiny GenAIMMD cycle on CPU (Sec. II E)."""

import jax
import jax.numpy as jnp

from genaimmd.committor import pB
from genaimmd.loop import init_state, run_cycle, run_loop


def test_tiny_cycle_finite():
    key = jax.random.PRNGKey(8)
    state = init_state(key, n_a=4, n_b=4, n_path=4)
    key, k = jax.random.split(key)
    out = run_cycle(
        state,
        k,
        n_committor_steps=3,
        n_bg_steps=3,
        n_relax=5,
        max_shoot_steps=20,
        batch_size=8,
    )
    assert jnp.isfinite(out["committor_loss"])
    assert jnp.isfinite(out["bg_loss"])
    assert out["samples"].shape[-1] == 2
    assert out["samples"].shape[0] >= 4
    pb = pB(out["committor_params"], out["samples"])
    assert jnp.all((pb > 0.0) & (pb < 1.0))


def test_tiny_run_loop():
    key = jax.random.PRNGKey(9)
    out = run_loop(
        key,
        n_cycles=1,
        n_a=3,
        n_b=3,
        n_path=3,
        n_committor_steps=2,
        n_bg_steps=2,
        n_relax=3,
        max_shoot_steps=10,
        batch_size=6,
    )
    assert jnp.isfinite(out["committor_loss"])
    assert out["samples"].ndim == 2
