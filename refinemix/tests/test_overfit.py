import jax
import jax.numpy as jnp
import optax

from refinemix.algorithm import make_train_step
from refinemix.denoiser import init_denoiser, make_apply_fn
from refinemix.sampler import ancestral_sample


def test_overfit_single_sequence_loss_drops_and_sample_matches():
    mask_id = 4
    vocab = 5
    seq = jnp.array([1, 2, 0, 3], dtype=jnp.int32)
    length = seq.shape[0]
    params, cfg = init_denoiser(
        jax.random.PRNGKey(0),
        vocab_size=vocab,
        seq_len=length,
        d_model=32,
        n_heads=2,
        mask_id=mask_id,
    )
    apply_fn = make_apply_fn(cfg.n_heads, cfg.mask_id)
    logits = apply_fn(params, jnp.full((1, length), mask_id), 1.0)
    assert jnp.all(jnp.isneginf(logits[..., mask_id]))

    optimizer = optax.adamw(3e-3, weight_decay=0.0)
    opt_state = optimizer.init(params)
    step = make_train_step(apply_fn, optimizer, mask_id)
    batch = jnp.broadcast_to(seq, (32, length))
    source = jnp.zeros_like(batch)
    key = jax.random.PRNGKey(1)
    params, opt_state, loss0, _, _ = step(
        params, opt_state, batch, source, key, jnp.asarray(0.0), jnp.asarray(0.0)
    )
    loss0 = float(loss0)
    loss_f = loss0
    for _ in range(199):
        key, sub = jax.random.split(key)
        params, opt_state, loss_f, _, _ = step(
            params, opt_state, batch, source, sub, jnp.asarray(0.0), jnp.asarray(0.0)
        )
    loss_f = float(loss_f)
    assert loss_f < loss0

    samples = ancestral_sample(
        apply_fn,
        params,
        jax.random.PRNGKey(2),
        batch_size=8,
        seq_len=length,
        n_steps=16,
        mask_id=mask_id,
    )
    assert jnp.all(samples != mask_id)
    matched = jnp.any(jnp.all(samples == seq[None, :], axis=-1))
    assert bool(matched)
