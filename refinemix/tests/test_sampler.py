import jax
import jax.numpy as jnp

from refinemix.sampler import ancestral_sample, masked_reverse_sample


def test_reverse_step_copies_unmasked_and_final_step_reveals():
    mask_id = 4
    vocab = 5
    tokens = jnp.array([[1, mask_id, 0, mask_id]], dtype=jnp.int32)
    logits = jnp.full((1, 4, vocab), -20.0)
    logits = logits.at[..., 2].set(20.0)
    out = masked_reverse_sample(
        jax.random.PRNGKey(0), tokens, logits, alpha_s=1.0, alpha_t=0.4, mask_id=mask_id
    )
    assert int(out[0, 0]) == 1
    assert int(out[0, 2]) == 0
    assert int(out[0, 1]) == 2
    assert int(out[0, 3]) == 2
    assert jnp.all(out != mask_id)


def test_ancestral_sampler_never_emits_mask():
    mask_id = 3
    vocab = 4

    def apply_fn(params, tokens, t):
        del params, t
        batch, length = tokens.shape
        logits = jnp.zeros((batch, length, vocab))
        logits = logits.at[..., 0].set(15.0)
        return logits

    samples = ancestral_sample(
        apply_fn,
        {},
        jax.random.PRNGKey(0),
        batch_size=4,
        seq_len=5,
        n_steps=8,
        mask_id=mask_id,
    )
    assert samples.shape == (4, 5)
    assert jnp.all(samples == 0)
    assert jnp.all(samples != mask_id)
