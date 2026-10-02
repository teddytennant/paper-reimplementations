import jax
import jax.numpy as jnp

from glfq import block_glf_loss, glf_loss, predict, quantize_symmetric, reconstruct


def _mlp(seed=0, hidden=8, batch=16):
    key = jax.random.PRNGKey(seed)
    kx, k0, kb0, k1, kb1 = jax.random.split(key, 5)
    x0 = jax.random.normal(kx, (batch, hidden))
    blocks = [
        (jax.random.normal(k0, (hidden, hidden)) * 0.4, jax.random.normal(kb0, (hidden,)) * 0.1),
        (jax.random.normal(k1, (hidden, hidden)) * 0.4, jax.random.normal(kb1, (hidden,)) * 0.1),
    ]
    return blocks, x0


def test_quantize_grid_and_ste_grad():
    weight = jnp.array([[0.2, -1.5, 0.7], [0.0, 0.4, -0.1]], dtype=jnp.float32)
    hat = quantize_symmetric(weight, bits=4, axis=-1)
    qmax = 7.0
    scale = jnp.max(jnp.abs(weight), axis=-1, keepdims=True) / qmax
    codes = hat / scale
    assert jnp.max(jnp.abs(codes - jnp.round(codes))) < 1e-4
    assert jnp.max(jnp.abs(codes)) <= qmax + 1e-4

    def loss(w):
        return jnp.sum(quantize_symmetric(w, bits=4, axis=-1))

    grad = jax.grad(loss)(weight)
    assert jnp.any(grad != 0)


def test_glf_loss_normalizes_and_guards_zero():
    assert float(glf_loss(2.0, 4.0, 2.0, 4.0)) == 2.0
    assert float(glf_loss(3.0, 5.0, 0.0, 0.0)) == 8.0


def test_reconstruct_reduces_each_block_loss():
    blocks, x0 = _mlp()
    quantized = reconstruct(blocks, x0, seed=0)
    prefix = []
    for index, (weight, _bias) in enumerate(blocks):
        init = block_glf_loss(blocks, x0, prefix, index, weight)
        final = block_glf_loss(blocks, x0, prefix, index, quantized[index], quantize=False)
        assert float(final) < float(init)
        prefix.append(quantized[index])


def test_output_shapes_match_fp():
    blocks, x0 = _mlp()
    quantized = reconstruct(blocks, x0, seed=0)
    fp = predict(blocks, x0)
    q_blocks = [(weight, bias) for weight, (_orig, bias) in zip(quantized, blocks)]
    out = predict(q_blocks, x0)
    assert fp.shape == out.shape == (16, 8)
    assert all(q.shape == w.shape for q, (w, _b) in zip(quantized, blocks))


def test_global_term_is_live():
    blocks, x0 = _mlp()
    with_global = reconstruct(blocks, x0, seed=0)
    local_only = reconstruct(blocks, x0, seed=0, use_global=False)
    assert not all(jnp.allclose(a, b) for a, b in zip(with_global, local_only))


def test_reconstruct_is_deterministic():
    blocks, x0 = _mlp()
    a = reconstruct(blocks, x0, seed=0)
    b = reconstruct(blocks, x0, seed=0)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b))
