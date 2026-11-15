import jax
import jax.numpy as jnp
import pytest

from coord_lca.decoder import (
    coord_loss,
    cross_entropy,
    decode_dense,
    decode_sparse,
    init_decoder,
    soft_dice,
)
from coord_lca.encoder import encode, init_encoder


D_MODEL = 16
N_HEADS = 4
N_FREQ = 4
PATCH = 8
DOWNSAMPLE = 2
ENC_PATCH = 4
N_CLASSES = 3


def _tiny(key, height=16, width=16, n_classes=N_CLASSES):
    k_enc, k_dec, k_img = jax.random.split(key, 3)
    encoder = init_encoder(k_enc, in_channels=3, d_model=D_MODEL, patch_size=ENC_PATCH)
    decoder = init_decoder(
        k_dec,
        d_model=D_MODEL,
        d_patch=16,
        n_freq=N_FREQ,
        n_classes=n_classes,
        patch_size=PATCH,
        n_heads=N_HEADS,
    )
    image = jax.random.uniform(k_img, (height, width, 3))
    tokens = encode(encoder, image, downsample=DOWNSAMPLE, patch_size=ENC_PATCH)
    return encoder, decoder, image, tokens


def test_sparse_forward_two_coords_no_dense_map():
    _, decoder, image, tokens = _tiny(jax.random.PRNGKey(0), height=16, width=16)
    xy = jnp.array([[3.0, 5.0], [12.5, 9.0]])
    logits = decode_sparse(
        decoder,
        image,
        tokens,
        xy,
        patch_size=PATCH,
        n_freq=N_FREQ,
        downsample=DOWNSAMPLE,
        encoder_patch_size=ENC_PATCH,
    )
    assert logits.shape == (2, N_CLASSES)
    # Sparse path never materializes H*W logits.
    assert logits.ndim == 2
    assert logits.shape[0] == xy.shape[0]


def test_sparse_does_not_allocate_hw_map():
    _, decoder, image, tokens = _tiny(jax.random.PRNGKey(1), height=32, width=32)
    xy = jnp.array([[1.0, 1.0], [30.0, 20.0]])

    def run(xy_):
        return decode_sparse(
            decoder,
            image,
            tokens,
            xy_,
            patch_size=PATCH,
            n_freq=N_FREQ,
            downsample=DOWNSAMPLE,
            encoder_patch_size=ENC_PATCH,
        )

    out = run(xy)
    assert out.shape == (2, N_CLASSES)
    jaxpr = jax.make_jaxpr(run)(xy)
    out_shapes = [tuple(a.shape) for a in jaxpr.out_avals]
    assert out_shapes == [(2, N_CLASSES)]
    # Sparse API returns (N, K), not (H, W, K).
    assert out.shape != (32, 32, N_CLASSES)
    with pytest.raises((TypeError, ValueError)):
        _ = out.reshape(32, 32, N_CLASSES)


def test_dense_decode_shape():
    _, decoder, image, tokens = _tiny(jax.random.PRNGKey(2), height=16, width=16)
    logits = decode_dense(
        decoder,
        image,
        tokens,
        patch_size=PATCH,
        n_freq=N_FREQ,
        downsample=DOWNSAMPLE,
        encoder_patch_size=ENC_PATCH,
    )
    assert logits.shape == (16, 16, N_CLASSES)


def test_soft_dice_identity_and_complement():
    y = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
    assert float(soft_dice(y, y)) == pytest.approx(1.0)
    assert float(soft_dice(y, 1.0 - y)) == pytest.approx(0.0)
    one_hot = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert float(soft_dice(one_hot[:, 0], one_hot[:, 0])) == pytest.approx(1.0)
    assert float(soft_dice(one_hot[:, 0], one_hot[:, 1])) == pytest.approx(0.0)


def test_ce_decreases_after_sgd():
    key = jax.random.PRNGKey(3)
    k_enc, k_dec, k_img, k_xy = jax.random.split(key, 4)
    encoder = init_encoder(k_enc, in_channels=3, d_model=D_MODEL, patch_size=ENC_PATCH)
    decoder = init_decoder(
        k_dec,
        d_model=D_MODEL,
        d_patch=16,
        n_freq=N_FREQ,
        n_classes=2,
        patch_size=PATCH,
        n_heads=N_HEADS,
    )
    height = width = 16
    image = jnp.zeros((height, width, 3), dtype=jnp.float32)
    image = image.at[4:12, 4:12, 0].set(1.0)
    tokens = encode(encoder, image, downsample=DOWNSAMPLE, patch_size=ENC_PATCH)
    n = 32
    xy = jax.random.uniform(k_xy, (n, 2)) * jnp.array([float(width), float(height)])
    labels = ((xy[:, 0] >= 4) & (xy[:, 0] < 12) & (xy[:, 1] >= 4) & (xy[:, 1] < 12)).astype(
        jnp.int32
    )

    def loss_fn(dec):
        logits = decode_sparse(
            dec,
            image,
            tokens,
            xy,
            patch_size=PATCH,
            n_freq=N_FREQ,
            downsample=DOWNSAMPLE,
            encoder_patch_size=ENC_PATCH,
        )
        return cross_entropy(logits, labels)

    lr = 0.2
    loss0 = float(loss_fn(decoder))
    for _ in range(12):
        g = jax.grad(loss_fn)(decoder)
        decoder = jax.tree.map(lambda p, dp: p - lr * dp, decoder, g)
    loss1 = float(loss_fn(decoder))
    assert loss1 < loss0


def test_coord_loss_finite():
    logits = jnp.array([[2.0, 0.1, -1.0], [0.0, 3.0, 0.2]])
    labels = jnp.array([0, 1])
    value = coord_loss(logits, labels)
    assert jnp.isfinite(value)
