"""Train a toy CoordFormer decoder on synthetic 32x32 squares."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from coord_lca.decoder import cross_entropy, decode_sparse, init_decoder
from coord_lca.encoder import encode, init_encoder


HEIGHT = 32
WIDTH = 32
D_MODEL = 32
N_HEADS = 4
N_FREQ = 6
PATCH = 8
DOWNSAMPLE = 2
ENC_PATCH = 4
N_CLASSES = 2
N_QUERIES = 64
STEPS = 200
LR = 0.05
BATCH = 4


def make_square(key: jax.Array) -> tuple[jnp.ndarray, tuple[int, int, int, int]]:
    """Colored square on a dark background; returns image and (y0, y1, x0, x1)."""
    k_pos, k_col, k_bg = jax.random.split(key, 3)
    image = 0.1 * jax.random.uniform(k_bg, (HEIGHT, WIDTH, 3))
    size = 12
    y0 = int(jax.random.randint(k_pos, (), 2, HEIGHT - size - 2))
    x0 = int(jax.random.randint(jax.random.fold_in(k_pos, 1), (), 2, WIDTH - size - 2))
    y1, x1 = y0 + size, x0 + size
    color = 0.4 + 0.6 * jax.random.uniform(k_col, (3,))
    image = image.at[y0:y1, x0:x1, :].set(color)
    return image, (y0, y1, x0, x1)


def labels_at(xy: jnp.ndarray, box: tuple[int, int, int, int]) -> jnp.ndarray:
    y0, y1, x0, x1 = box
    inside = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
    return inside.astype(jnp.int32)


def main() -> None:
    key = jax.random.PRNGKey(0)
    k_enc, k_dec, key = jax.random.split(key, 3)
    encoder = init_encoder(k_enc, in_channels=3, d_model=D_MODEL, patch_size=ENC_PATCH)
    decoder = init_decoder(
        k_dec,
        d_model=D_MODEL,
        d_patch=32,
        n_freq=N_FREQ,
        n_classes=N_CLASSES,
        patch_size=PATCH,
        n_heads=N_HEADS,
    )

    def batch_loss(dec, images, tokens, xy, labels):
        def one(img, tok, q, y):
            logits = decode_sparse(
                dec,
                img,
                tok,
                q,
                patch_size=PATCH,
                n_freq=N_FREQ,
                downsample=DOWNSAMPLE,
                encoder_patch_size=ENC_PATCH,
            )
            return cross_entropy(logits, y), logits

        losses, logits = jax.vmap(one)(images, tokens, xy, labels)
        return jnp.mean(losses), logits

    def sample(key_s):
        keys = jax.random.split(key_s, BATCH)
        images = []
        tokens = []
        queries = []
        labs = []
        for i, k in enumerate(keys):
            k_img, k_xy = jax.random.split(k)
            img, box = make_square(k_img)
            tok = encode(encoder, img, downsample=DOWNSAMPLE, patch_size=ENC_PATCH)
            xy = jax.random.uniform(k_xy, (N_QUERIES, 2)) * jnp.array(
                [float(WIDTH), float(HEIGHT)]
            )
            images.append(img)
            tokens.append(tok)
            queries.append(xy)
            labs.append(labels_at(xy, box))
        return (
            jnp.stack(images),
            jnp.stack(tokens),
            jnp.stack(queries),
            jnp.stack(labs),
        )

    key_data, key = jax.random.split(key)
    images0, tokens0, xy0, labels0 = sample(key_data)
    loss_fn = lambda dec, im, to, q, y: batch_loss(dec, im, to, q, y)[0]
    loss_before = float(loss_fn(decoder, images0, tokens0, xy0, labels0))
    _, logits0 = batch_loss(decoder, images0, tokens0, xy0, labels0)
    pred0 = jnp.argmax(logits0, axis=-1)
    acc_before = float(jnp.mean((pred0 == labels0).astype(jnp.float32)))

    @jax.jit
    def step(dec, im, to, q, y):
        def l(d):
            return loss_fn(d, im, to, q, y)

        loss, g = jax.value_and_grad(l)(dec)
        dec = jax.tree.map(lambda p, dp: p - LR * dp, dec, g)
        return dec, loss

    for t in range(STEPS):
        key, k_b = jax.random.split(key)
        im, to, q, y = sample(k_b)
        decoder, _ = step(decoder, im, to, q, y)

    loss_after = float(loss_fn(decoder, images0, tokens0, xy0, labels0))
    _, logits1 = batch_loss(decoder, images0, tokens0, xy0, labels0)
    pred1 = jnp.argmax(logits1, axis=-1)
    acc_after = float(jnp.mean((pred1 == labels0).astype(jnp.float32)))

    print(f"loss_before={loss_before:.4f} acc_before={acc_before:.4f}")
    print(f"loss_after={loss_after:.4f} acc_after={acc_after:.4f}")
    if loss_after >= loss_before:
        raise SystemExit("smoke failed: loss did not drop")


if __name__ == "__main__":
    main()
