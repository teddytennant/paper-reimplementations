"""Train a tiny reverse denoiser on synthetic 4x4 discrete blobs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qd_dtqw.losses import histogram_kl, intensity_histogram
from qd_dtqw.reverse import generate, init_reverse, sgd_step
from qd_dtqw.walk import DTQW


K = 4
H = W = 4
T = 6
N_IMAGES = 8
STEPS = 80


def blob_images(key, n: int = N_IMAGES):
    """Random 2x2 squares of intensity K-1 on a zero background."""
    imgs = np.zeros((n, H, W), dtype=np.int32)
    keys = jax.random.split(key, n)
    for i, k in enumerate(keys):
        r, c = jax.random.randint(k, (2,), 0, H - 1)
        r, c = int(r), int(c)
        imgs[i, r : r + 2, c : c + 2] = K - 1
    return jnp.asarray(imgs)


def make_batch(walk: DTQW, images, key):
    n = int(images.shape[0])
    key, kt, kf = jax.random.split(key, 3)
    ts = jax.random.randint(kt, (n,), 0, T)
    x_t = np.empty((n, H, W), dtype=np.int32)
    x_next = np.empty((n, H, W), dtype=np.int32)
    keys = jax.random.split(kf, n)
    for i in range(n):
        xt, xt1 = walk.training_pair_image(images[i], int(ts[i]), keys[i])
        x_t[i] = np.asarray(xt)
        x_next[i] = np.asarray(xt1)
    return jnp.asarray(x_next), jnp.asarray(x_t), ts.astype(jnp.float64)


def main():
    key = jax.random.PRNGKey(0)
    walk = DTQW(n_pos=2)
    key, k_data, k_model, k_batch = jax.random.split(key, 4)
    images = blob_images(k_data)
    params = init_reverse(k_model, k=K, hidden=8)

    x_next, x_t, ts = make_batch(walk, images, k_batch)
    print(f"batch x_t unique values: {np.unique(np.asarray(x_t)).tolist()}")

    losses = []
    p = params
    for step in range(STEPS):
        p, loss = sgd_step(p, x_next, x_t, ts, lr=0.15)
        losses.append(float(loss))
        if step % 20 == 0 or step == STEPS - 1:
            print(f"step {step:3d}  ce={losses[-1]:.4f}")

    print(f"ce start={losses[0]:.4f}  ce end={losses[-1]:.4f}  decreased={losses[-1] < losses[0]}")

    key, k_gen = jax.random.split(key)
    fake = generate(p, shape=(N_IMAGES, H, W), T=T, key=k_gen, sample=True)
    p_data = intensity_histogram(images, k=K)
    p_gen = intensity_histogram(fake, k=K)
    kl = float(histogram_kl(p_data, p_gen))
    print(f"data hist={np.asarray(p_data)}")
    print(f"gen  hist={np.asarray(p_gen)}")
    print(f"histogram KL(data || gen)={kl:.4f}")
    print(f"generated range [{int(np.min(fake))}, {int(np.max(fake))}] subset of 0..{K-1}")


if __name__ == "__main__":
    main()
