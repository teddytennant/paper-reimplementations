"""Train the outer autoencoder of Fig. 6 on MNIST, then run Algorithm 1 in its latent space.

The paper trains a VQ-VAE with an L1, perceptual and adversarial loss, eq (30).
This is a plain conv autoencoder with an L1 reconstruction loss and a
hand-rolled Adam, small enough to train on a CPU in a few minutes. It gets the
3 x 7 x 7 latent shape of Table II and nothing else about the setup.

There is no download here. Point --data at a local MNIST file, or pass
--synthetic to exercise the pipeline on generated blobs.

    python scripts/train_mnist_ae.py --data ~/data/mnist.npz --out /tmp/mnist_alg1.npz
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

from diffusion_is_attention import Codec, generate

LATENT_CHANNELS = 3  # Table II, z-shape 3 x 7 x 7 for MNIST


def load_mnist(path):
    """Read MNIST from a local .npz or the raw idx files. No network access."""
    path = pathlib.Path(path).expanduser()
    if path.is_dir():
        idx = path / "train-images-idx3-ubyte.gz"
        if not idx.exists():
            idx = path / "train-images-idx3-ubyte"
        if not idx.exists():
            raise FileNotFoundError(f"no train-images-idx3-ubyte[.gz] under {path}")
        opener = gzip.open if idx.suffix == ".gz" else open
        with opener(idx, "rb") as fh:
            blob = fh.read()
        n = int.from_bytes(blob[4:8], "big")
        images = np.frombuffer(blob[16:], dtype=np.uint8).reshape(n, 28, 28)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as z:
            key = "x_train" if "x_train" in z else list(z.keys())[0]
            images = np.asarray(z[key])
        if images.ndim == 4:
            images = images[..., 0]
    else:
        raise FileNotFoundError(f"do not know how to read {path}")
    return (images.astype(np.float32) / 255.0)[:, None]


def synthetic_digits(n=4096, seed=0):
    """Gaussian blobs on a 28 x 28 grid. Not MNIST, just something to train on."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(np.arange(28), np.arange(28), indexing="ij"), -1)
    out = np.zeros((n, 28, 28), np.float32)
    for i in range(n):
        for _ in range(3):
            c = rng.uniform(6, 22, size=2)
            s = rng.uniform(2.0, 4.5)
            out[i] += np.exp(-((grid - c) ** 2).sum(-1) / (2 * s**2))
    return np.clip(out, 0, 1)[:, None]


def init_params(key):
    ks = jax.random.split(key, 6)

    def conv(k, shape):
        fan_in = np.prod(shape[1:])
        return jax.random.normal(k, shape) * np.sqrt(2.0 / fan_in)

    return {
        "e1": conv(ks[0], (16, 1, 4, 4)),
        "e2": conv(ks[1], (32, 16, 4, 4)),
        "e3": conv(ks[2], (LATENT_CHANNELS, 32, 3, 3)),
        "d1": conv(ks[3], (32, LATENT_CHANNELS, 3, 3)),
        "d2": conv(ks[4], (32, 16, 4, 4)),
        "d3": conv(ks[5], (16, 1, 4, 4)),
    }


def encode(params, x):
    """(B, 1, 28, 28) -> (B, 3, 7, 7)."""
    h = jax.nn.relu(_conv(x, params["e1"], 2))
    h = jax.nn.relu(_conv(h, params["e2"], 2))
    return _conv(h, params["e3"], 1)


def decode(params, z):
    h = jax.nn.relu(_conv(z, params["d1"], 1))
    h = jax.nn.relu(_deconv(h, params["d2"], 2))
    return jax.nn.sigmoid(_deconv(h, params["d3"], 2))


def _conv(x, w, stride):
    return jax.lax.conv_general_dilated(
        x, w, (stride, stride), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")
    )


def _deconv(x, w, stride):
    return jax.lax.conv_transpose(
        x,
        w,
        (stride, stride),
        "SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
        transpose_kernel=False,
    )


def loss_fn(params, batch):
    return jnp.mean(jnp.abs(decode(params, encode(params, batch)) - batch))


def train(params, images, steps, batch_size, lr, key):
    """Adam, written out, so the only dependency stays jax."""
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)

    @jax.jit
    def step(params, m, v, t, batch):
        loss, g = jax.value_and_grad(loss_fn)(params, batch)
        m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree.map(lambda a, b: 0.99 * a + 0.01 * b * b, v, g)
        mh = jax.tree.map(lambda a: a / (1 - 0.9**t), m)
        vh = jax.tree.map(lambda a: a / (1 - 0.99**t), v)
        params = jax.tree.map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + 1e-8), params, mh, vh)
        return params, m, v, loss

    n = images.shape[0]
    for t in range(1, steps + 1):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch_size,), 0, n)
        params, m, v, loss = step(params, m, v, t, images[idx])
        if t % max(steps // 10, 1) == 0:
            print(f"  step {t:5d}  L1 {float(loss):.4f}")
    return params


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="~/data/mnist.npz")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bank", type=int, default=2048, help="latents to attend over")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--out", default="mnist_alg1.npz")
    args = ap.parse_args(argv)

    if args.synthetic:
        print("using synthetic blobs, not MNIST")
        images = synthetic_digits()
    else:
        try:
            images = load_mnist(args.data)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"could not read MNIST: {exc}", file=sys.stderr)
            print(
                "Pass --data pointing at a local mnist.npz or a directory of the raw "
                "idx files, or --synthetic to run the pipeline on generated blobs. "
                "This script does not download anything.",
                file=sys.stderr,
            )
            return 1
    images = jnp.asarray(images)
    print(f"{images.shape[0]} images of shape {images.shape[1:]}")

    key = jax.random.key(0)
    kp, kt, kq, kn = jax.random.split(key, 4)
    params = train(init_params(kp), images, args.steps, args.batch_size, args.lr, kt)

    bank = images[: args.bank]
    latents = jax.jit(encode)(params, bank)
    print(f"latent bank {latents.shape}, mean radius "
          f"{float(jnp.mean(jnp.linalg.norm(latents.reshape(latents.shape[0], -1), axis=-1))):.3f}")

    codec = Codec(
        encode=lambda x: encode(params, x[None])[0],
        decode=lambda z: decode(params, z[None])[0],
        refine=lambda x: decode(params, encode(params, x[None]))[0],
    )

    picks = jax.random.randint(kq, (args.samples,), 0, latents.shape[0])
    keys = jax.random.split(kn, args.samples)
    images_out, queries, neighbors = [], [], []
    for pick, k in zip(picks, keys):
        out = generate(
            latents, latents[pick], k=args.k, sigma=args.sigma, eta=args.eta,
            key=k, codec=codec,
        )
        images_out.append(np.asarray(out.image))
        queries.append(np.asarray(bank[pick]))
        neighbors.append(out.neighbor_indices)

    np.savez(
        args.out,
        generated=np.stack(images_out),
        queries=np.stack(queries),
        neighbors=np.stack(neighbors),
        bank=np.asarray(bank),
    )
    print(f"wrote {args.samples} samples to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
