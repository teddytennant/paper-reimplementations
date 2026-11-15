"""Sec. V, Algorithms 1 and 2 of the paper.

Algorithm 1 skips the reverse trajectory of Sec. II-D entirely. Since the
converged solution is a sparse combination of near neighbours anyway, it starts
from a dataset latent, takes one attention step against that latent's K nearest
neighbours, rescales to the mean latent radius, adds a little noise, quantizes,
decodes, and pushes the image through the outer autoencoder three times.

Algorithm 2 is the attention weighting, done per pixel location rather than per
whole latent: each (j, k) position gets its own softmax over the K neighbours.

The codec is a plug. The paper uses a trained VQ-VAE; the default here is the
identity, which makes Algorithm 1 run on raw arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "Codec",
    "IDENTITY_CODEC",
    "spatial_attention",
    "Generation",
    "generate",
]


def _identity(x):
    return x


@dataclass(frozen=True)
class Codec:
    """The outer VAE of Fig. 6, as four swappable functions.

    `decode` is D in line 11 of Alg. 1, `quantize` is Quantize in line 10, and
    `refine` is the AE of line 13. `encode` is not used by Alg. 1 itself; it is
    how you build the latent bank in the first place.
    """

    encode: Callable = _identity
    decode: Callable = _identity
    quantize: Callable = _identity
    refine: Callable = _identity


IDENTITY_CODEC = Codec()


def spatial_attention(q, neighbors, sigma, eps=1e-8):
    """Algorithm 2. `q` is (D, H, W), `neighbors` is (K, D, H, W), returns (K, H, W).

    Line 6 is the distance between the depth vector of the query at a pixel
    location and the neighbour's depth vector at the same location,
    d_ijk = ||q_{:jk} - s_{i:jk}||_2. Line 7 squares it again,
    p_ijk = -d_ijk^2 / (sigma^2 + eps), so what reaches the softmax is the
    squared distance, matching eq (11). Line 8 takes the softmax over the
    neighbour index i, independently at every (j, k).
    """
    q = jnp.asarray(q)
    neighbors = jnp.asarray(neighbors)
    d2 = jnp.sum((q[None] - neighbors) ** 2, axis=1)
    p = -d2 / (jnp.asarray(sigma) ** 2 + eps)
    return jax.nn.softmax(p, axis=0)


class Generation(NamedTuple):
    """Everything Alg. 1 produced, so the intermediate lines can be inspected."""

    image: jnp.ndarray
    latent: jnp.ndarray  # after line 9, the noise
    rescaled_latent: jnp.ndarray  # after line 8, the norm rescale
    weights: jnp.ndarray  # (K, H, W), from Alg. 2
    neighbor_indices: np.ndarray  # line 5
    radius: jnp.ndarray  # r of line 3


def generate(
    latents,
    query,
    k=3,
    sigma=0.01,
    eta=0.01,
    key=None,
    codec=IDENTITY_CODEC,
    refine_steps=3,
    eps=1e-8,
    tree=None,
    max_leaves=2,
):
    """Algorithm 1, Attention-Based Image Generation.

    `latents` is (N, D, H, W) and `query` is (D, H, W). Pass a
    `tree.SimilarityTree` to do line 5 with Sec. IV's tree cross attention
    instead of the O(N) scan of line 4.

    Note the paper writes epsilon for two different things: the constant in the
    denominator of line 8, and the noise sample in line 9. They are `eps` and
    `key` here.
    """
    latents = jnp.asarray(latents)
    query = jnp.asarray(query)
    n = latents.shape[0]

    # Line 3: the mean latent radius, what line 8 rescales to.
    r = jnp.mean(jnp.sqrt(jnp.sum(latents.reshape(n, -1) ** 2, axis=-1)))

    # Lines 4 and 5: the K nearest latents.
    if tree is None:
        d = jnp.sum((query[None] - latents) ** 2, axis=tuple(range(1, latents.ndim)))
        idx = np.asarray(jnp.argsort(d))[:k]
    else:
        idx = tree.query(query.reshape(-1), k=k, max_leaves=max_leaves).indices
    neighbors = latents[idx]

    # Lines 6 and 7: spatial attention, then the weighted sum over neighbours.
    w = spatial_attention(query, neighbors, sigma, eps=eps)
    z = jnp.sum(w[:, None] * neighbors, axis=0)

    # Line 8: put the result back on the sphere of mean latent radius.
    z = r * z / (jnp.linalg.norm(z) + eps)
    rescaled = z

    # Line 9: additive noise.
    if eta != 0.0:
        if key is None:
            raise ValueError("eta != 0 needs a PRNG key")
        z = z + eta * jax.random.normal(key, z.shape, dtype=z.dtype)

    # Lines 10 and 11, then the three refinement passes of lines 12 to 14.
    x = codec.decode(codec.quantize(z))
    for _ in range(refine_steps):
        x = codec.refine(x)

    return Generation(
        image=x,
        latent=z,
        rescaled_latent=rescaled,
        weights=w,
        neighbor_indices=np.asarray(idx),
        radius=r,
    )
