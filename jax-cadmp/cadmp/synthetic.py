"""A synthetic stand-in for the MSCOCO measurement in Figure 3.

This is not the paper's data and it is not a replication. Nothing here runs an
LVLM. What it does is generate cross-modal attention tensors and target-token
probabilities with the qualitative structure the paper reports, then push them
through the real feature extraction and the real detector, so the pipeline is
exercised end to end on inputs whose ground truth is known.

The two properties it is built to have, both stated in the paper:

* Hallucinated samples drift more in the middle layers. Figure 3 shows the
  grounded and hallucinated average drift curves separating around layers 8 to
  16, and Figure 6 puts the peak-drift layer most often at 15, 10 and 13 for
  hallucinated samples on Qwen2.5-VL-7B. Here that comes from a per-layer step
  size in a random walk over attention logits, with a bump over layers 8 to 16
  that is larger for hallucinated samples.
* Grounded samples lose more probability when the attended region is masked.
  "Grounded predictions are expected to be more sensitive to removing visual
  evidence, whereas hallucinated predictions typically show smaller changes."

Where the fabrication actually sits. The attention tensors are generated, and
the drift features are then computed from them by the same code a real trace
would go through, including the head averaging, the restriction to the visual
subset and the KL. The masking is real too: each sample has an image and a
hidden object region, ``l_mask`` is chosen from the sample's own drift profile,
and the mask is the actual 70th-percentile threshold of that layer's upsampled
attention. What is invented is the second forward pass. In place of an LVLM,
``P_masked = P_orig * (1 - s * c)`` where ``c`` is the fraction of the object
region the mask covered and ``s`` is a per-sample sensitivity, drawn high for
grounded samples and near zero for hallucinated ones. So ``delta_P`` is not
sampled directly; it is produced by the masking code, but through a response
model that stands in for the LVLM rather than for the LVLM itself.

Grounded samples also attend to the object region, so their masks cover more of
it, which is the second route by which ``delta_P`` separates. Class balance
defaults to 30% hallucinated, so the weighted sampler has something to do.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .drift import drift_profile, select_mask_layer
from .attention import attention_distributions
from .masking import GRAY, apply_mask, attention_mask, upsample_attention

__all__ = ["SyntheticData", "generate"]


class SyntheticData(NamedTuple):
    """One generated split."""

    features: np.ndarray
    """``[n, L + 2]`` CADMP feature vectors."""

    labels: np.ndarray
    """``[n]``, 1 for hallucinated and 0 for grounded."""

    drift: np.ndarray
    """``[n, L - 1]`` drift profiles, kept so the Figure 3 curves can be drawn."""

    mask_layer: np.ndarray
    """``[n]`` 0-based ``l_mask`` per sample, for the Figure 6 histogram."""

    coverage: np.ndarray
    """``[n]`` fraction of each sample's object region that the mask covered."""


def _blur(x: np.ndarray) -> np.ndarray:
    """3x3 box blur over the last two axes, with edge padding.

    Raw Gaussian noise on a token grid looks nothing like attention. One blur
    gives blobs a few tokens wide, which is enough for the percentile mask to
    select connected regions rather than salt and pepper.
    """
    padded = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    out = np.zeros_like(x)
    for di in range(3):
        for dj in range(3):
            out += padded[..., di : di + x.shape[-2], dj : dj + x.shape[-1]]
    return out / 9.0


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _step_sizes(num_layers: int, labels: np.ndarray, rng) -> np.ndarray:
    """Per-layer random-walk step size, ``[n, L]``.

    A flat baseline plus a Gaussian bump centered at layer 12 with a width of 3,
    scaled up for hallucinated samples. Layers here are 0-based, so the bump
    covers roughly the paper's layers 8 to 16.
    """
    n = labels.shape[0]
    layers = np.arange(num_layers)
    center = rng.normal(12.0, 1.5, size=(n, 1))
    bump = np.exp(-0.5 * ((layers[None, :] - center) / 3.0) ** 2)
    base = 0.30 + rng.normal(0.0, 0.10, size=(n, 1))
    amplitude = np.where(
        labels == 1, rng.normal(0.52, 0.22, size=n), rng.normal(0.16, 0.22, size=n)
    )[:, None]
    return np.clip(base + amplitude * bump, 0.05, None)


def _object_boxes(n: int, grid: int, rng):
    """Random object boxes on the token grid, returned as a ``[n, grid, grid]`` mask."""
    sizes = rng.integers(3, 6, size=n)
    mask = np.zeros((n, grid, grid), dtype=np.float32)
    for k in range(n):
        s = int(sizes[k])
        r0 = int(rng.integers(0, grid - s + 1))
        c0 = int(rng.integers(0, grid - s + 1))
        mask[k, r0 : r0 + s, c0 : c0 + s] = 1.0
    return mask


def _attention_tensor(labels, obj, num_layers, num_heads, rng):
    """Build ``[n, L, H, N]`` raw cross-modal attention.

    Layer logits follow a random walk whose steps are spatially blurred, started
    from a bias on the object region that is strong for grounded samples and
    weak for hallucinated ones. Heads differ by an extra per-head perturbation,
    so averaging over ``H`` is not a no-op. Each layer's weights are then scaled
    by a per-sample visual mass in [0.2, 0.6], because a decoder softmax runs
    over the whole prefix and the visual tokens only get part of it; that is
    what makes the renormalization in ``cadmp.attention`` do real work.
    """
    n, grid = obj.shape[0], obj.shape[-1]
    anchor = np.clip(
        np.where(
            labels == 1,
            rng.normal(0.9, 0.8, size=n),
            rng.normal(1.7, 0.8, size=n),
        ),
        0.0,
        None,
    )[:, None, None]
    logits = anchor * obj + _blur(rng.normal(0.0, 1.0, size=(n, grid, grid)))

    sigma = _step_sizes(num_layers, labels, rng)
    steps = _blur(rng.normal(0.0, 1.0, size=(n, num_layers, grid, grid)))
    steps = steps * sigma[:, :, None, None]
    steps[:, 0] = 0.0
    layer_logits = logits[:, None] + np.cumsum(steps, axis=1)

    head_noise = _blur(rng.normal(0.0, 0.45, size=(n, num_layers, num_heads, grid, grid)))
    per_head = _softmax(
        (layer_logits[:, :, None] + head_noise).reshape(n, num_layers, num_heads, -1),
        axis=-1,
    )
    visual_mass = rng.uniform(0.2, 0.6, size=(n, 1, 1, 1))
    return (per_head * visual_mass).astype(np.float32)


def _sample_features(attn, image, obj_pixels, p_orig, sensitivity, tau, eps):
    """Per-sample CADMP pass, written for ``jax.vmap`` over a chunk.

    Returns ``(features, drift, mask_layer, coverage)``. The masked image is
    built and thrown away; it exists so the code path a real caller takes is the
    code path the demo takes.
    """
    dists = attention_distributions(attn)
    drift = drift_profile(dists, eps=eps)
    mask_layer = jnp.argmax(drift)
    layer_attn = dists[mask_layer]

    pixel_attn = upsample_attention(layer_attn, image.shape[:2], method="bilinear")
    mask = attention_mask(pixel_attn, tau=tau)
    _ = apply_mask(image, mask, fill=GRAY)

    coverage = jnp.sum(mask * obj_pixels) / jnp.maximum(jnp.sum(obj_pixels), 1.0)
    p_masked = jnp.clip(p_orig * (1.0 - sensitivity * coverage), 1e-4, 1.0)
    tail = jnp.stack([p_orig, p_masked, p_orig - p_masked])
    return jnp.concatenate([drift, tail]), drift, mask_layer, coverage


_batched = jax.jit(
    jax.vmap(_sample_features, in_axes=(0, 0, 0, 0, 0, None, None)),
    static_argnums=(5, 6),
)


def generate(
    n_samples: int = 3000,
    num_layers: int = 32,
    num_heads: int = 4,
    grid: int = 16,
    patch: int = 4,
    hallucination_rate: float = 0.3,
    tau: float = 30.0,
    eps: float = 1e-9,
    seed: int = 0,
    chunk: int = 250,
) -> SyntheticData:
    """Generate a labeled synthetic CADMP dataset.

    Args:
        n_samples: how many samples.
        num_layers: ``L``. 32 is LLaVA-1.5-7B's depth, giving 31 drift features
            and a 34-dimensional feature vector.
        num_heads: ``H``. Small on purpose; head averaging is a mean, and its
            cost is linear in ``H`` while its effect saturates.
        grid: token grid side, so ``N = grid ** 2``. 16 rather than the paper's
            24 keeps the demo on a CPU budget.
        patch: pixels per token along each axis, so the image is
            ``grid * patch`` square.
        hallucination_rate: fraction of positives. The default 0.3 leaves the
            class imbalance the weighted sampler is there to fix.
        tau: masking percentage.
        eps: uniform-mixing weight for the KL.
        seed: numpy seed. The whole generator is deterministic given this.
        chunk: samples per vmapped batch, a memory knob only.

    Returns:
        ``SyntheticData``.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_samples) < hallucination_rate).astype(np.int32)
    side = grid * patch

    feats, drifts, layers_out, covs = [], [], [], []
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        sub = labels[start:stop]
        m = stop - start

        obj = _object_boxes(m, grid, rng)
        attn = _attention_tensor(sub, obj, num_layers, num_heads, rng)

        images = rng.uniform(0.0, 255.0, size=(m, side, side, 3)).astype(np.float32)
        obj_pixels = np.repeat(np.repeat(obj, patch, axis=1), patch, axis=2)

        p_orig = np.where(
            sub == 1, rng.beta(4.0, 2.2, size=m), rng.beta(4.4, 2.0, size=m)
        ).astype(np.float32)
        sensitivity = np.clip(
            np.where(
                sub == 1,
                rng.normal(0.34, 0.30, size=m),
                rng.normal(0.48, 0.30, size=m),
            ),
            0.0,
            1.0,
        ).astype(np.float32)

        f, d, l, c = _batched(
            jnp.asarray(attn),
            jnp.asarray(images),
            jnp.asarray(obj_pixels),
            jnp.asarray(p_orig),
            jnp.asarray(sensitivity),
            tau,
            eps,
        )
        feats.append(np.asarray(f))
        drifts.append(np.asarray(d))
        layers_out.append(np.asarray(l))
        covs.append(np.asarray(c))

    return SyntheticData(
        features=np.concatenate(feats),
        labels=labels,
        drift=np.concatenate(drifts),
        mask_layer=np.concatenate(layers_out),
        coverage=np.concatenate(covs),
    )
