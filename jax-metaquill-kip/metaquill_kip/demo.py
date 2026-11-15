"""Fit an 8x8 Gaussian blob from 25% of its pixels.

Run with `python -m metaquill_kip.demo --steps 80`.

Compares closed-form NTK ridge regression (Eqs. 4-5) against NTK-KIP (Eq. 9)
with a distilled support of 8 points. The field is a width-16, depth-2 MLP
with 3 Fourier bands so the whole run stays on CPU.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from metaquill_kip.data import coord_grid, gaussian_blob, random_mask
from metaquill_kip.field import MLPConfig, encode, encoded_dim, init_params, make_field
from metaquill_kip.kip import init_support, kip_predict, ntk_kip
from metaquill_kip.metrics import psnr
from metaquill_kip.ntk import ntk_regression


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ridge", type=float, default=1e-4)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    h = 8
    k_bands = 3
    kind = "fourier"
    cfg = MLPConfig(in_dim=encoded_dim(2, kind, k_bands), width=16, depth=2)
    key = jax.random.PRNGKey(args.seed)
    params = init_params(cfg, key)
    f = make_field(cfg)

    coords = coord_grid(h)
    target = gaussian_blob(h)
    obs = random_mask(target.size, 0.25, args.seed + 1)
    encode_fn = lambda x, kind=kind, k_bands=k_bands: encode(x, kind, k_bands)
    Z = encode_fn(jnp.asarray(coords))
    Y = jnp.asarray(target)
    Z_obs, Y_obs = Z[obs], Y[obs]

    ntk_pred, _ = ntk_regression(f, params, Z_obs, Y_obs, Z, ridge=args.ridge)
    ntk_db = float(psnr(ntk_pred, Y))

    support, labels = init_support(coords[obs], target[obs], m=8, seed=args.seed + 2, space="coord", encode_fn=encode_fn)
    support, labels, hist = ntk_kip(
        f,
        params,
        Z_obs,
        Y_obs,
        support,
        labels,
        ridge=args.ridge,
        encode_fn=encode_fn,
        space="coord",
        steps=args.steps,
        lr=1e-2,
    )
    kip_pred, _ = kip_predict(f, params, support, labels, Z, args.ridge, encode_fn, space="coord")
    kip_db = float(psnr(kip_pred, Y))

    print(f"8x8 blob, {obs.size} observed pixels, ridge={args.ridge}")
    print(f"NTK ridge PSNR: {ntk_db:.2f} dB")
    print(f"NTK-KIP (m=8, {args.steps} steps) PSNR: {kip_db:.2f} dB  loss {float(hist[0]):.4f} -> {float(hist[-1]):.4f}")


if __name__ == "__main__":
    main()
