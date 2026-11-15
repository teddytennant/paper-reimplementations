"""CPU/GPU smoke: tiny STAR-RIS instance, writes results.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax
import jax.numpy as jnp

from psagml.channels import rayleigh_channels
from psagml.pso import random_phase_zf_wsr
from psagml.train_meta import meta_train, run_psagml


def main():
    Nt, N, Kt, Kr = 8, 8, 1, 1
    Pmax, sigma2 = 10.0, 1.0
    hidden, nu, S = 8, 0.05, 5
    key = jax.random.PRNGKey(0)
    k_train, k_eval, k_rnd = jax.random.split(key, 3)

    train_channels = [
        rayleigh_channels(jax.random.fold_in(k_train, i), Nt, N, Kt, Kr)
        for i in range(2)
    ]
    psi, losses = meta_train(
        k_train,
        train_channels,
        T=3,
        n_steps=3,
        hidden=hidden,
        nu=nu,
        lr=1e-3,
        Pmax=Pmax,
        sigma2=sigma2,
        n_particles=8,
        n_pso_iters=5,
        warm_start="pso",
    )
    Mc, Mx = psi

    G, Ht, Hr = rayleigh_channels(k_eval, Nt, N, Kt, Kr)
    omega = jnp.ones((Kt + Kr,), dtype=jnp.float32)
    out = run_psagml(
        k_eval,
        G,
        Ht,
        Hr,
        Pmax=Pmax,
        sigma2=sigma2,
        omega=omega,
        n_particles=8,
        n_pso_iters=10,
        hidden=hidden,
        nu=nu,
        S=S,
        Mc=Mc,
        Mx=Mx,
    )
    wsr_random = random_phase_zf_wsr(k_rnd, G, Ht, Hr, Pmax, sigma2, omega)

    payload = {
        "wsr_pso": float(out["wsr_pso"]),
        "wsr_refined": float(out["wsr_refined"]),
        "wsr_random": float(wsr_random),
        "meta_loss_start": float(losses[0]),
        "meta_loss_end": float(losses[-1]),
        "Nt": Nt,
        "N": N,
        "K": Kt + Kr,
    }
    path = Path("results.json")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"PSA-GML smoke: wsr_pso={payload['wsr_pso']:.4f} "
        f"wsr_refined={payload['wsr_refined']:.4f} "
        f"wsr_random={payload['wsr_random']:.4f} wrote {path}"
    )


if __name__ == "__main__":
    main()
