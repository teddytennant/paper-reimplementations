"""Run the small combinatorial checks this package actually tests."""

import jax.numpy as jnp

from iar.data import combinatorial_codes
from iar.refine import (
    multi_restart_recovery,
    one_atom_iar,
    refine_step,
    selection_frequencies,
)


def main():
    K, s = 7, 2
    X = combinatorial_codes(K, s)
    L = int(X.shape[1] // K)
    p = jnp.arange(K, 0, -1, dtype=jnp.float64)
    d0 = p / jnp.linalg.norm(p)
    d1, selected = refine_step(d0, X, L)
    freq = selection_frequencies(selected, X)
    print(f"monotonicity K={K} s={s} L={L} M={int(X.shape[1])}")
    print("frequencies", [float(v) for v in freq])
    print("A.T d1", [float(v) for v in d1])

    Ka, sa = 11, 2
    Xa = combinatorial_codes(Ka, sa)
    La = int(Xa.shape[1] // Ka)
    pa = jnp.full(Ka, 0.25).at[0].set(1.0)
    d0a = pa / jnp.linalg.norm(pa)
    trajectory = one_atom_iar(Xa, La, n_steps=3, d0=d0a)
    d = trajectory[3]
    align = abs(float(d[0]))
    arg = int(jnp.argmax(jnp.abs(d)))
    print(f"amplification K={Ka} s={sa} L={La} M={int(Xa.shape[1])}")
    print(f"alignment {align:.12f} argmax {arg}")

    kept = multi_restart_recovery(X, L, n_restarts=12, n_steps=3, seed=0)
    recovered = {}
    for row in kept:
        k = int(jnp.argmax(jnp.abs(row)))
        atom_align = abs(float(row[k]))
        if atom_align > 0.8:
            recovered[k] = max(recovered.get(k, 0.0), atom_align)
    print(
        f"restarts seed=0 n=12 kept={int(kept.shape[0])} "
        f"recovered={sorted(recovered)}"
    )


if __name__ == "__main__":
    main()
