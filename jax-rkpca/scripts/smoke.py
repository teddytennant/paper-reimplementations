"""Short synthetic tracking demo. Prints Grassmann distances."""

from __future__ import annotations

import numpy as np

from jax_rkpca.metrics import (
    cosine_similarity_matched,
    generate_synthetic_stream,
    grassmann_distance,
    variance_weighted_grassmann,
)
from jax_rkpca.update import track


def main() -> None:
    n, D, d = 200, 8, 2
    eigenvalues = np.array([6.0, 3.0])
    X, Q, _ = generate_synthetic_stream(
        n=n, D=D, d=d, eigenvalues=eigenvalues, noise_std=0.08, seed=0
    )
    U = np.eye(D)
    kwargs = dict(d=d, eta=0.01, alpha0=0.1, kernel="linear")
    state_rot, _ = track(X, U, rotate=True, **kwargs)
    state_grad, _ = track(X, U, rotate=False, **kwargs)

    g_rot = float(grassmann_distance(state_rot.Gamma, Q))
    g_grad = float(grassmann_distance(state_grad.Gamma, Q))
    vw_rot = float(
        variance_weighted_grassmann(state_rot.Gamma, Q, eigenvalues)
    )
    cos_rot = np.asarray(cosine_similarity_matched(state_rot.Gamma, Q))

    print(f"n={n} D={D} d={d}")
    print(f"grad-only Grassmann: {g_grad:.6f}")
    print(f"grad+rot Grassmann:  {g_rot:.6f}")
    print(f"grad+rot variance-weighted Grassmann: {vw_rot:.6f}")
    print(f"grad+rot matched |cosine|: {cos_rot}")


if __name__ == "__main__":
    main()
