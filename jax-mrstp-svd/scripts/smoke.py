"""Tiny synthetic compression smoke (not the paper's image/video experiments)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from jax import random

from jax_mrstp_svd import mrstp_svd, mstp_reconstruct, mstp_svd, relative_frobenius_error


def main() -> None:
    key = random.PRNGKey(0)
    m1, m2, n1, n2, n3 = 4, 2, 4, 2, 6
    A = random.normal(key, (m1 * m2, n1 * n2, n3))
    print(f"synthetic tensor {tuple(A.shape)}  (m1,m2,n1,n2,n3)=({m1},{m2},{n1},{n2},{n3})")
    print("transform = identity")
    print(f"{'method':<22} {'k':>3}  rel. Frobenius error")
    for k in (1, 2, 3):
        rec = mstp_reconstruct(
            mstp_svd(A, m1, m2, n1, n2, k=k, transform="identity"), "identity"
        )
        err = float(relative_frobenius_error(A, rec))
        print(f"{'MSTP-SVD':<22} {k:>3}  {err:.6e}")
    rec_r = mstp_reconstruct(
        mrstp_svd(
            A,
            m1,
            m2,
            n1,
            n2,
            k=2,
            transform="identity",
            oversampling=3,
            n_power=1,
            key=random.PRNGKey(1),
        ),
        "identity",
    )
    err_r = float(relative_frobenius_error(A, rec_r))
    print(f"{'MRSTP-SVD (q=1,s=3)':<22} {2:>3}  {err_r:.6e}")


if __name__ == "__main__":
    main()
