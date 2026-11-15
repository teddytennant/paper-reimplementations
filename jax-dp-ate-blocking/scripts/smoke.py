"""Well-specified n=400: noise-free IPW/BPS and one SeqIPW + DPBlocking at ε=1."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dp_ate_blocking.dgp import well_specified
from dp_ate_blocking.estimators import bps, difference_in_means, ipw
from dp_ate_blocking.privacy import default_m, dp_blocking, seq_ipw
from dp_ate_blocking.propensity import fit_propensity, predict_propensity


def main() -> None:
    tau = 2.0
    n = 400
    key = jax.random.PRNGKey(0)
    x, w, y, e_true = well_specified(key, n=n, tau=tau)
    theta = fit_propensity(x, w, add_noise=False)
    e_fit = predict_propensity(theta, x)
    m = default_m(n, 1.0)

    ipw_true = float(ipw(y, w, e_true, omega=0.1))
    ipw_fit = float(ipw(y, w, e_fit, omega=0.1))
    bps_fit = float(bps(y, w, e_fit, m=m))
    dim = float(difference_in_means(y, w))

    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    seq = float(seq_ipw(x, w, y, key=k1, eps=1.0, C_y=1.5))
    blk = float(dp_blocking(x, w, y, key=k2, eps=1.0, C_y=1.5, m=m))

    print(f"n={n} tau={tau} m={m}")
    print(f"difference_in_means={dim:.4f}")
    print(f"ipw_true_e={ipw_true:.4f}")
    print(f"ipw_fit={ipw_fit:.4f}")
    print(f"bps_fit={bps_fit:.4f}")
    print(f"seq_ipw_eps1={seq:.4f}")
    print(f"dp_blocking_eps1={blk:.4f}")
    print(f"finite={jnp.isfinite(jnp.array([ipw_true, ipw_fit, bps_fit, seq, blk])).all()}")


if __name__ == "__main__":
    main()
