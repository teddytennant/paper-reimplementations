"""Tiny CPU run of Algorithm 1 on a 2-D quadratic minimax toy."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_pdeg import params_from_L_mu_tau, quadratic_scsc, run_pdeg


def main():
    mu = 1.0
    L = float(np.sqrt(2.0))
    prob = quadratic_scsc(mu=mu)
    params = params_from_L_mu_tau(L=L, mu=mu, tau=0.0)
    x0 = jnp.array([1.0, -0.5])
    y0 = jnp.array([0.3, 0.2])
    T = 40
    res = run_pdeg(prob.grad_x, prob.grad_y, x0, y0, params, T=T, os_index=0)
    S = np.asarray(res.S)
    print("jax-pdeg smoke (Alg 1, 2-D SC-SC quadratic)")
    print(f"  T={T}  h={params.h:.4e}  beta={params.beta:.4e}  oracles={res.n_oracle}")
    print(f"  S[0]={S[0]:.6e}  S[min]={S.min():.6e}  S[-1]={S[-1]:.6e}")
    print(f"  GS index={res.gs_index}  ||x_out||={float(jnp.linalg.norm(res.x_out)):.6e}")
    print(f"  ||z_T||={float(jnp.linalg.norm(res.state.z)):.6e}")
    print(f"  residuals S[:5]={S[:5]}")


if __name__ == "__main__":
    main()
