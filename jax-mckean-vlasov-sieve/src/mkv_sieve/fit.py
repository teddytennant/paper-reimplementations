"""Generate from a known Xi_0, fit the sieve MLE of eq (3.9), report the error.

    python -m mkv_sieve.fit --n 2000
"""

import argparse
import time

import jax

from mkv_sieve.mle import fit_mle, ise_on_interval, kl_stationary, l2_on_interval
from mkv_sieve.potential import OUPotential
from mkv_sieve.profile import SolverConfig, profile_a, solve_profile
from mkv_sieve.sample import sample_particles
from mkv_sieve.sieve import domain_bound, init_requ_params, make_xi
from mkv_sieve.truth import ENVELOPE, TRUTH


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000, help="sample size")
    ap.add_argument("--d", type=int, default=2, help="dimension")
    ap.add_argument("--xi", type=int, default=3, choices=[1, 2, 3],
                    help="which Xi_0 from Section 8")
    ap.add_argument("--width", type=int, default=3, help="hidden width of N_theta")
    ap.add_argument("--steps", type=int, default=500, help="Adam steps")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    pot = OUPotential(args.d)
    kappa, K = ENVELOPE[args.xi]
    xi_true = TRUTH[args.xi]
    U = domain_bound(pot, K)
    cfg = SolverConfig()

    prof0 = solve_profile(xi_true, pot, kappa, K, cfg)
    a0 = float(profile_a(prof0))

    key = jax.random.PRNGKey(args.seed)
    k_data, k_init = jax.random.split(key)
    x = sample_particles(k_data, prof0, pot, args.n)
    energies = pot.V(x)

    params0 = init_requ_params(k_init, width=args.width, U=U)
    t0 = time.time()
    params, losses = fit_mle(
        params0, energies, pot, kappa, K, U=U, cfg=cfg,
        steps=args.steps, lr=args.lr,
    )
    jax.block_until_ready(losses)
    elapsed = time.time() - t0

    xi_hat = make_xi(params, kappa, K, U=U)
    prof_hat = solve_profile(xi_hat, pot, kappa, K, cfg)

    # the identified interval of Remark 3.6, as in the paper's Section 8
    r_lo, r_hi = 0.05 * a0, 0.9 * a0
    ise = float(ise_on_interval(xi_hat, xi_true, r_lo, r_hi))
    l2 = float(l2_on_interval(xi_hat, xi_true, r_lo, r_hi))
    kl = float(kl_stationary(prof0, prof_hat, pot))

    print(f"Xi_0,{args.xi}  d={args.d}  n={args.n}  kappa={kappa} K={K} U={U:g}")
    print(f"a_0 = q_0(v_star) = {a0:.6f}   I = [{r_lo:.4f}, {r_hi:.4f}]")
    print(f"nll  {float(losses[0]):.6f} -> {float(losses[-1]):.6f}"
          f"   ({args.steps} Adam steps, {elapsed:.1f}s)")
    print(f"L2(I) error   {l2:.5f}")
    print(f"ISE(I)        {ise:.6f}")
    print(f"KL(pi_0||pi_hat)  {kl:.3e}")


if __name__ == "__main__":
    main()
