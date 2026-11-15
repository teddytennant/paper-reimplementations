"""Print Table 2 of arXiv:2609.04420, then check a few cells against simulation.

    python -m stratified_allocation.demo
"""

import jax
import jax.numpy as jnp

from stratified_allocation.designs import (
    balanced_allocation,
    class_index_sets,
    stratified_estimate,
)
from stratified_allocation.population import (
    class_counts,
    class_dispersions,
    equidispersed_population,
)
from stratified_allocation.variance import (
    efficiency_prediction,
    realised_allocation_ratio,
    variance_stratified,
)

PREVALENCES = (0.0037, 0.0071, 0.0143)
RETENTIONS = (0.01, 0.025, 0.05, 0.10, 0.20)


def table_2() -> str:
    head = "  pi     " + "".join(f"{f:>14.3%}" for f in RETENTIONS)
    rows = [head]
    for pi in PREVALENCES:
        cells = []
        for f in RETENTIONS:
            a = float(efficiency_prediction(pi, f))
            g = float(realised_allocation_ratio(pi, f))
            cells.append(f"{a:>8.1f} ({g:.2f})")
        rows.append(f"{pi:.4f}   " + "".join(f"{c:>14}" for c in cells))
    return "\n".join(rows)


def simulated_check(reps: int = 20_000) -> str:
    """Simulated variance ratio against the prediction, at three design points.

    A(pi, f) drops the finite-population corrections, so the two agree only where the
    sampling fractions are small and the truncation is slack. The last row is a point
    where the truncation binds and the prediction understates the gain by a lot, which
    is the honest thing to show.
    """
    lines = ["   n     N1       f    gamma   simulated    exact    A(pi,f)"]
    for n, n1, f in [(2000, 500, 0.02), (1000, 250, 0.04), (2000, 40, 0.10)]:
        pop = equidispersed_population(jax.random.key(0), n, n1, sigma=1.0)
        counts = class_counts(pop.labels)
        disp = class_dispersions(pop.losses, pop.labels)
        budget = int(jnp.ceil(f * n))
        bal = balanced_allocation(counts, budget)
        total = bal[0] + bal[1]
        prop = (round(total * float(counts[0]) / n), round(total * float(counts[1]) / n))

        idx0, idx1 = class_index_sets(pop.labels)
        keys = jax.random.split(jax.random.key(1), reps)
        sims = []
        for k0, k1 in (prop, bal):
            draw = jax.jit(
                jax.vmap(lambda k: stratified_estimate(k, pop.losses, idx0, idx1, k0, k1))
            )(keys)
            sims.append(float(jnp.var(draw, ddof=1)))
        exact = float(
            variance_stratified(counts, disp, jnp.asarray(prop, jnp.float64))
            / variance_stratified(counts, disp, jnp.asarray(bal, jnp.float64))
        )
        lines.append(
            f"{n:>5} {n1:>6} {f:>7.2%} {float(realised_allocation_ratio(n1 / n, f)):>8.2f}"
            f" {sims[0] / sims[1]:>11.2f} {exact:>8.2f} {float(efficiency_prediction(n1 / n, f)):>10.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    print("Predicted efficiency A(pi, f), Table 2 of arXiv:2609.04420 (gamma in parens)")
    print(table_2())
    print()
    print("Simulated variance ratio, proportional over truncated balanced")
    print(simulated_check())


if __name__ == "__main__":
    main()
