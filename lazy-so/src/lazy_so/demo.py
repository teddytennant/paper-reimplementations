"""Run one outer step of search-free acceleration on a small quadratic."""

from lazy_so.core import CountingOracle, search_free_ms_acceleration

import jax.numpy as jnp


def main() -> None:
    x_star = jnp.asarray([0.2, -0.4, 0.1, 0.3], dtype=jnp.float64)
    x0 = jnp.zeros(4, dtype=jnp.float64)

    def f(x):
        gap = x - x_star
        return 0.5 * jnp.dot(gap, gap)

    oracle = CountingOracle(f)
    y = search_free_ms_acceleration(
        oracle.grad, oracle.hess, x0, gamma=1.0, T=1, L=0.0
    )
    start = float(oracle.value(x0))
    end = float(oracle.value(y))
    print(f"start_f={start:.8f}")
    print(f"end_f={end:.8f}")
    print(f"n_grad={oracle.n_grad}")
    print(f"n_hess={oracle.n_hess}")


if __name__ == "__main__":
    main()
