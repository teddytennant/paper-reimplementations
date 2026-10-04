"""Small Condorcet instance for DCTAS."""

import numpy as np

from dctas import dctas


def main() -> None:
    preference = np.array(
        [
            [0.5, 0.9, 0.9],
            [0.1, 0.5, 0.6],
            [0.1, 0.4, 0.5],
        ],
        dtype=float,
    )
    costs = np.array([1.0, 5.0, 2.0], dtype=float)
    result = dctas(preference, costs, delta=0.1, seed=0)
    print(f"winner {result.winner}")
    print(f"duel count {result.duels}")
    print(f"total cost {result.total_cost}")


if __name__ == "__main__":
    main()
