"""Shared helpers. scipy is used here only as an independent oracle, never in the model."""

import numpy as np
import pytest


def gauss_legendre_panels(lo, hi, n_panels, order=8):
    """Nodes and weights of a composite Gauss-Legendre rule on [lo, hi]."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(lo, hi, n_panels + 1)
    a, b = edges[:-1, None], edges[1:, None]
    nodes = (0.5 * (b - a) * x + 0.5 * (a + b)).ravel()
    weights = (0.5 * (b - a) * w).ravel()
    return nodes, weights


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
