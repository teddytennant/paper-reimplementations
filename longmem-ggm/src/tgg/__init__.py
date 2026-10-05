"""Gaussian graphical model tests for long-memory time series.

Implements the local-inverse residual statistic, overlapping block bootstrap,
and consistency-empowered correction from arXiv 2609.30565.
"""

import jax

jax.config.update("jax_enable_x64", True)

from tgg.core import (
    TestResult,
    block_bootstrap_quantile,
    ce_test,
    consistency_empowered_statistic,
    direct_statistic,
    direct_test,
    local_inverse,
    plugin_scores,
    sample_covariance,
    screen_correction,
    suggested_block_length,
)
from tgg.generate import (
    band_edges,
    generate_linear_process,
    power_law_coefficients,
    sample_iid,
    simulation_precision,
    tridiagonal_precision,
)
from tgg.twosample import (
    thresholded_partial_correlation_graph,
    two_sample_test,
)

__all__ = [
    "TestResult",
    "band_edges",
    "block_bootstrap_quantile",
    "ce_test",
    "consistency_empowered_statistic",
    "direct_statistic",
    "direct_test",
    "generate_linear_process",
    "local_inverse",
    "plugin_scores",
    "power_law_coefficients",
    "sample_covariance",
    "sample_iid",
    "screen_correction",
    "simulation_precision",
    "suggested_block_length",
    "thresholded_partial_correlation_graph",
    "tridiagonal_precision",
    "two_sample_test",
]
