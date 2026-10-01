import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

from roc_vus.vus import (
    cross_fit_vus,
    label_vus,
    mixed_volume,
    plugin_vus,
    population_vus_mc,
    population_vus_ustat,
    u_statistic_g,
)

__all__ = [
    "cross_fit_vus",
    "label_vus",
    "mixed_volume",
    "plugin_vus",
    "population_vus_mc",
    "population_vus_ustat",
    "u_statistic_g",
]
