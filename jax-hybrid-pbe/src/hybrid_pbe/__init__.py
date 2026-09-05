"""JAX implementation of arXiv:2609.04011, differentiable hybrid PBE modelling."""

from .grid import Grid, aspect_ratio, crystal_volume, mean_lengths, moment
from .hybrid import LearnableSeed, NeuralDissolution
from .kinetics import (
    K_V,
    M1,
    M4,
    RHO_C,
    S_MAX,
    S_MIN,
    ConstantRate,
    GrowthDissolution,
    SizeDependentGrowth,
    solubility,
    supersaturation,
)
from .solver import (
    ConstantTemperature,
    SimResult,
    TemperatureProfile,
    euler_step,
    simulate,
    substeps_for_cfl,
    total_mass,
)

__all__ = [
    "Grid", "moment", "crystal_volume", "mean_lengths", "aspect_ratio",
    "solubility", "supersaturation", "M1", "M4", "SizeDependentGrowth",
    "GrowthDissolution", "ConstantRate", "RHO_C", "K_V", "S_MIN", "S_MAX",
    "simulate", "euler_step", "SimResult", "TemperatureProfile",
    "ConstantTemperature", "substeps_for_cfl", "total_mass",
    "NeuralDissolution", "LearnableSeed",
]
