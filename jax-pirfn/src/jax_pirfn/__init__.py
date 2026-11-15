"""Physics-informed random feature networks (arXiv:2609.16406)."""

from jax_pirfn.features import (
    cosine_features,
    init_product_features,
    init_random_features,
    product_features,
    sample_bias,
    sample_omega,
)
from jax_pirfn.losses import pirfn_loss, relative_l2
from jax_pirfn.pde import (
    apply_1d,
    apply_product,
    heat_1d_exact,
    helmholtz_1d_source,
    poisson_1d_source,
    poisson_2d_source,
)
from jax_pirfn.train import (
    adam_init,
    adam_update,
    jacobian_map,
    ridge_readout,
    train_readout,
    train_step,
)

__all__ = [
    "adam_init",
    "adam_update",
    "apply_1d",
    "apply_product",
    "cosine_features",
    "heat_1d_exact",
    "helmholtz_1d_source",
    "init_product_features",
    "init_random_features",
    "jacobian_map",
    "pirfn_loss",
    "ridge_readout",
    "poisson_1d_source",
    "poisson_2d_source",
    "product_features",
    "relative_l2",
    "sample_bias",
    "sample_omega",
    "train_readout",
    "train_step",
]
