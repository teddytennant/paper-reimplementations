"""x64 everywhere, so the exactness tests mean something on a GPU.

float32 matmuls run at TF32 on NVIDIA, which costs about 10 bits of mantissa
and would turn the decomposition test into a test of the tolerance rather than
of the code. float64 has no TF32 path, so the same numbers come out of a CPU
and an H200. The toy classifier still trains in float32 for speed and gets cast
up when a test needs exactness.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (after the x64 switch)

from head_influence.model import ModelConfig, init_params  # noqa: E402
from head_influence.train import train  # noqa: E402

DTYPE = jnp.float64


def to_float64(params):
    return jax.tree.map(lambda a: a.astype(jnp.float64), params)


@pytest.fixture(scope="session")
def cfg():
    return ModelConfig()


@pytest.fixture(scope="session")
def random_params(cfg):
    return init_params(jax.random.key(0), cfg, DTYPE)


@pytest.fixture(scope="session")
def deep_cfg():
    """More layers than the toy model, for the layer-subset tests."""
    return ModelConfig(n_layers=3, n_heads=4, d_model=64, d_head=16, d_mlp=128)


@pytest.fixture(scope="session")
def deep_params(deep_cfg):
    return init_params(jax.random.key(1), deep_cfg, DTYPE)


@pytest.fixture(scope="session")
def trained():
    """The toy classifier, trained once for the whole session."""
    cfg = ModelConfig()
    params, acc = train(jax.random.key(0), cfg, steps=800, dtype=jnp.float32)
    return cfg, params, acc
