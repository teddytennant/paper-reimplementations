"""Load host C++/zlib before test modules import jax."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from snapshot_ident._host import preload

preload()

import jax

jax.config.update("jax_enable_x64", True)
