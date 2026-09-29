"""CPU + float64 before test modules import jax."""

import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gp_dps._host import preload

preload()

import jax

jax.config.update("jax_enable_x64", True)
