import ctypes
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_LIBSTDCXX = "/run/current-system/sw/share/nix-ld/lib/libstdc++.so.6"
if os.path.isfile(_LIBSTDCXX):
    ctypes.CDLL(_LIBSTDCXX, mode=ctypes.RTLD_GLOBAL)

from jax import config

config.update("jax_enable_x64", True)
