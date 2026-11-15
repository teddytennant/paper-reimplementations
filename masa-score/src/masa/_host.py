"""Set the nix-ld library path before JAX is imported."""

import os

NIX_LD_LIB = "/run/current-system/sw/share/nix-ld/lib"


def configure_host():
    """Prepend the nix-ld lib directory when LD_LIBRARY_PATH lacks nix-ld."""
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "nix-ld" not in ld:
        os.environ["LD_LIBRARY_PATH"] = (
            NIX_LD_LIB + (":" + ld if ld else "")
        )
