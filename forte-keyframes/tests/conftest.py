import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
