import os
import sys
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tgi_cv._host import ensure_nix_ld

ensure_nix_ld()
