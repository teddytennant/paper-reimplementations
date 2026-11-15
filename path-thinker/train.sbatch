#!/usr/bin/env bash
#
# Smoke run for Painter and Thinker (arXiv 2610.09876). Copied from
# templates/sbatch.sh with the two placeholders filled in.
#
# __VENV_SETUP__   : reuse the shared CUDA jax venv
# __RUN_COMMAND__  : short synthetic Painter then Thinker steps, 1x H200

set -uo pipefail
cd "$(dirname "$0")"

PY="$HOME/arxiv-jax/venv/bin/python"
export JAX_PLATFORMS=cuda
unset JAX_PLATFORM_NAME || true

ROOT="${SLURM_SUBMIT_DIR:-/work/ttennant1/arxiv-impl/20261009T041804Z}"
export PYTHONPATH="$ROOT/path-thinker/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/path-thinker"

"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
"$PY" scripts/smoke.py --steps-painter 5 --steps-thinker 8
rc=$?

echo "exit $rc"
exit $rc
