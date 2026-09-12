JAX implementation of SV-GCN (https://arxiv.org/abs/2609.11680).

Synthetic 16-joint skeletons only. This does not reproduce E-Gait numbers. The compact-concat of eq. 9 is replaced by mask-multiply. A 1x H200 smoke of `scripts/smoke_train.py` was submitted but sat in Priority, so this release is CPU-tested only.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke_train.py
```
