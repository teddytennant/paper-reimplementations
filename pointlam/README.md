PointLAM kernels from arXiv:2609.21780: Laplacian Point Sampler (LPS) and Local Hadamard Aggregator (LHA), wired as a tiny LAM block in JAX.

```
pip install -e ".[tests]"
python -m pytest -q
```

```
python scripts/smoke_train.py
POINTLAM_CPU=1 python scripts/smoke_train.py
```

Transient grid: a regular 3D cell hash. Neighbors of a point are every point whose cell is within Chebyshev distance 1 of its cell (home cell plus the 26 adjacent cells). Not kNN.

BDM: for the X axis then the Y axis, points are sorted by that coordinate, a tiny input-gated selective SSM is scanned forward and backward, and the two directions are added. State width is 8. No Hilbert curve.

## What does not match the paper

- no nuScenes/Waymo, no OpenPCDet, no pretrained backbone, no 3D detection head
- no Hilbert serialization
- simplified BDM (tiny selective scan, not a full Mamba)
- LPS deviation is ||MLP(p_i) - mean_{j in N(i)} MLP(p_j)||_2 as specified, not ||Concat(h_i, delta_i)||_2
- DSS is the unique crop of Argsort(Dev)[:M] union Argsort(Random)[:M], not the paper's region-window uniqueness
- LHA uses mean-pooled Hadamard gating as specified, not the paper's W(V(p_j)-V(p_i)) · f_j sum
- residual is LayerNorm(f_in + Linear(f') odot sigma(Linear(f_in))) as specified, not (g odot c) + f without LN
- LAM block is LHA then BDM. The paper interleaves BDM with LHA
- tiny feature dim (16), synthetic clouds only
