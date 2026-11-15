JAX kernel for 3D point-cloud splatting (3DPCS) from PRI-Net (arXiv:2609.14469).

Implements the virtual pinhole projection of LiDAR points (eq. 7), Chebyshev
(L_inf) disks of radius R, and min-depth splat into a dense map (eq. 8). Empty
pixels are tracked by occupancy O and written as 0 in D_final. Tiny tensor RAF
gates follow eqs. 9-11 (sigmoid weights, residual feature gate, coarse estimate
plus residual).

```
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: Does not train PRI-Net, YOLOv7, ResNet, MIB, or UAV localization. Only the min-depth Chebyshev splat on synthetic cameras, plus tiny tensor RAF gates (eqs. 9-11 without an MLP/CNN backbone). Skips Mei unified projection and distortion (eqs. 1-3), LiDAR spherical ranging (eqs. 4-5), YOLO ROI boxes (eq. 6), ResNet encoders, auxiliary loss (eq. 12), and MIB/VIB compression.
