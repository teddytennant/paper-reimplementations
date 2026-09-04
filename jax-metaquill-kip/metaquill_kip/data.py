"""Toy fields and masks, all synthesized in-process.

The paper runs on MNIST digits, CelebA and a 180x180 Flowers image. Nothing is
downloaded here: these are analytic stand-ins with the same structure the methods
care about, namely a low-frequency blob, a high-frequency checkerboard, and a
small family of stroke shapes that stands in for a digit dataset when
meta-training MetaQuill over several tasks.

Coordinates live in [-1, 1]^2, as in Fig. 1.
"""

import numpy as np


def coord_grid(h, w=None):
    """The (h*w, 2) grid of neural field inputs, row-major over (row, col)."""
    w = h if w is None else w
    ys = np.linspace(-1.0, 1.0, h)
    xs = np.linspace(-1.0, 1.0, w)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gy.ravel(), gx.ravel()], axis=-1).astype(np.float64)


def gaussian_blob(h, w=None, center=(-0.1, 0.15), sigma=0.45):
    """A smooth low-frequency field, the easy case for a kernel fit."""
    w = h if w is None else w
    xy = coord_grid(h, w)
    d2 = (xy[:, 0] - center[0]) ** 2 + (xy[:, 1] - center[1]) ** 2
    return np.exp(-d2 / (2.0 * sigma**2))


def checkerboard(h, w=None, freq=3.0):
    """A high-frequency field. This is where raw coordinates alone fall over."""
    w = h if w is None else w
    xy = coord_grid(h, w)
    v = np.sin(freq * np.pi * xy[:, 0]) * np.sin(freq * np.pi * xy[:, 1])
    return 0.5 * (v + 1.0)


def _stroke(xy, p0, p1, width):
    """Soft distance-to-segment mask, the drawing primitive for the shapes."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    seg = p1 - p0
    t = np.clip(((xy - p0) @ seg) / (seg @ seg), 0.0, 1.0)
    closest = p0 + t[:, None] * seg
    d = np.linalg.norm(xy - closest, axis=-1)
    return np.exp(-((d / width) ** 2))


_SHAPES = {
    # crude digit-like stroke sets, enough to give the meta-learner a family of
    # related tasks without touching MNIST
    "seven": [((-0.7, -0.6), (-0.7, 0.6)), ((-0.7, 0.6), (0.7, -0.1))],
    "one": [((-0.7, 0.0), (0.7, 0.0)), ((-0.7, 0.0), (-0.3, -0.4))],
    "l": [((-0.7, -0.4), (0.6, -0.4)), ((0.6, -0.4), (0.6, 0.5))],
    "t": [((-0.7, -0.6), (-0.7, 0.6)), ((-0.7, 0.0), (0.7, 0.0))],
    "v": [((-0.7, -0.6), (0.7, 0.0)), ((0.7, 0.0), (-0.7, 0.6))],
    "x": [((-0.7, -0.6), (0.7, 0.6)), ((-0.7, 0.6), (0.7, -0.6))],
    "c": [((-0.6, 0.5), (-0.6, -0.4)), ((-0.6, -0.4), (0.6, -0.4)), ((0.6, -0.4), (0.6, 0.5))],
    "z": [((-0.7, -0.6), (-0.7, 0.6)), ((-0.7, 0.6), (0.7, -0.6)), ((0.7, -0.6), (0.7, 0.6))],
}

SHAPE_NAMES = tuple(_SHAPES)


def stroke_field(name, h, w=None, width=0.18):
    """One digit-like task from the synthetic family."""
    w = h if w is None else w
    xy = coord_grid(h, w)
    img = np.zeros(xy.shape[0], dtype=np.float64)
    for p0, p1 in _SHAPES[name]:
        img = np.maximum(img, _stroke(xy, p0, p1, width))
    return img


def stroke_tasks(h, w=None, names=None):
    """A stack of (n_tasks, h*w) fields to meta-train MetaQuill on."""
    names = SHAPE_NAMES if names is None else names
    return np.stack([stroke_field(n, h, w) for n in names], axis=0)


def target_field(name, h, w=None):
    """Look up any of the toy fields by name."""
    if name == "blob":
        return gaussian_blob(h, w)
    if name == "checker":
        return checkerboard(h, w)
    if name in _SHAPES:
        return stroke_field(name, h, w)
    raise ValueError(f"unknown field {name!r}")


def random_mask(n, fraction, seed):
    """Indices of the observed pixels o, drawn uniformly (Sec. IV-C2)."""
    rng = np.random.default_rng(seed)
    m = max(1, int(round(fraction * n)))
    return np.sort(rng.choice(n, size=m, replace=False))


def center_hole_mask(h, w, radius):
    """The set L_{r^2} of Sec. II-A: everything outside a centered circle."""
    xy = coord_grid(h, w)
    keep = np.linalg.norm(xy, axis=-1) >= radius
    return np.nonzero(keep)[0]
