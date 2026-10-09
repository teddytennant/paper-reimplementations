"""Closed-form RGBD mesh refinement (arXiv:2610.11187), CPU stand-in."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.optimize import linear_sum_assignment

jax.config.update("jax_enable_x64", True)

DEPTH_THRESHOLD = 0.1
SYMMETRY_THRESHOLD = 0.2
COT_CLAMP = 50.0
GAMMA = 1e-3
TAU_SCALE = 1e-6
TAUBIN_STEPS = 5
TAUBIN_LAMBDA = 0.5
TAUBIN_NU = -0.5


def _as_array(vertices):
    return np.asarray(vertices, dtype=np.float64)


def _faces(faces):
    return np.asarray(faces, dtype=np.int32)


def _intrinsics(intrinsics):
    if isinstance(intrinsics, dict):
        return (
            float(intrinsics["fx"]),
            float(intrinsics["fy"]),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
        )
    fx, fy, cx, cy = intrinsics
    return float(fx), float(fy), float(cx), float(cy)


def project(points, intrinsics):
    """Pixel coordinates of camera-frame points. Points in front have z < 0."""
    fx, fy, cx, cy = _intrinsics(intrinsics)
    pts = _as_array(points)
    depth = -pts[:, 2]
    xy = np.stack(
        [fx * pts[:, 0] / depth + cx, fy * pts[:, 1] / depth + cy],
        axis=1,
    )
    return xy, depth


def back_project(pixels, depth, intrinsics):
    """Eq. (2). pixels are (x, y) in the same space as the intrinsics."""
    fx, fy, cx, cy = _intrinsics(intrinsics)
    q = _as_array(pixels)
    d = _as_array(depth).reshape(-1)
    return np.stack(
        [d * (q[:, 0] - cx) / fx, d * (q[:, 1] - cy) / fy, -d],
        axis=1,
    )


def _barycentric(px, py, xs, ys):
    x0, x1, x2 = xs
    y0, y1, y2 = ys
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(den) < 1e-12:
        return None
    a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
    b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
    c = 1.0 - a - b
    return a, b, c


def rasterize(vertices, faces, height, width, intrinsics):
    """Software z-buffer. Depth is -z. Nearer (smaller depth) wins. Background is 0, face -1."""
    verts = _as_array(vertices)
    tris = _faces(faces)
    height = int(height)
    width = int(width)
    depth_out = np.zeros((height, width), dtype=np.float64)
    face_out = np.full((height, width), -1, dtype=np.int32)
    xy, depth = project(verts, intrinsics)
    in_front = depth > 1e-8
    for fi, tri in enumerate(tris):
        if not in_front[tri].all():
            continue
        xs = xy[tri, 0]
        ys = xy[tri, 1]
        ds = depth[tri]
        c0 = max(int(np.floor(xs.min())), 0)
        c1 = min(int(np.ceil(xs.max())), width - 1)
        r0 = max(int(np.floor(ys.min())), 0)
        r1 = min(int(np.ceil(ys.max())), height - 1)
        if c0 > c1 or r0 > r1:
            continue
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                bc = _barycentric(c + 0.5, r + 0.5, xs, ys)
                if bc is None or min(bc) < -1e-6:
                    continue
                inv = bc[0] / ds[0] + bc[1] / ds[1] + bc[2] / ds[2]
                if inv <= 1e-12:
                    continue
                frag = 1.0 / inv
                if depth_out[r, c] == 0.0 or frag < depth_out[r, c]:
                    depth_out[r, c] = frag
                    face_out[r, c] = fi
    return depth_out, face_out


def visible_indices(faces, face_map, rendered_depth):
    tris = _faces(faces)
    hit = face_map[rendered_depth > 0]
    hit = hit[hit >= 0]
    if hit.size == 0:
        return np.zeros(0, dtype=np.int32)
    return np.unique(tris[np.unique(hit)].ravel())


def depth_points(depth, intrinsics):
    image = _as_array(depth)
    ys, xs = np.where(image > 0)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pixels = np.stack([xs + 0.5, ys + 0.5], axis=1)
    return back_project(pixels, image[ys, xs], intrinsics)


def depth_displacements(vertices, faces, depth, intrinsics, threshold=DEPTH_THRESHOLD):
    """Eq. (5). Returns displacements and the matched vertex indices."""
    verts = _as_array(vertices)
    image = _as_array(depth)
    rendered, face_map = rasterize(verts, faces, image.shape[0], image.shape[1], intrinsics)
    visible = visible_indices(faces, face_map, rendered)
    points = depth_points(image, intrinsics)
    disp = np.zeros_like(verts)
    if visible.size == 0 or points.shape[0] == 0:
        return disp, np.zeros(0, dtype=np.int32)
    cost = np.linalg.norm(verts[visible, None, :] - points[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    matched = []
    for row, col in zip(rows, cols):
        if cost[row, col] > threshold:
            continue
        vi = int(visible[row])
        disp[vi] = points[col] - verts[vi]
        matched.append(vi)
    return disp, np.asarray(matched, dtype=np.int32)


def _plane_normal(azimuth_deg, elevation_deg):
    """Elevation is tilt from vertical. 0 is a vertical plane, normal in the XZ plane."""
    theta = np.deg2rad(azimuth_deg)
    alpha = np.deg2rad(90.0 - elevation_deg)
    return np.array(
        [
            np.sin(alpha) * np.cos(theta),
            np.cos(alpha),
            np.sin(alpha) * np.sin(theta),
        ],
        dtype=np.float64,
    )


def default_candidates():
    azimuth = np.arange(0.0, 180.0, 15.0)
    elevation = np.array([0.0, 15.0, 30.0])
    return [(float(a), float(e)) for a in azimuth for e in elevation]


def _plane_frame(normal):
    n = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def _signed(vertices, normal):
    n = normal / np.linalg.norm(normal)
    centroid = vertices.mean(axis=0)
    offset = -float(np.dot(n, centroid))
    signed = vertices @ n + offset
    projected = vertices - signed[:, None] * n
    return n, offset, signed, projected, centroid


def _splat_score(coords, signed, resolution):
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - 0.05 * span
    hi = hi + 0.05 * span
    span = hi - lo
    pix = np.floor((coords - lo) / span * (resolution - 1)).astype(np.int32)
    pix = np.clip(pix, 0, resolution - 1)
    plus = np.zeros((resolution, resolution), dtype=np.float64)
    minus = np.zeros((resolution, resolution), dtype=np.float64)
    for (u, v), s in zip(pix, signed):
        if s > 1e-10:
            plus[v, u] += s
        elif s < -1e-10:
            minus[v, u] += s
    plus_f = uniform_filter(plus, size=5, mode="constant")
    minus_f = uniform_filter(minus, size=5, mode="constant")
    both = (plus_f != 0.0) & (minus_f != 0.0)
    if not np.any(both):
        return np.inf
    return float(np.mean(np.abs(plus_f[both] + minus_f[both])))


def detect_symmetry(vertices, candidates=None, resolution=64):
    """Eqs. (6)-(9). Returns unit normal, offset, and score. Plane passes through the centroid."""
    verts = _as_array(vertices)
    if candidates is None:
        candidates = default_candidates()
    best = None
    for azimuth, elevation in candidates:
        normal = _plane_normal(azimuth, elevation)
        n, offset, signed, projected, centroid = _signed(verts, normal)
        e1, e2 = _plane_frame(n)
        rel = projected - centroid
        coords = np.stack([rel @ e1, rel @ e2], axis=1)
        score = _splat_score(coords, signed, resolution)
        if best is None or score < best[0]:
            best = (score, n, offset)
    return best[1], best[2], best[0]


def symmetry_displacements(
    vertices,
    depth_disp,
    matched,
    normal,
    offset,
    threshold=SYMMETRY_THRESHOLD,
):
    """Eqs. (10)-(13). Does not write a target that already has a depth displacement."""
    verts = _as_array(vertices)
    disp = np.zeros_like(verts)
    matched = np.asarray(matched, dtype=np.int32)
    matched_set = set(int(i) for i in matched)
    n = normal / np.linalg.norm(normal)
    signed = verts @ n + float(offset)
    e1, e2 = _plane_frame(n)
    centroid = verts.mean(axis=0)
    projected = verts - signed[:, None] * n
    rel = projected - centroid
    phi = np.stack([rel @ e1, rel @ e2, np.abs(signed)], axis=1)
    left = np.array([i for i in matched if signed[i] > 1e-10], dtype=np.int32)
    right = np.flatnonzero(signed < -1e-10).astype(np.int32)
    applied = []
    if left.size == 0 or right.size == 0:
        return disp, np.asarray(applied, dtype=np.int32)
    cost = np.linalg.norm(phi[left, None, :] - phi[right, None, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    corrected = verts + _as_array(depth_disp)
    for row, col in zip(rows, cols):
        if cost[row, col] > threshold:
            continue
        target = int(right[col])
        if target in matched_set:
            continue
        source = corrected[int(left[row])]
        reflected = source - 2.0 * (float(np.dot(n, source)) + float(offset)) * n
        disp[target] = reflected - verts[target]
        applied.append(target)
    return disp, np.asarray(applied, dtype=np.int32)


def _cotangent_weights(vertices, faces):
    n = vertices.shape[0]
    weight = np.zeros((n, n), dtype=np.float64)
    for tri in _faces(faces):
        i, j, k = (int(x) for x in tri)
        corners = (i, j, k)
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            u = vertices[b] - vertices[a]
            v = vertices[c] - vertices[a]
            area2 = np.linalg.norm(np.cross(u, v))
            if area2 < 1e-12:
                cot = 0.0
            else:
                cot = float(np.dot(u, v) / area2)
                cot = float(np.clip(cot, 0.0, COT_CLAMP))
            # angle at a contributes to the opposite edge bc
            weight[b, c] += 0.5 * cot
            weight[c, b] += 0.5 * cot
        del corners
    return weight


def _boundary(faces, n):
    counts = {}
    for tri in _faces(faces):
        a, b, c = (int(x) for x in tri)
        for i, j in ((a, b), (b, c), (c, a)):
            edge = (i, j) if i < j else (j, i)
            counts[edge] = counts.get(edge, 0) + 1
    bound = set()
    for (i, j), count in counts.items():
        if count == 1:
            bound.add(i)
            bound.add(j)
    return bound


def laplacian(vertices, faces):
    weight = _cotangent_weights(_as_array(vertices), faces)
    degree = weight.sum(axis=1)
    matrix = -weight
    np.fill_diagonal(matrix, degree)
    return matrix


def propagate(vertices, faces, displacements, handles):
    """Eq. (14) with Tikhonov tau and boundary anchor gamma. Handles stay exact."""
    verts = _as_array(vertices)
    disp = _as_array(displacements)
    n = verts.shape[0]
    handle = np.unique(np.asarray(handles, dtype=np.int32))
    handle = handle[(handle >= 0) & (handle < n)]
    solved = disp.copy()
    if handle.size == 0 or handle.size == n:
        return solved
    free_mask = np.ones(n, dtype=bool)
    free_mask[handle] = False
    free = np.flatnonzero(free_mask)
    matrix = laplacian(verts, faces)
    lff = matrix[np.ix_(free, free)].copy()
    lfh = matrix[np.ix_(free, handle)]
    diag = np.diag(lff)
    tau = TAU_SCALE * float(np.mean(diag)) if diag.size else 0.0
    if not np.isfinite(tau) or tau < 0.0:
        tau = 0.0
    lff[np.arange(free.size), np.arange(free.size)] += tau
    bound = _boundary(faces, n)
    for local, vi in enumerate(free):
        if int(vi) in bound:
            lff[local, local] += GAMMA
    rhs = -lfh @ disp[handle]
    free_u = np.asarray(jnp.linalg.solve(jnp.asarray(lff), jnp.asarray(rhs)))
    solved[free] = free_u
    solved[handle] = disp[handle]
    return solved


def _neighbors(faces, n):
    nbrs = [set() for _ in range(n)]
    for tri in _faces(faces):
        a, b, c = (int(x) for x in tri)
        for i, j in ((a, b), (b, c), (c, a)):
            nbrs[i].add(j)
            nbrs[j].add(i)
    return [np.asarray(sorted(s), dtype=np.int32) for s in nbrs]


def _umbrella(vertices, neighbors):
    out = np.zeros_like(vertices)
    for i, nbr in enumerate(neighbors):
        if nbr.size == 0:
            continue
        out[i] = vertices[nbr].mean(axis=0) - vertices[i]
    return out


def taubin(vertices, faces, steps=TAUBIN_STEPS, lam=TAUBIN_LAMBDA, nu=TAUBIN_NU):
    verts = _as_array(vertices)
    neighbors = _neighbors(faces, verts.shape[0])
    for _ in range(int(steps)):
        verts = verts + lam * _umbrella(verts, neighbors)
        verts = verts + nu * _umbrella(verts, neighbors)
    return verts


def refine(
    vertices,
    faces,
    depth,
    intrinsics,
    mode="full",
    depth_threshold=DEPTH_THRESHOLD,
    symmetry_threshold=SYMMETRY_THRESHOLD,
    candidates=None,
    plane=None,
):
    """Full pipeline, or mode='visible' for depth handles plus five Taubin steps."""
    verts = _as_array(vertices)
    depth_disp, matched = depth_displacements(
        verts, faces, depth, intrinsics, threshold=depth_threshold
    )
    if mode == "visible":
        field = propagate(verts, faces, depth_disp, matched)
        return taubin(verts + field, faces)
    if plane is None:
        normal, offset, _score = detect_symmetry(verts, candidates=candidates)
    else:
        normal, offset = plane
        normal = _as_array(normal)
        offset = float(offset)
    # Prefer the half that actually received depth matches as the source.
    signed = verts @ (normal / np.linalg.norm(normal)) + offset
    pos = sum(1 for i in matched if signed[i] > 0)
    neg = sum(1 for i in matched if signed[i] < 0)
    if neg > pos:
        normal = -normal
        offset = -offset
    sym_disp, sym_handles = symmetry_displacements(
        verts,
        depth_disp,
        matched,
        normal,
        offset,
        threshold=symmetry_threshold,
    )
    field = depth_disp + sym_disp
    handles = np.concatenate([matched, sym_handles]) if sym_handles.size else matched
    solved = propagate(verts, faces, field, handles)
    return verts + solved
