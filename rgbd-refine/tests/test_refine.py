import numpy as np

from rgbd_refine import (
    back_project,
    depth_displacements,
    detect_symmetry,
    project,
    propagate,
    rasterize,
    refine,
    symmetry_displacements,
    taubin,
)
from rgbd_refine.refine import laplacian, visible_indices


def _sphere(n_lat=5, n_lon=8, radius=0.8):
    verts = [[0.0, radius, 0.0], [0.0, -radius, 0.0]]
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            verts.append(
                [
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.cos(phi),
                    radius * np.sin(phi) * np.sin(theta),
                ]
            )
    verts = np.asarray(verts, dtype=np.float64)
    faces = []
    for j in range(n_lon):
        a = 2 + j
        b = 2 + (j + 1) % n_lon
        faces.append([0, b, a])
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a = 2 + i * n_lon + j
            b = 2 + i * n_lon + (j + 1) % n_lon
            c = a + n_lon
            d = b + n_lon
            faces.append([a, b, d])
            faces.append([a, d, c])
    last = 2 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        a = last + j
        b = last + (j + 1) % n_lon
        faces.append([1, a, b])
    return verts, np.asarray(faces, dtype=np.int32)


def _sheared_sphere():
    """X-mirror only. A sphere is symmetric in every plane through the origin."""
    verts, faces = _sphere()
    out = verts.copy()
    out[:, 1] = verts[:, 1] + 0.35 * verts[:, 2]
    out[:, 2] = 0.65 * verts[:, 2]
    return out, faces


def _grid(n=5):
    xs = np.linspace(-0.5, 0.5, n)
    ys = np.linspace(-0.5, 0.5, n)
    verts = np.array([[x, y, -2.0] for y in ys for x in xs], dtype=np.float64)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.asarray(faces, dtype=np.int32)


def test_back_project_round_trip():
    k = (80.0, 75.0, 32.0, 30.0)
    point = np.array([[0.3, -0.2, -2.5]])
    xy, depth = project(point, k)
    recovered = back_project(xy, depth, k)
    assert np.allclose(recovered, point, atol=1e-5)


def test_rasterizer_nearer_wins():
    k = (40.0, 40.0, 16.0, 16.0)
    near = np.array([[-0.4, -0.3, -2.0], [0.5, -0.2, -2.0], [0.0, 0.5, -2.0]])
    far = near.copy()
    far[:, 2] = -4.0
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    d_near, f_near = rasterize(near, faces, 32, 32, k)
    both = np.vstack([near, far])
    faces_both = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    d_both, f_both = rasterize(both, faces_both, 32, 32, k)
    covered = f_near >= 0
    assert covered.sum() > 10
    assert np.all(f_both[covered] == 0)
    assert np.allclose(d_both[covered], d_near[covered])
    assert d_near[0, 0] == 0.0


def test_depth_stage_moves_close_matches_only():
    k = (80.0, 80.0, 32.0, 32.0)
    verts, faces = _sphere()
    verts = verts.copy()
    verts[:, 2] -= 2.5
    depth, face_map = rasterize(verts, faces, 64, 64, k)
    visible = visible_indices(faces, face_map, depth)
    ys, xs = np.where(depth > 0)
    points = back_project(np.stack([xs + 0.5, ys + 0.5], axis=1), depth[ys, xs], k)
    dist = np.linalg.norm(verts[visible, None, :] - points[None, :, :], axis=2).min(axis=1)
    close_pool = visible[dist < 0.03]
    assert close_pool.size >= 4
    close = close_pool[:3]
    far = close_pool[3]
    shifted = verts.copy()
    shifted[close, 0] += 0.04
    shifted[far, 0] += 0.3
    disp, matched = depth_displacements(shifted, faces, depth, k, threshold=0.1)
    matched_set = set(int(i) for i in matched)
    for i in close:
        assert int(i) in matched_set
        assert np.linalg.norm(disp[i]) > 1e-4
        assert disp[i, 0] < 0.0
    assert int(far) not in matched_set
    assert np.allclose(disp[far], 0.0)
    hidden = [i for i in range(len(verts)) if i not in set(int(v) for v in visible)]
    assert hidden
    assert np.allclose(disp[hidden[0]], 0.0)


def test_propagation_keeps_handles_and_smooths():
    verts, faces = _grid()
    n = len(verts)
    side = 5
    corners = [0, side - 1, n - side, n - 1]
    neighbor = 2 * side + 1  # next to the center, not on the boundary ring's far corner
    handles = corners + [neighbor]
    disp = np.zeros_like(verts)
    disp[neighbor] = np.array([0.15, 0.0, 0.0])
    field = propagate(verts, faces, disp, handles)
    assert np.allclose(field[handles], disp[handles], atol=1e-5)
    center = (side // 2) * side + side // 2
    assert np.linalg.norm(field[center]) > 1e-4
    matrix = laplacian(verts, faces)
    counts = {}
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(i), int(j))))
            counts[edge] = counts.get(edge, 0) + 1
    bound = set()
    for (i, j), count in counts.items():
        if count == 1:
            bound.add(i)
            bound.add(j)
    free_interior = [i for i in range(n) if i not in handles and i not in bound]
    solved_res = np.linalg.norm(matrix[free_interior] @ field)
    raw_res = np.linalg.norm(matrix[free_interior] @ disp)
    assert raw_res > 1e-4
    assert solved_res < 0.05 * raw_res


def test_symmetry_detector_and_transfer():
    rng = np.random.default_rng(0)
    right = rng.normal(size=(40, 3))
    right[:, 0] = np.abs(right[:, 0]) + 0.2
    cloud = np.vstack([right, right * np.array([-1.0, 1.0, 1.0])])
    normal, _offset, score = detect_symmetry(cloud)
    assert abs(float(np.dot(normal, [1.0, 0.0, 0.0]))) > np.cos(np.deg2rad(15.0))
    assert score < 1e-6

    verts, _faces = _sphere(radius=1.0)
    good = np.flatnonzero(verts[:, 0] > 0.05)
    bad = np.flatnonzero(verts[:, 0] < -0.05)
    corrupted = verts.copy()
    corrupted[bad, 0] -= 0.05
    disp, applied = symmetry_displacements(
        corrupted,
        np.zeros_like(verts),
        good,
        np.array([1.0, 0.0, 0.0]),
        0.0,
        threshold=0.2,
    )
    assert applied.size > 0
    moved = [i for i in applied if i in set(int(b) for b in bad)]
    assert moved
    before = np.mean(np.abs(corrupted[moved, 0] - verts[moved, 0]))
    after = np.mean(np.abs((corrupted[moved] + disp[moved])[:, 0] - verts[moved, 0]))
    assert after < before
    protected = int(bad[0])
    depth_disp = np.zeros_like(verts)
    depth_disp[protected] = np.array([-0.02, 0.0, 0.0])
    disp2, applied2 = symmetry_displacements(
        corrupted,
        depth_disp,
        np.concatenate([good, [protected]]),
        np.array([1.0, 0.0, 0.0]),
        0.0,
    )
    assert protected not in set(int(i) for i in applied2)
    assert np.allclose(disp2[protected], 0.0)


def test_symmetry_match_allows_unequal_sides():
    # The reviewer's call: 2 positive matched vertices, 3 negative targets.
    verts = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.05, 0.0, 0.0],
            [-1.05, 1.0, 0.0],
            [-1.05, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    disp, applied = symmetry_displacements(
        verts,
        np.zeros_like(verts),
        matched=[0, 1],
        normal=np.array([1.0, 0.0, 0.0]),
        offset=0.0,
    )
    assert set(int(i) for i in applied) == {2, 3}
    assert np.allclose(verts[applied] + disp[applied], [[-1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
    assert np.allclose(disp[4], 0.0)

    # The other rectangle, 3 positive and 2 negative, must broadcast too.
    wide = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [-1.05, 0.0, 0.0],
            [-1.05, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    disp2, applied2 = symmetry_displacements(
        wide,
        np.zeros_like(wide),
        matched=[0, 1, 2],
        normal=np.array([1.0, 0.0, 0.0]),
        offset=0.0,
    )
    assert set(int(i) for i in applied2) == {3, 4}


def test_full_pipeline_reduces_error():
    k = (160.0, 160.0, 64.0, 64.0)
    true, faces = _sheared_sphere()
    true = true.copy()
    true[:, 2] -= 2.5
    depth, _ = rasterize(true, faces, 128, 128, k)
    corrupted = true.copy()
    corrupted[corrupted[:, 0] < -0.05, 0] -= 0.06
    refined = refine(corrupted, faces, depth, k)
    before = np.mean(np.linalg.norm(corrupted - true, axis=1))
    after = np.mean(np.linalg.norm(refined - true, axis=1))
    assert after < before


def test_visible_only_is_finite_and_bounded():
    k = (80.0, 80.0, 32.0, 32.0)
    verts, faces = _sphere()
    verts = verts.copy()
    verts[:, 2] -= 2.5
    depth, face_map = rasterize(verts, faces, 64, 64, k)
    visible = set(int(i) for i in visible_indices(faces, face_map, depth))
    corrupted = verts.copy()
    corrupted[:, 0] += 0.03
    out = refine(corrupted, faces, depth, k, mode="visible")
    assert np.isfinite(out).all()
    hidden = [i for i in range(len(verts)) if i not in visible]
    assert np.max(np.linalg.norm(out[hidden] - corrupted[hidden], axis=1)) < 0.25
    assert np.isfinite(taubin(corrupted, faces)).all()
