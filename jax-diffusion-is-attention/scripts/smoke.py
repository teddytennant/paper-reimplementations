"""Accelerator smoke check. Prints the backend and a few scalars, exits 1 on failure.

Sizes scale with the backend, so this runs in a few seconds on CPU and gives a
GPU something to chew on. Nothing here is a benchmark.

    python scripts/smoke.py
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp

from diffusion_is_attention import (
    SimilarityTree,
    attention_weights,
    dot_product_attention,
    exact_topk,
    figure2_setup,
    generate,
    mixture_log_density,
    optimal_denoiser,
    reverse_attention_trajectory,
    score,
    sigma_schedule,
    squared_distances,
    weight_entropy,
)

FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def clustered_sphere(key, n, dim, per_cluster=32, spread=0.05):
    """Clustered points, then projected to the unit sphere.

    Two reasons for the sphere: eq (12) is then exact, and the norms are all
    equal so the closed form has nothing but the angles to work with. Clusters
    matter because in 32 or 128 isotropic dimensions every point is the same
    distance from every other one and eq (11) never sparsifies.
    """
    kc, kj = jax.random.split(key)
    n_clusters = max(n // per_cluster, 2)
    centers = jax.random.normal(kc, (n_clusters, dim))
    centers = centers / jnp.linalg.norm(centers, axis=-1, keepdims=True)
    pts = centers[:, None, :] + spread * jax.random.normal(
        kj, (n_clusters, per_cluster, dim)
    )
    pts = pts.reshape(-1, dim)
    return pts / jnp.linalg.norm(pts, axis=-1, keepdims=True), centers


def main():
    platform = jax.default_backend()
    print(f"backend: {platform}, devices: {jax.devices()}")
    big = platform != "cpu"
    n = 16384 if big else 2048
    dim = 128 if big else 32
    batch = 2048 if big else 256
    n_latents = 4096 if big else 512
    print(f"sizes: dataset {n} x {dim}, {batch} queries, {n_latents} latents")

    kd, kq, kj, kl, kn = jax.random.split(jax.random.key(0), 5)
    data, centers = clustered_sphere(kd, n, dim)
    idx = jax.random.randint(kq, (batch,), 0, centers.shape[0])
    queries = centers[idx] + 0.05 * jax.random.normal(kj, (batch, dim))
    queries = queries / jnp.linalg.norm(queries, axis=-1, keepdims=True)

    # Sec. II-D, a batched reverse sigma sweep. The published schedule stops at
    # 0.37, which is nowhere near small enough to sparsify in this many
    # dimensions, so this one keeps going.
    sigmas = sigma_schedule(sigma_start=1.0, ratio=0.94, num_steps=100)
    step = jax.jit(optimal_denoiser)
    t0 = time.perf_counter()
    z = queries
    for sigma in sigmas:
        z = step(z, data, sigma)
    z.block_until_ready()
    sweep_s = time.perf_counter() - t0

    weights = jax.jit(attention_weights)
    h_start = float(jnp.mean(weight_entropy(weights(queries, data, sigmas[0]))))
    h_end = float(jnp.mean(weight_entropy(weights(z, data, sigmas[-1]))))
    nearest = jnp.min(squared_distances(z, data), axis=-1)
    print(f"reverse sweep: {len(sigmas)} steps, sigma {float(sigmas[0]):.2f} -> "
          f"{float(sigmas[-1]):.4f}, {sweep_s:.2f}s")
    print(f"  weight entropy {h_start:.4f} -> {h_end:.4f} nats "
          f"(uniform would be {float(jnp.log(n)):.4f})")
    print(f"  worst gap from the final iterate to a signal: "
          f"{float(jnp.sqrt(jnp.max(nearest))):.4f}")
    check("entropy falls as sigma falls", h_end < 0.1 * h_start)
    check("sweep output is finite", bool(jnp.all(jnp.isfinite(z))))
    check("sweep lands on the data", float(jnp.sqrt(jnp.max(nearest))) < 1e-3)

    # Sec. II-C, eq (12) is exact on the unit sphere.
    err = jnp.max(jnp.abs(
        dot_product_attention(queries[:64], data, 0.3)
        - optimal_denoiser(queries[:64], data, 0.3)
    ))
    print(f"eq (12) vs eq (11) on unit-norm data: max abs error {float(err):.3e}")
    check("eq (12) matches eq (11) on the sphere", float(err) < 1e-4)

    # Sec. II-B, eq (4) against autodiff of eq (2).
    s_err = jnp.linalg.norm(
        score(queries[0], data[:256], 0.4)
        - jax.grad(mixture_log_density)(queries[0], data[:256], 0.4)
    )
    print(f"eq (4) vs grad log p_t: {float(s_err):.3e}")
    check("score matches autodiff", float(s_err) < 1e-3)

    # Sec. II-D, Figure 2.
    fig_data, fig_y, fig_mean = figure2_setup()
    traj = reverse_attention_trajectory(fig_y, fig_data, sigma_schedule())
    to_mean = jnp.linalg.norm(traj.points - fig_mean, axis=-1)
    gaps = jnp.linalg.norm(fig_data - traj.points[-1], axis=-1)
    print(f"figure 2: nearest approach to M {float(jnp.min(to_mean)):.4f}, "
          f"final gap to a signal {float(jnp.min(gaps)):.4f}")
    check("figure 2 passes the mean", float(jnp.min(to_mean)) < 0.1)
    check("figure 2 ends on a signal", float(jnp.min(gaps)) < 0.02)

    # Sec. IV.
    tree = SimilarityTree(data, branching=4, leaf_size=64)
    q = queries[0] / jnp.linalg.norm(queries[0])
    hit = tree.query(q, k=16, max_leaves=8)
    want = set(exact_topk(data, q, 16).indices.tolist())
    recall = len(set(hit.indices.tolist()) & want) / 16
    print(f"tree: depth {tree.depth}, {hit.distance_evals} distance evals of {n}, "
          f"recall {recall:.2f}")
    check("tree query is sublinear", hit.distance_evals < n // 2)
    check("tree recall is high", recall > 0.75)

    # Sec. V, Algorithm 1.
    lat = jax.random.normal(kl, (n_latents, 3, 7, 7))
    t0 = time.perf_counter()
    out = generate(lat, lat[0], k=3, sigma=0.01, eta=0.01, key=kn)
    out.latent.block_until_ready()
    gen_s = time.perf_counter() - t0
    norm = float(jnp.linalg.norm(out.rescaled_latent))
    wsum = float(jnp.mean(jnp.sum(out.weights, axis=0)))
    print(f"algorithm 1: {gen_s:.2f}s, r = {float(out.radius):.4f}, "
          f"||z|| after line 8 = {norm:.4f}, weight sums = {wsum:.6f}")
    check("line 8 rescales to r", abs(norm - float(out.radius)) < 1e-3 * float(out.radius))
    check("alg 2 weights normalize", abs(wsum - 1.0) < 1e-5)
    check("generated image is finite", bool(jnp.all(jnp.isfinite(out.image))))

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
