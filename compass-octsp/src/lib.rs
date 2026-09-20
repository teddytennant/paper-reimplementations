//! COMPASS for Ordered Clustered TSP (OCTSP), arXiv:2609.20352.
//!
//! Exact special case of Algorithm 1: candidate endpoint pairs, Held-Karp
//! fixed-endpoint sub-TSP, layered DAG shortest path. Geometry + greedy-bridge
//! candidates from Appendix C.1; the GPU triplet-optimization source is omitted.

/// λ in score(s, t) = λ · linearity(s, t) − bridge_cost(s, t) (Appendix C.1).
pub const LINEARITY_WEIGHT: f64 = 0.1;

#[derive(Clone, Debug)]
pub struct OctspInstance {
    /// Possibly asymmetric distance matrix, `distances[u][v] = d(u, v)`.
    pub distances: Vec<Vec<f64>>,
    /// Clusters `C1..Cm` in the prescribed visit order. Node indices into `distances`.
    pub clusters: Vec<Vec<usize>>,
    /// Open-path depot S (dummy source of the layered DAG).
    pub source: usize,
    /// Open-path depot T (dummy sink of the layered DAG).
    pub target: usize,
}

#[derive(Clone, Debug)]
pub struct Solution {
    /// Hamiltonian S-T path visiting every cluster node once, clusters in order.
    pub tour: Vec<usize>,
    pub cost: f64,
    /// Winning `(s_i, t_i)` endpoint pair per cluster.
    pub endpoints: Vec<(usize, usize)>,
}

/// Held-Karp DP for an open Hamiltonian path on `nodes` from `s` to `t`.
///
/// Invalid when `s == t` and `|nodes| > 1`: returns `(∞, [])`.
pub fn held_karp_path(dist: &[Vec<f64>], nodes: &[usize], s: usize, t: usize) -> (f64, Vec<usize>) {
    let n = nodes.len();
    if n == 0 {
        return (f64::INFINITY, vec![]);
    }
    let s_pos = match nodes.iter().position(|&x| x == s) {
        Some(p) => p,
        None => return (f64::INFINITY, vec![]),
    };
    let t_pos = match nodes.iter().position(|&x| x == t) {
        Some(p) => p,
        None => return (f64::INFINITY, vec![]),
    };
    if s == t {
        if n == 1 {
            return (0.0, vec![s]);
        }
        return (f64::INFINITY, vec![]);
    }

    let full = 1usize << n;
    let mut dp = vec![vec![f64::INFINITY; n]; full];
    let mut parent = vec![vec![usize::MAX; n]; full];
    dp[1 << s_pos][s_pos] = 0.0;

    for mask in 0..full {
        for j in 0..n {
            if mask & (1 << j) == 0 {
                continue;
            }
            let cur = dp[mask][j];
            if !cur.is_finite() {
                continue;
            }
            for k in 0..n {
                if mask & (1 << k) != 0 {
                    continue;
                }
                let nmask = mask | (1 << k);
                let cand = cur + dist[nodes[j]][nodes[k]];
                if cand < dp[nmask][k] {
                    dp[nmask][k] = cand;
                    parent[nmask][k] = j;
                }
            }
        }
    }

    let cost = dp[full - 1][t_pos];
    if !cost.is_finite() {
        return (f64::INFINITY, vec![]);
    }

    let mut path = Vec::with_capacity(n);
    let mut mask = full - 1;
    let mut cur = t_pos;
    loop {
        path.push(nodes[cur]);
        if cur == s_pos && mask == (1 << s_pos) {
            break;
        }
        let p = parent[mask][cur];
        if p == usize::MAX {
            return (f64::INFINITY, vec![]);
        }
        mask ^= 1 << cur;
        cur = p;
    }
    path.reverse();
    (cost, path)
}

/// Connect-first baseline (Appendix C.3): greedy bridges, then open TSP per cluster.
///
/// For adjacent clusters pick `(t_i, s_{i+1})` minimizing `d(t_i, s_{i+1})`.
/// Since `s` and `t` must differ, `s_{i+1}` is excluded from candidates for `t_{i+1}`.
/// With depots, `S → C1` and `Cm → T` are treated as the first and last bridges.
pub fn connect_first(instance: &OctspInstance) -> Solution {
    let endpoints = connect_first_endpoints(instance);
    assemble_from_endpoints(instance, &endpoints)
}

/// COMPASS with exhaustive `P_i` (all valid ordered pairs) and Held-Karp.
/// Corollary 2: globally optimal OCTSP.
pub fn compass_exact(instance: &OctspInstance) -> Solution {
    let pairs: Vec<Vec<(usize, usize)>> = instance
        .clusters
        .iter()
        .map(|c| exhaustive_pairs(c))
        .collect();
    solve_with_pairs(instance, &pairs)
}

/// COMPASS Algorithm 1 with at most `k` candidate endpoint pairs per cluster.
///
/// `P_i` is assembled from (a) geometry-based scoring (Appendix C.1, λ = 0.1)
/// and (c) the Connect-first greedy-bridge pair. Source (b), GPU triplet
/// optimization, is not implemented.
pub fn compass(instance: &OctspInstance, k: usize) -> Solution {
    let pairs = candidate_pairs(instance, k);
    let mut sol = solve_with_pairs(instance, &pairs);
    sol = post_optimize(instance, &sol);
    sol
}

/// Re-solve winning endpoint pairs. With Held-Karp this is a no-op on cost.
pub fn post_optimize(instance: &OctspInstance, solution: &Solution) -> Solution {
    assemble_from_endpoints(instance, &solution.endpoints)
}

/// Candidate endpoint pairs `P_i` of size up to `k` for every cluster.
///
/// Always includes the Connect-first pair so COMPASS is never worse-informed
/// than Connect-first on any single bridge (Appendix C.1 (c)).
pub fn candidate_pairs(instance: &OctspInstance, k: usize) -> Vec<Vec<(usize, usize)>> {
    let m = instance.clusters.len();
    if m == 0 {
        return Vec::new();
    }
    let greedy = connect_first_endpoints(instance);
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        out.push(pairs_for_cluster(instance, i, k, greedy[i]));
    }
    out
}

/// M-reduction: `d'(u,v) = d(u,v) + M · 1[cluster(u) > cluster(v)]`.
/// Not used by the DAG solver; provided for the paper's flatten-to-ATSP reduction.
pub fn m_reduction(instance: &OctspInstance, m_penalty: f64) -> Vec<Vec<f64>> {
    let n = instance.distances.len();
    let mut cluster_of = vec![None; n];
    for (i, c) in instance.clusters.iter().enumerate() {
        for &u in c {
            if u < n {
                cluster_of[u] = Some(i);
            }
        }
    }
    let mut d = instance.distances.clone();
    for u in 0..n {
        for v in 0..n {
            if let (Some(cu), Some(cv)) = (cluster_of[u], cluster_of[v]) {
                if cu > cv {
                    d[u][v] += m_penalty;
                }
            }
        }
    }
    d
}

fn exhaustive_pairs(cluster: &[usize]) -> Vec<(usize, usize)> {
    if cluster.len() == 1 {
        return vec![(cluster[0], cluster[0])];
    }
    let mut out = Vec::with_capacity(cluster.len() * cluster.len().saturating_sub(1));
    for &s in cluster {
        for &t in cluster {
            if s != t {
                out.push((s, t));
            }
        }
    }
    out
}

fn connect_first_endpoints(instance: &OctspInstance) -> Vec<(usize, usize)> {
    let m = instance.clusters.len();
    if m == 0 {
        return Vec::new();
    }
    let d = &instance.distances;
    let mut s = vec![0usize; m];
    let mut t = vec![0usize; m];

    s[0] = argmin_from(instance.source, &instance.clusters[0], d);

    for i in 0..m - 1 {
        let mut best = f64::INFINITY;
        let mut best_t = s[i];
        let mut best_s = instance.clusters[i + 1][0];
        for &ti in &instance.clusters[i] {
            if ti == s[i] && instance.clusters[i].len() > 1 {
                continue;
            }
            for &sj in &instance.clusters[i + 1] {
                let cost = d[ti][sj];
                if cost < best {
                    best = cost;
                    best_t = ti;
                    best_s = sj;
                }
            }
        }
        t[i] = best_t;
        s[i + 1] = best_s;
    }

    let last = m - 1;
    t[last] = argmin_to_excluding(
        &instance.clusters[last],
        instance.target,
        s[last],
        instance.clusters[last].len() > 1,
        d,
    );
    s.into_iter().zip(t).collect()
}

fn argmin_from(from: usize, nodes: &[usize], d: &[Vec<f64>]) -> usize {
    let mut best_u = nodes[0];
    let mut best = f64::INFINITY;
    for &u in nodes {
        if d[from][u] < best {
            best = d[from][u];
            best_u = u;
        }
    }
    best_u
}

fn argmin_to_excluding(
    nodes: &[usize],
    to: usize,
    exclude: usize,
    do_exclude: bool,
    d: &[Vec<f64>],
) -> usize {
    let mut best_u = nodes[0];
    let mut best = f64::INFINITY;
    for &u in nodes {
        if do_exclude && u == exclude {
            continue;
        }
        if d[u][to] < best {
            best = d[u][to];
            best_u = u;
        }
    }
    best_u
}

fn pairs_for_cluster(
    instance: &OctspInstance,
    i: usize,
    k: usize,
    greedy: (usize, usize),
) -> Vec<(usize, usize)> {
    let valid = exhaustive_pairs(&instance.clusters[i]);
    if valid.is_empty() {
        return valid;
    }
    if k >= valid.len() {
        return valid;
    }

    let mut selected: Vec<(usize, usize)> = Vec::new();
    if valid.contains(&greedy) {
        selected.push(greedy);
    }
    if k == 0 {
        return selected;
    }

    let scored: Vec<(usize, usize, f64)> = valid
        .iter()
        .copied()
        .filter(|p| *p != greedy)
        .map(|(s, t)| (s, t, geometry_score(instance, i, s, t)))
        .collect();

    let mut remaining = scored;
    while selected.len() < k && !remaining.is_empty() {
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (idx, &(s, t, score)) in remaining.iter().enumerate() {
            let div = diversity((s, t), &selected, &instance.distances);
            let val = score + div;
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        let (s, t, _) = remaining.swap_remove(best_idx);
        selected.push((s, t));
    }
    selected
}

fn geometry_score(instance: &OctspInstance, i: usize, s: usize, t: usize) -> f64 {
    let cluster = &instance.clusters[i];
    let deg = mst_degrees(&instance.distances, cluster);
    let lin = linearity(s, t, cluster, &deg);
    let bridge = bridge_cost(instance, i, s, t);
    LINEARITY_WEIGHT * lin - bridge
}

fn linearity(s: usize, t: usize, cluster: &[usize], deg: &[usize]) -> f64 {
    let ds = deg[cluster.iter().position(|&x| x == s).unwrap_or(0)].max(1);
    let dt = deg[cluster.iter().position(|&x| x == t).unwrap_or(0)].max(1);
    1.0 / ds as f64 + 1.0 / dt as f64
}

fn bridge_cost(instance: &OctspInstance, i: usize, s: usize, t: usize) -> f64 {
    let d = &instance.distances;
    let prev: &[usize] = if i == 0 {
        // Dummy previous "cluster" is the source depot.
        std::slice::from_ref(&instance.source)
    } else {
        &instance.clusters[i - 1]
    };
    let next: &[usize] = if i + 1 == instance.clusters.len() {
        std::slice::from_ref(&instance.target)
    } else {
        &instance.clusters[i + 1]
    };
    let mut in_b = f64::INFINITY;
    for &u in prev {
        in_b = in_b.min(d[u][s]);
    }
    let mut out_b = f64::INFINITY;
    for &v in next {
        out_b = out_b.min(d[t][v]);
    }
    in_b + out_b
}

/// Undirected MST on `min(d(u,v), d(v,u))`; return degree of each cluster node.
fn mst_degrees(dist: &[Vec<f64>], nodes: &[usize]) -> Vec<usize> {
    let n = nodes.len();
    let mut deg = vec![0usize; n];
    if n <= 1 {
        return deg;
    }
    let mut in_tree = vec![false; n];
    let mut min_e = vec![f64::INFINITY; n];
    let mut parent = vec![None; n];
    min_e[0] = 0.0;
    for _ in 0..n {
        let mut u = 0usize;
        let mut best = f64::INFINITY;
        let mut found = false;
        for i in 0..n {
            if !in_tree[i] && min_e[i] < best {
                best = min_e[i];
                u = i;
                found = true;
            }
        }
        if !found {
            break;
        }
        in_tree[u] = true;
        if let Some(p) = parent[u] {
            deg[u] += 1;
            deg[p] += 1;
        }
        for v in 0..n {
            if in_tree[v] {
                continue;
            }
            let w = dist[nodes[u]][nodes[v]].min(dist[nodes[v]][nodes[u]]);
            if w < min_e[v] {
                min_e[v] = w;
                parent[v] = Some(u);
            }
        }
    }
    deg
}

/// Diversity bonus: min over already-chosen pairs of the sum of endpoint distances.
fn diversity(pair: (usize, usize), selected: &[(usize, usize)], dist: &[Vec<f64>]) -> f64 {
    if selected.is_empty() {
        return 0.0;
    }
    let (s, t) = pair;
    let mut best = f64::INFINITY;
    for &(s2, t2) in selected {
        let ds = dist[s][s2].min(dist[s2][s]);
        let dt = dist[t][t2].min(dist[t2][t]);
        best = best.min(ds + dt);
    }
    best
}

fn assemble_from_endpoints(instance: &OctspInstance, endpoints: &[(usize, usize)]) -> Solution {
    if instance.clusters.is_empty() {
        let cost = instance.distances[instance.source][instance.target];
        return Solution {
            tour: vec![instance.source, instance.target],
            cost,
            endpoints: Vec::new(),
        };
    }
    let mut tour = vec![instance.source];
    let mut cost = 0.0;
    let mut last = instance.source;
    for (i, &(s, t)) in endpoints.iter().enumerate() {
        cost += instance.distances[last][s];
        let (intra, path) = held_karp_path(&instance.distances, &instance.clusters[i], s, t);
        cost += intra;
        tour.extend(path);
        last = t;
    }
    cost += instance.distances[last][instance.target];
    tour.push(instance.target);
    Solution {
        tour,
        cost,
        endpoints: endpoints.to_vec(),
    }
}

/// Layered DAG shortest S-T path by a single forward DP pass (Algorithm 1, line 8).
/// Red edges: `τ̂(s,t)` intra-cluster Held-Karp. Blue edges: `d(t, s')` bridges.
fn solve_with_pairs(instance: &OctspInstance, pairs: &[Vec<(usize, usize)>]) -> Solution {
    let m = instance.clusters.len();
    if m == 0 {
        return assemble_from_endpoints(instance, &[]);
    }
    let d = &instance.distances;

    let mut tau: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut paths: Vec<Vec<Vec<usize>>> = Vec::with_capacity(m);
    for i in 0..m {
        let mut ts = Vec::with_capacity(pairs[i].len());
        let mut ps = Vec::with_capacity(pairs[i].len());
        for &(s, t) in &pairs[i] {
            let (c, p) = held_karp_path(d, &instance.clusters[i], s, t);
            ts.push(c);
            ps.push(p);
        }
        tau.push(ts);
        paths.push(ps);
    }

    let mut dp: Vec<Vec<f64>> = (0..m)
        .map(|i| vec![f64::INFINITY; pairs[i].len()])
        .collect();
    let mut pred: Vec<Vec<Option<usize>>> = (0..m).map(|i| vec![None; pairs[i].len()]).collect();

    for k in 0..pairs[0].len() {
        let (s, _) = pairs[0][k];
        dp[0][k] = d[instance.source][s] + tau[0][k];
    }
    for i in 1..m {
        for k in 0..pairs[i].len() {
            let (s, _) = pairs[i][k];
            for j in 0..pairs[i - 1].len() {
                let (_, tprev) = pairs[i - 1][j];
                let cand = dp[i - 1][j] + d[tprev][s] + tau[i][k];
                if cand < dp[i][k] {
                    dp[i][k] = cand;
                    pred[i][k] = Some(j);
                }
            }
        }
    }

    let mut best = f64::INFINITY;
    let mut best_k = 0usize;
    for k in 0..pairs[m - 1].len() {
        let (_, t) = pairs[m - 1][k];
        let cand = dp[m - 1][k] + d[t][instance.target];
        if cand < best {
            best = cand;
            best_k = k;
        }
    }

    let mut chosen = vec![0usize; m];
    chosen[m - 1] = best_k;
    for i in (1..m).rev() {
        chosen[i - 1] = pred[i][chosen[i]].unwrap_or(0);
    }

    let mut tour = vec![instance.source];
    let mut endpoints = Vec::with_capacity(m);
    for i in 0..m {
        let k = chosen[i];
        endpoints.push(pairs[i][k]);
        tour.extend_from_slice(&paths[i][k]);
    }
    tour.push(instance.target);
    Solution {
        tour,
        cost: best,
        endpoints,
    }
}
