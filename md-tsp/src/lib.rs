//! A (1 + 1/√2)-approximation for the metric Multiple-Depot TSP
//! (Zhao, Liu, Xiao, arXiv:2609.19537).
//!
//! Algorithm 1: two-speed primal-dual rooted spanning forest (Section 3.1),
//! a minimum-cost perfect matching on `Odd(F)`, then Eulerian shortcutting.
//! The balanced growth parameter is λ = η = √2 − 1 (Lemma 5, Theorem 1).

use std::cmp::Ordering;
use std::collections::HashSet;

/// Balanced two-speed coefficient λ = η = √2 − 1 (Lemma 5).
pub const LAMBDA: f64 = std::f64::consts::SQRT_2 - 1.0;

/// Same as [`LAMBDA`]; the paper balances both matching-bound coefficients at √2 − 1.
pub const TAU: f64 = LAMBDA;

/// Approximation ratio 1 + 1/√2 of Theorem 1.
pub fn approximation_ratio() -> f64 {
    1.0 + 1.0 / std::f64::consts::SQRT_2
}

/// η = (1 − λ)/(1 + λ) from Section 3.1.
pub fn eta(lambda: f64) -> f64 {
    (1.0 - lambda) / (1.0 + lambda)
}

/// Eq. (1): θ_ℓ(uv) = (c(uv) − min{ℓ(u), ℓ(v)})⁺.
pub fn theta_ell(c_uv: f64, ell_u: f64, ell_v: f64) -> f64 {
    (c_uv - ell_u.min(ell_v)).max(0.0)
}

/// Eq. (7): w = c + λ θ_ℓ.
pub fn w_weight(c_uv: f64, lambda: f64, ell_u: f64, ell_v: f64) -> f64 {
    c_uv + lambda * theta_ell(c_uv, ell_u, ell_v)
}

/// Metric MD-TSP instance: complete graph on depots `0..n_depots` and
/// clients `n_depots..n`, with symmetric nonnegative costs.
#[derive(Clone, Debug)]
pub struct Instance {
    n_depots: usize,
    cost: Vec<Vec<f64>>,
}

impl Instance {
    /// Build from a square symmetric cost matrix. Vertices `0..n_depots` are depots.
    pub fn from_cost_matrix(n_depots: usize, cost: Vec<Vec<f64>>) -> Self {
        let n = cost.len();
        assert!(n_depots >= 1, "MD-TSP requires at least one depot");
        assert!(n_depots <= n, "n_depots exceeds vertex count");
        for row in &cost {
            assert_eq!(row.len(), n, "cost matrix must be square");
        }
        for i in 0..n {
            assert!(cost[i][i].abs() <= 1e-12, "c(x,x) = 0");
            for j in 0..n {
                assert!(cost[i][j] >= -1e-12, "costs must be nonnegative");
                assert!(
                    (cost[i][j] - cost[j][i]).abs() <= 1e-9 * (1.0 + cost[i][j].abs()),
                    "costs must be symmetric"
                );
            }
        }
        Self { n_depots, cost }
    }

    /// Euclidean instance from 2-D coordinates.
    pub fn from_euclidean(depots: &[[f64; 2]], clients: &[[f64; 2]]) -> Self {
        assert!(!depots.is_empty(), "MD-TSP requires at least one depot");
        let mut pts: Vec<[f64; 2]> = Vec::with_capacity(depots.len() + clients.len());
        pts.extend_from_slice(depots);
        pts.extend_from_slice(clients);
        let n = pts.len();
        let mut cost = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let dx = pts[i][0] - pts[j][0];
                let dy = pts[i][1] - pts[j][1];
                cost[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }
        Self::from_cost_matrix(depots.len(), cost)
    }

    /// Line metric: cost is absolute difference of positions.
    pub fn from_line(depot_pos: &[f64], client_pos: &[f64]) -> Self {
        assert!(!depot_pos.is_empty(), "MD-TSP requires at least one depot");
        let mut pos = Vec::with_capacity(depot_pos.len() + client_pos.len());
        pos.extend_from_slice(depot_pos);
        pos.extend_from_slice(client_pos);
        let n = pos.len();
        let mut cost = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                cost[i][j] = (pos[i] - pos[j]).abs();
            }
        }
        Self::from_cost_matrix(depot_pos.len(), cost)
    }

    pub fn n(&self) -> usize {
        self.cost.len()
    }

    pub fn n_depots(&self) -> usize {
        self.n_depots
    }

    pub fn n_clients(&self) -> usize {
        self.n() - self.n_depots
    }

    pub fn is_depot(&self, v: usize) -> bool {
        v < self.n_depots
    }

    pub fn cost(&self, u: usize, v: usize) -> f64 {
        self.cost[u][v]
    }

    /// Triangle inequality, symmetry, nonnegativity, and zero diagonal.
    pub fn is_metric(&self, eps: f64) -> bool {
        let n = self.n();
        for i in 0..n {
            if self.cost[i][i].abs() > eps {
                return false;
            }
            for j in 0..n {
                if self.cost[i][j] < -eps {
                    return false;
                }
                if (self.cost[i][j] - self.cost[j][i]).abs() > eps {
                    return false;
                }
                for k in 0..n {
                    if self.cost[i][k] > self.cost[i][j] + self.cost[j][k] + eps {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Minimum-cost rooted spanning forest under the given edge weights
    /// (depots pre-contracted, Kruskal).
    pub fn min_weight_rsf(
        &self,
        weight: impl Fn(usize, usize) -> f64,
    ) -> (Vec<(usize, usize)>, f64) {
        let n = self.n();
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                edges.push((weight(u, v), u, v));
            }
        }
        edges.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        let mut dsu = Dsu::with_depots_contracted(n, self.n_depots);
        let mut chosen = Vec::new();
        let mut total = 0.0;
        for (w, u, v) in edges {
            if dsu.find(u) != dsu.find(v) {
                dsu.union(u, v);
                chosen.push((u, v));
                total += w;
            }
        }
        (chosen, total)
    }

    /// Minimum-cost RSF under the original costs `c` (OPT lower bound).
    pub fn min_cost_rsf(&self) -> (Vec<(usize, usize)>, f64) {
        self.min_weight_rsf(|u, v| self.cost(u, v))
    }
}

/// One merge of the event implementation in Section 3.1.
#[derive(Clone, Debug)]
pub struct ForestEvent {
    pub u: usize,
    pub v: usize,
    /// Event time s = κ(uv) from Eq. (6).
    pub time: f64,
    /// Clients that received label `time` (empty on a rootless–rootless merge).
    pub newly_labeled: Vec<usize>,
}

/// Rooted spanning forest F together with vertex labels ℓ (Section 3.1).
#[derive(Clone, Debug)]
pub struct RootedSpanningForest {
    pub events: Vec<ForestEvent>,
    /// ℓ(v); depots have 0, clients receive their connection time.
    pub labels: Vec<f64>,
    pub lambda: f64,
}

impl RootedSpanningForest {
    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.events.iter().map(|e| (e.u, e.v)).collect()
    }

    pub fn cost(&self, inst: &Instance) -> f64 {
        self.events.iter().map(|e| inst.cost(e.u, e.v)).sum()
    }

    pub fn w_cost(&self, inst: &Instance) -> f64 {
        self.events
            .iter()
            .map(|e| {
                w_weight(
                    inst.cost(e.u, e.v),
                    self.lambda,
                    self.labels[e.u],
                    self.labels[e.v],
                )
            })
            .sum()
    }
}

/// Feasible MD-TSP solution: one closed tour per depot (trivial tours allowed).
#[derive(Clone, Debug)]
pub struct MdTspSolution {
    /// `tours[u]` starts and ends at depot `u` (or is `[u]` if unused).
    pub tours: Vec<Vec<usize>>,
    pub forest: RootedSpanningForest,
    pub matching: Vec<(usize, usize)>,
}

impl MdTspSolution {
    pub fn cost(&self, inst: &Instance) -> f64 {
        self.tours.iter().map(|t| walk_cost(inst, t)).sum()
    }

    /// Clients that appear in some tour.
    pub fn covered_clients(&self, inst: &Instance) -> HashSet<usize> {
        let mut s = HashSet::new();
        for tour in &self.tours {
            for &v in tour {
                if !inst.is_depot(v) {
                    s.insert(v);
                }
            }
        }
        s
    }
}

fn walk_cost(inst: &Instance, walk: &[usize]) -> f64 {
    if walk.len() < 2 {
        return 0.0;
    }
    walk.windows(2).map(|w| inst.cost(w[0], w[1])).sum()
}

struct Dsu {
    parent: Vec<usize>,
    members: Vec<Vec<usize>>,
    rooted: Vec<bool>,
}

impl Dsu {
    fn with_depots_contracted(n: usize, n_depots: usize) -> Self {
        let parent: Vec<usize> = (0..n).collect();
        let members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut rooted = vec![false; n];
        for u in 0..n_depots {
            rooted[u] = true;
        }
        let mut dsu = Self {
            parent,
            members,
            rooted,
        };
        for u in 1..n_depots {
            dsu.union(0, u);
        }
        dsu
    }

    fn find(&mut self, x: usize) -> usize {
        let mut x = x;
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.members[ra].len() < self.members[rb].len() {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        let extra = std::mem::take(&mut self.members[rb]);
        self.members[ra].extend(extra);
        self.rooted[ra] = self.rooted[ra] || self.rooted[rb];
    }
}

/// Current key κ(uv) from Eq. (6).
fn key(inst: &Instance, lambda: f64, labels: &[f64], dsu: &mut Dsu, u: usize, v: usize) -> f64 {
    let fu = dsu.find(u);
    let fv = dsu.find(v);
    let ru = dsu.rooted[fu];
    let rv = dsu.rooted[fv];
    let c = inst.cost(u, v);
    match (ru, rv) {
        (false, false) => c,
        (true, false) => (1.0 + lambda) * c - lambda * labels[u],
        (false, true) => (1.0 + lambda) * c - lambda * labels[v],
        (true, true) => f64::INFINITY,
    }
}

/// Two-speed primal-dual RSF of Section 3.1 (event implementation).
pub fn primal_dual_rsf(inst: &Instance, lambda: f64) -> RootedSpanningForest {
    assert!((0.0..=1.0).contains(&lambda), "λ must lie in [0, 1]");
    let n = inst.n();
    let n_depots = inst.n_depots();
    let mut dsu = Dsu::with_depots_contracted(n, n_depots);
    let mut labels = vec![f64::NAN; n];
    for u in 0..n_depots {
        labels[u] = 0.0;
    }
    let mut events = Vec::new();
    let n_clients = inst.n_clients();
    for _ in 0..n_clients {
        let mut best_key = f64::INFINITY;
        let mut best = (0, 0);
        for u in 0..n {
            for v in (u + 1)..n {
                if dsu.find(u) == dsu.find(v) {
                    continue;
                }
                let k = key(inst, lambda, &labels, &mut dsu, u, v);
                let better = match k.partial_cmp(&best_key) {
                    Some(Ordering::Less) => true,
                    Some(Ordering::Greater) => false,
                    _ => (u, v) < best,
                };
                if better {
                    best_key = k;
                    best = (u, v);
                }
            }
        }
        assert!(
            best_key.is_finite(),
            "no candidate edge: instance is disconnected under current keys"
        );
        let (u, v) = best;
        let s = best_key;
        let fu = dsu.find(u);
        let fv = dsu.find(v);
        let ru = dsu.rooted[fu];
        let rv = dsu.rooted[fv];
        let joining_root = ru ^ rv;
        let unrooted_root = if joining_root {
            if ru {
                fv
            } else {
                fu
            }
        } else {
            0
        };
        let mut newly_labeled = Vec::new();
        if joining_root {
            newly_labeled = dsu.members[unrooted_root].clone();
            for &x in &newly_labeled {
                labels[x] = s;
            }
        }
        dsu.union(u, v);
        events.push(ForestEvent {
            u,
            v,
            time: s,
            newly_labeled,
        });
    }
    for v in n_depots..n {
        assert!(labels[v].is_finite(), "client {v} never received a label");
    }
    RootedSpanningForest {
        events,
        labels,
        lambda,
    }
}

/// Odd-degree vertices of an undirected edge list.
pub fn odd_vertices(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut deg = vec![0usize; n];
    for &(u, v) in edges {
        deg[u] += 1;
        deg[v] += 1;
    }
    (0..n).filter(|&v| deg[v] % 2 == 1).collect()
}

const MATCHING_DP_LIMIT: usize = 20;

/// Minimum-cost perfect matching on `verts` under `c`.
///
/// Exact subset DP when `|verts| ≤ 20`; greedy thereafter (see README).
pub fn min_cost_perfect_matching(inst: &Instance, verts: &[usize]) -> Vec<(usize, usize)> {
    let m = verts.len();
    assert!(m % 2 == 0, "Odd(F) must have even cardinality");
    if m == 0 {
        return Vec::new();
    }
    if m > MATCHING_DP_LIMIT {
        return greedy_matching(inst, verts);
    }
    let nmask = 1usize << m;
    let mut dp = vec![f64::INFINITY; nmask];
    let mut choice = vec![(0u8, 0u8); nmask];
    dp[0] = 0.0;
    for mask in 0..nmask {
        if dp[mask].is_infinite() {
            continue;
        }
        let mut i = 0;
        while i < m && (mask & (1 << i)) != 0 {
            i += 1;
        }
        if i >= m {
            continue;
        }
        for j in (i + 1)..m {
            if mask & (1 << j) != 0 {
                continue;
            }
            let new_mask = mask | (1 << i) | (1 << j);
            let c = dp[mask] + inst.cost(verts[i], verts[j]);
            if c < dp[new_mask] {
                dp[new_mask] = c;
                choice[new_mask] = (i as u8, j as u8);
            }
        }
    }
    let mut mask = nmask - 1;
    assert!(dp[mask].is_finite(), "no perfect matching");
    let mut matching = Vec::new();
    while mask != 0 {
        let (i, j) = choice[mask];
        matching.push((verts[i as usize], verts[j as usize]));
        mask &= !((1 << i) | (1 << j));
    }
    matching
}

fn greedy_matching(inst: &Instance, verts: &[usize]) -> Vec<(usize, usize)> {
    let m = verts.len();
    let mut pairs = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            pairs.push((inst.cost(verts[i], verts[j]), i, j));
        }
    }
    pairs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut used = vec![false; m];
    let mut matching = Vec::new();
    for (_, i, j) in pairs {
        if !used[i] && !used[j] {
            used[i] = true;
            used[j] = true;
            matching.push((verts[i], verts[j]));
        }
    }
    matching
}

fn connected_components(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    let mut seen = vec![false; n];
    let mut comps = Vec::new();
    for i in 0..n {
        if seen[i] {
            continue;
        }
        let mut stack = vec![i];
        seen[i] = true;
        let mut comp = Vec::new();
        while let Some(v) = stack.pop() {
            comp.push(v);
            for &w in &adj[v] {
                if !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }
        comps.push(comp);
    }
    comps
}

fn hierholzer(n: usize, edges: &[(usize, usize)], start: usize) -> Vec<usize> {
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (id, &(u, v)) in edges.iter().enumerate() {
        adj[u].push((v, id));
        adj[v].push((u, id));
    }
    let mut used = vec![false; edges.len()];
    let mut stack = vec![start];
    let mut circuit = Vec::new();
    while let Some(&v) = stack.last() {
        if let Some(pos) = adj[v].iter().position(|&(_, eid)| !used[eid]) {
            let (to, eid) = adj[v][pos];
            used[eid] = true;
            stack.push(to);
        } else {
            circuit.push(stack.pop().unwrap());
        }
    }
    circuit.reverse();
    circuit
}

/// Shortcut repeated clients and intermediate depots (Algorithm 1, line 6).
fn shortcut(circuit: &[usize], start_depot: usize, n_depots: usize) -> Vec<usize> {
    let mut seen_clients = HashSet::new();
    let mut tour = vec![start_depot];
    for &v in circuit.iter().skip(1) {
        if v == start_depot {
            continue;
        }
        if v < n_depots {
            continue;
        }
        if !seen_clients.insert(v) {
            continue;
        }
        tour.push(v);
    }
    tour.push(start_depot);
    tour
}

/// Algorithm 1 with a given λ ∈ [0, 1].
pub fn approximate_with_lambda(inst: &Instance, lambda: f64) -> MdTspSolution {
    let forest = primal_dual_rsf(inst, lambda);
    let f_edges = forest.edges();
    let odd = odd_vertices(inst.n(), &f_edges);
    let matching = min_cost_perfect_matching(inst, &odd);
    let mut multi = f_edges;
    multi.extend_from_slice(&matching);
    let comps = connected_components(inst.n(), &multi);
    let mut tours = vec![Vec::new(); inst.n_depots()];
    let mut used_depot = vec![false; inst.n_depots()];
    for comp in comps {
        if comp.len() <= 1 {
            continue;
        }
        let depot = comp
            .iter()
            .copied()
            .filter(|&v| inst.is_depot(v))
            .min()
            .expect("nontrivial component of F ⊎ M contains a depot");
        let in_comp: HashSet<usize> = comp.iter().copied().collect();
        let sub: Vec<(usize, usize)> = multi
            .iter()
            .copied()
            .filter(|(a, b)| in_comp.contains(a) && in_comp.contains(b))
            .collect();
        let circuit = hierholzer(inst.n(), &sub, depot);
        let tour = shortcut(&circuit, depot, inst.n_depots());
        tours[depot] = tour;
        used_depot[depot] = true;
    }
    for u in 0..inst.n_depots() {
        if !used_depot[u] {
            tours[u] = vec![u];
        }
    }
    MdTspSolution {
        tours,
        forest,
        matching,
    }
}

/// Algorithm 1 with λ = √2 − 1 (Theorem 1).
pub fn approximate(inst: &Instance) -> MdTspSolution {
    approximate_with_lambda(inst, LAMBDA)
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn eta_of_lambda_half_is_one_third() {
        assert!((eta(0.5) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn eta_equals_lambda_at_sqrt2_minus_1() {
        assert!((eta(LAMBDA) - LAMBDA).abs() < 1e-12);
        assert!((TAU - LAMBDA).abs() < 1e-15);
    }
}
