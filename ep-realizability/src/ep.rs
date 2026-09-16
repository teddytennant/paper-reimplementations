//! Equitable-partition realizability (Algorithms 1 and 2).

use crate::graph::{Error, SimpleGraph};
use crate::graphic::{
    bipartite_havel_hakimi, gale_ryser, havel_hakimi, is_graphic_regular,
};

/// Cell sizes `H = (|C_1|, …, |C_K|)`.
pub type CellSizes = Vec<usize>;

/// `K × K` quotient: `AH[i][j]` is the number of neighbours in `C_j` of any
/// vertex in `C_i` (exact EP).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AhMatrix {
    data: Vec<Vec<usize>>,
}

impl AhMatrix {
    pub fn new(data: Vec<Vec<usize>>) -> Result<Self, Error> {
        let k = data.len();
        for row in &data {
            if row.len() != k {
                return Err(Error::DimensionMismatch);
            }
        }
        Ok(Self { data })
    }

    pub fn zeros(k: usize) -> Self {
        Self {
            data: vec![vec![0; k]; k],
        }
    }

    pub fn k(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, i: usize, j: usize) -> usize {
        self.data[i][j]
    }

    pub fn rows(&self) -> &[Vec<usize>] {
        &self.data
    }

    pub fn into_inner(self) -> Vec<Vec<usize>> {
        self.data
    }
}

impl std::ops::Index<(usize, usize)> for AhMatrix {
    type Output = usize;
    fn index(&self, (i, j): (usize, usize)) -> &usize {
        &self.data[i][j]
    }
}

/// Inter-cell edge counts: `Q_ii = |E(C_i)|`, `Q_ij = |E(C_i, C_j)|` for `i ≠ j`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QMatrix {
    data: Vec<Vec<usize>>,
}

impl QMatrix {
    pub fn new(data: Vec<Vec<usize>>) -> Result<Self, Error> {
        let k = data.len();
        for row in &data {
            if row.len() != k {
                return Err(Error::DimensionMismatch);
            }
        }
        Ok(Self { data })
    }

    pub fn k(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, i: usize, j: usize) -> usize {
        self.data[i][j]
    }

    pub fn rows(&self) -> &[Vec<usize>] {
        &self.data
    }
}

/// `Q_ij = AH_ij · H_i` for `i ≠ j`; `Q_ii = AH_ii · H_i / 2`.
pub fn q_from_ah(ah: &AhMatrix, h: &[usize]) -> Result<QMatrix, Error> {
    let k = ah.k();
    if h.len() != k {
        return Err(Error::DimensionMismatch);
    }
    let mut q = vec![vec![0usize; k]; k];
    for i in 0..k {
        let prod = ah.get(i, i)
            .checked_mul(h[i])
            .ok_or(Error::Infeasible)?;
        if prod % 2 == 1 {
            return Err(Error::NotDivisible);
        }
        q[i][i] = prod / 2;
        for j in 0..k {
            if i == j {
                continue;
            }
            q[i][j] = ah.get(i, j)
                .checked_mul(h[i])
                .ok_or(Error::Infeasible)?;
        }
    }
    for i in 0..k {
        for j in 0..k {
            if i != j && q[i][j] != q[j][i] {
                return Err(Error::Infeasible);
            }
        }
    }
    QMatrix::new(q)
}

/// Inverse of [`q_from_ah`]: entries must divide evenly.
pub fn ah_from_q(q: &QMatrix, h: &[usize]) -> Result<AhMatrix, Error> {
    let k = q.k();
    if h.len() != k {
        return Err(Error::DimensionMismatch);
    }
    let mut ah = vec![vec![0usize; k]; k];
    for i in 0..k {
        if h[i] == 0 {
            if q.get(i, i) != 0 {
                return Err(Error::NotDivisible);
            }
            continue;
        }
        let two_q = q.get(i, i)
            .checked_mul(2)
            .ok_or(Error::Infeasible)?;
        if two_q % h[i] != 0 {
            return Err(Error::NotDivisible);
        }
        ah[i][i] = two_q / h[i];
        for j in 0..k {
            if i == j {
                continue;
            }
            if q.get(i, j) % h[i] != 0 {
                return Err(Error::NotDivisible);
            }
            ah[i][j] = q.get(i, j) / h[i];
        }
    }
    AhMatrix::new(ah)
}

fn bound_diag(h_i: usize) -> usize {
    h_i.saturating_sub(1)
}

/// Exact-EP graphicality: regular intra-cell sequences and Gale–Ryser on every
/// ordered pair of distinct cells, plus the handshaking identities.
pub fn realizable(ah: &AhMatrix, h: &[usize]) -> bool {
    check_realizable(ah, h).is_ok()
}

fn check_realizable(ah: &AhMatrix, h: &[usize]) -> Result<(), Error> {
    let k = ah.k();
    if h.len() != k {
        return Err(Error::DimensionMismatch);
    }
    for i in 0..k {
        let d_ii = ah.get(i, i);
        if d_ii > bound_diag(h[i]) {
            return Err(Error::Infeasible);
        }
        if !is_graphic_regular(h[i], d_ii) {
            return Err(Error::NotGraphic);
        }
        for j in 0..k {
            if i == j {
                continue;
            }
            let aij = ah.get(i, j);
            let aji = ah.get(j, i);
            if aij > h[j] || aji > h[i] {
                return Err(Error::Infeasible);
            }
            if aij.checked_mul(h[i]) != aji.checked_mul(h[j]) {
                return Err(Error::Infeasible);
            }
        }
    }
    for i in 0..k {
        for j in (i + 1)..k {
            let left = vec![ah.get(i, j); h[i]];
            let right = vec![ah.get(j, i); h[j]];
            if !gale_ryser(&left, &right) {
                return Err(Error::NotBipartiteGraphic);
            }
        }
    }
    Ok(())
}

/// Vertices `0..h[0]`, `h[0]..h[0]+h[1]`, … belong to cells `0, 1, …`.
pub fn partition_from_h(h: &[usize]) -> Vec<usize> {
    let n: usize = h.iter().sum();
    let mut p = vec![0usize; n];
    let mut v = 0;
    for (c, &sz) in h.iter().enumerate() {
        for _ in 0..sz {
            p[v] = c;
            v += 1;
        }
    }
    p
}

fn members_from_h(h: &[usize]) -> Vec<Vec<usize>> {
    let mut members = vec![Vec::new(); h.len()];
    let mut v = 0;
    for (c, &sz) in h.iter().enumerate() {
        for _ in 0..sz {
            members[c].push(v);
            v += 1;
        }
    }
    members
}

/// Algorithm 1: realize an exact EP `(AH, H)` by heap-based Havel–Hakimi on
/// every diagonal layer and heap-based bipartite Havel–Hakimi on every
/// off-diagonal layer.
pub fn realize(ah: &AhMatrix, h: &[usize]) -> Result<SimpleGraph, Error> {
    check_realizable(ah, h)?;
    let n: usize = h.iter().sum();
    let k = h.len();
    let members = members_from_h(h);
    let mut g = SimpleGraph::empty(n);
    for i in 0..k {
        let deg = vec![ah.get(i, i); h[i]];
        let local = havel_hakimi(&deg)?;
        for (a, b) in local {
            g.add_edge(members[i][a], members[i][b])?;
        }
    }
    for i in 0..k {
        for j in (i + 1)..k {
            let left = vec![ah.get(i, j); h[i]];
            let right = vec![ah.get(j, i); h[j]];
            let local = bipartite_havel_hakimi(&left, &right)?;
            for (a, b) in local {
                g.add_edge(members[i][a], members[j][b])?;
            }
        }
    }
    Ok(g)
}

/// Neighbour-count matrix of `graph` under `partition` (cell id per vertex).
///
/// For an exact EP every vertex of a cell has the same row; otherwise each
/// entry is the rounded mean over the cell (half rounds away from zero via
/// ` (sum + |C|/2) / |C| `).
pub fn ep_of(graph: &SimpleGraph, partition: &[usize]) -> (AhMatrix, CellSizes) {
    let n = graph.n();
    assert_eq!(partition.len(), n, "partition length must equal |V|");
    let k = partition.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut sizes = vec![0usize; k];
    for &c in partition {
        sizes[c] += 1;
    }
    let mut counts = vec![vec![0usize; k]; n];
    for &(u, v) in graph.edges() {
        counts[u][partition[v]] += 1;
        counts[v][partition[u]] += 1;
    }
    let mut ah = vec![vec![0usize; k]; k];
    for c in 0..k {
        if sizes[c] == 0 {
            continue;
        }
        for d in 0..k {
            let mut sum = 0usize;
            for v in 0..n {
                if partition[v] == c {
                    sum += counts[v][d];
                }
            }
            ah[c][d] = (sum + sizes[c] / 2) / sizes[c];
        }
    }
    (AhMatrix::new(ah).expect("square"), sizes)
}

/// True iff every vertex of a cell has the same number of neighbours in every
/// cell (the definition of an equitable partition).
pub fn is_equitable(graph: &SimpleGraph, partition: &[usize]) -> bool {
    let n = graph.n();
    if partition.len() != n {
        return false;
    }
    let k = partition.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut counts = vec![vec![0usize; k]; n];
    for &(u, v) in graph.edges() {
        if partition[u] >= k || partition[v] >= k {
            return false;
        }
        counts[u][partition[v]] += 1;
        counts[v][partition[u]] += 1;
    }
    let mut ref_row: Vec<Option<Vec<usize>>> = vec![None; k];
    for v in 0..n {
        let c = partition[v];
        match &ref_row[c] {
            None => ref_row[c] = Some(counts[v].clone()),
            Some(row) if row != &counts[v] => return false,
            Some(_) => {}
        }
    }
    true
}

/// `|E(g1) ∩ E(g2)| / |E(g1)|`. Vertices are identified by index. Empty `g1`
/// returns `1.0`.
pub fn edge_overlap(g1: &SimpleGraph, g2: &SimpleGraph) -> f64 {
    if g1.m() == 0 {
        return 1.0;
    }
    let set2: std::collections::BTreeSet<(usize, usize)> =
        g2.edges().iter().copied().collect();
    let inter = g1
        .edges()
        .iter()
        .filter(|e| set2.contains(e))
        .count();
    inter as f64 / g1.m() as f64
}

/// Equation (4): layer-wise uniform predictor
/// `Ê(EO) = (1/|E(A)|) ∑_ℓ m^A_ℓ m^T_ℓ / N_ℓ`.
pub fn predicted_edge_overlap(
    original: &SimpleGraph,
    reconstructed: &SimpleGraph,
    partition: &[usize],
) -> f64 {
    let n = original.n();
    if original.m() == 0 {
        return 1.0;
    }
    let k = partition.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut sizes = vec![0usize; k];
    for &c in partition.iter().take(n) {
        sizes[c] += 1;
    }
    let mut m_a = vec![vec![0usize; k]; k];
    let mut m_t = vec![vec![0usize; k]; k];
    for &(u, v) in original.edges() {
        let i = partition[u];
        let j = partition[v];
        if i <= j {
            m_a[i][j] += 1;
        } else {
            m_a[j][i] += 1;
        }
    }
    for &(u, v) in reconstructed.edges() {
        if u >= partition.len() || v >= partition.len() {
            continue;
        }
        let i = partition[u];
        let j = partition[v];
        if i <= j {
            m_t[i][j] += 1;
        } else {
            m_t[j][i] += 1;
        }
    }
    let mut acc = 0.0;
    for i in 0..k {
        for j in i..k {
            let n_ell = if i == j {
                sizes[i].saturating_mul(sizes[i].saturating_sub(1)) / 2
            } else {
                sizes[i].saturating_mul(sizes[j])
            };
            if n_ell == 0 {
                continue;
            }
            acc += (m_a[i][j] as f64) * (m_t[i][j] as f64) / (n_ell as f64);
        }
    }
    acc / (original.m() as f64)
}

/// Implied number of undirected edges of a regular EP profile.
pub fn implied_edges(ah: &AhMatrix, h: &[usize]) -> Result<usize, Error> {
    let k = ah.k();
    if h.len() != k {
        return Err(Error::DimensionMismatch);
    }
    let mut two_m = 0usize;
    for i in 0..k {
        for j in 0..k {
            two_m = two_m
                .checked_add(ah.get(i, j).saturating_mul(h[i]))
                .ok_or(Error::Infeasible)?;
        }
        for j in (i + 1)..k {
            let qij = ah.get(i, j).saturating_mul(h[i]);
            let qji = ah.get(j, i).saturating_mul(h[j]);
            if qij != qji {
                return Err(Error::Infeasible);
            }
        }
    }
    if two_m % 2 == 1 {
        return Err(Error::NotDivisible);
    }
    Ok(two_m / 2)
}

fn ceil_int(x: f64) -> i64 {
    x.ceil() as i64
}

fn floor_int(x: f64) -> i64 {
    x.floor() as i64
}

/// Closest integer in `[lo, hi]` to `x`; ties toward the smaller integer
/// (`ζ_J` in Algorithm 2).
fn zeta(lo: i64, hi: i64, x: f64) -> i64 {
    if lo > hi {
        return lo;
    }
    let f = x.floor();
    let c = f + 1.0;
    let nearest = if x - f <= c - x { f } else { c };
    (nearest as i64).clamp(lo, hi)
}

fn cap(i: usize, j: usize, h: &[usize]) -> usize {
    if i == j {
        bound_diag(h[i])
    } else {
        h[j]
    }
}

/// Algorithm 2 (`Φ_ε`) collapsed to a cell-constant integer quotient `AH`.
///
/// The paper returns an `n × k` node-to-block profile `T`. This wrapper uses
/// the same intervals `L, U`, the same `ζ_J` edge-count selector, and the
/// same nearly-constant `δ` construction, then takes the integer closest to
/// the per-cell mean so that [`realize`] (regular sequences) can run. When
/// `n_edges` does not match the selected profile, a greedy bound-respecting
/// adjustment is applied.
pub fn phi_eps(target_ah: &[Vec<f64>], h: &[usize], n_edges: usize, eps: f64) -> AhMatrix {
    let k = h.len();
    assert_eq!(target_ah.len(), k, "target_ah must be K×K");
    for row in target_ah {
        assert_eq!(row.len(), k, "target_ah must be K×K");
    }

    let mut lo = vec![vec![0i64; k]; k];
    let mut hi = vec![vec![0i64; k]; k];
    let mut intervals_ok = true;
    for i in 0..k {
        for j in 0..k {
            let q = target_ah[i][j];
            let nij = cap(i, j, h) as i64;
            let l = 0i64.max(ceil_int(q - eps));
            let u = nij.min(floor_int(q + eps));
            if l > u {
                intervals_ok = false;
                lo[i][j] = 0;
                hi[i][j] = nij.max(0);
            } else {
                lo[i][j] = l;
                hi[i][j] = u;
            }
        }
    }
    if !intervals_ok {
        for i in 0..k {
            for j in 0..k {
                lo[i][j] = 0;
                hi[i][j] = cap(i, j, h) as i64;
            }
        }
    }

    let mut ah = vec![vec![0usize; k]; k];

    // Diagonal layers (Algorithm 2 steps 3–4).
    for i in 0..k {
        let s = h[i] as i64;
        let l = lo[i][i];
        let u = hi[i][i];
        if s == 0 {
            ah[i][i] = 0;
            continue;
        }
        let j_lo = ceil_int((s as f64) * (l as f64) / 2.0)
            .max(0)
            .min(s * (s - 1) / 2);
        let j_hi = floor_int((s as f64) * (u as f64) / 2.0)
            .max(0)
            .min(s * (s - 1) / 2);
        let target_m = (s as f64) * target_ah[i][i] / 2.0;
        let m = if j_lo <= j_hi {
            zeta(j_lo, j_hi, target_m)
        } else {
            zeta(0, s * (s - 1) / 2, target_m)
        };
        // Collapse δ(s, 2m, L, U) to a regular degree.
        let two_m = (2 * m).max(0) as usize;
        let mut d = if s > 0 {
            zeta(l.max(0), u.max(l), (two_m as f64) / (s as f64))
        } else {
            0
        } as usize;
        d = d.min(cap(i, i, h));
        if s % 2 == 1 && d % 2 == 1 {
            // nd even: odd cell size needs even degree.
            if d > 0 && (d as i64) - 1 >= l {
                d -= 1;
            } else if d + 1 <= cap(i, i, h) && (d as i64) + 1 <= u {
                d += 1;
            } else if d > 0 {
                d -= 1;
            }
        }
        ah[i][i] = d;
    }

    // Off-diagonal layers (Algorithm 2 steps 5–6).
    for i in 0..k {
        for j in (i + 1)..k {
            let sa = h[i] as i64;
            let sb = h[j] as i64;
            let lab = lo[i][j];
            let uab = hi[i][j];
            let lba = lo[j][i];
            let uba = hi[j][i];
            let j_lo = (sa * lab).max(sb * lba);
            let j_hi = (sa * uab).min(sb * uba).min(sa * sb);
            let target_m =
                (sa as f64 * target_ah[i][j] + sb as f64 * target_ah[j][i]) / 2.0;
            let m = if j_lo <= j_hi {
                zeta(j_lo, j_hi, target_m)
            } else {
                zeta(0, sa * sb, target_m)
            };
            let m = m.max(0) as usize;
            // Prefer exact regular pair AH_ij * sa = AH_ji * sb = m.
            let (aij, aji) = regular_bipartite_pair(h[i], h[j], m, cap(i, j, h), cap(j, i, h));
            ah[i][j] = aij;
            ah[j][i] = aji;
        }
    }

    let mut matrix = AhMatrix::new(ah).expect("square");
    adjust_edge_budget(&mut matrix, h, target_ah, n_edges);
    matrix
}

/// Integers `(a, b)` with `a ≤ cap_a`, `b ≤ cap_b`, `a * sa ≈ b * sb ≈ m`.
fn regular_bipartite_pair(
    sa: usize,
    sb: usize,
    m: usize,
    cap_a: usize,
    cap_b: usize,
) -> (usize, usize) {
    if sa == 0 || sb == 0 {
        return (0, 0);
    }
    // Search multiples of lcm near m, clipped to the degree caps.
    let g = gcd(sa, sb);
    let step = sa / g * sb; // lcm
    if step == 0 {
        return (0, 0);
    }
    let max_m = sa
        .saturating_mul(cap_a)
        .min(sb.saturating_mul(cap_b))
        .min(sa.saturating_mul(sb));
    let mut best_m = (m / step) * step;
    if best_m > max_m {
        best_m = (max_m / step) * step;
    }
    let up = best_m.saturating_add(step);
    if up <= max_m && m.abs_diff(up) < m.abs_diff(best_m) {
        best_m = up;
    }
    let a = (best_m / sa).min(cap_a);
    let b = (best_m / sb).min(cap_b);
    // Re-sync if clipping broke the identity.
    if a.saturating_mul(sa) != b.saturating_mul(sb) {
        let mm = a.saturating_mul(sa).min(b.saturating_mul(sb));
        let mm = (mm / step) * step;
        return ((mm / sa).min(cap_a), (mm / sb).min(cap_b));
    }
    (a, b)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn adjust_edge_budget(
    ah: &mut AhMatrix,
    h: &[usize],
    target: &[Vec<f64>],
    n_edges: usize,
) {
    let k = h.len();
    for _ in 0..(k * k * 64 + n_edges + 8) {
        let cur = match implied_edges(ah, h) {
            Ok(m) => m,
            Err(_) => return,
        };
        if cur == n_edges {
            return;
        }
        let want_up = cur < n_edges;
        let mut best: Option<(i64, usize, usize, i32)> = None;
        // Diagonal: try ±1 (or ±2 when cell size is odd).
        for i in 0..k {
            let step = if h[i] % 2 == 1 { 2 } else { 1 };
            let d = ah.get(i, i);
            let cand = if want_up {
                d.saturating_add(step)
            } else {
                d.saturating_sub(step)
            };
            if cand == d || cand > cap(i, i, h) {
                continue;
            }
            if !is_graphic_regular(h[i], cand) {
                continue;
            }
            let err = ((cand as f64) - target[i][i]).abs();
            let score = -((err * 1000.0) as i64);
            let key = (score, i, i, cand as i32 - d as i32);
            if best.map(|b| key > (b.0, b.1, b.2, b.3)).unwrap_or(true) {
                best = Some(key);
            }
        }
        for i in 0..k {
            for j in (i + 1)..k {
                let a = ah.get(i, j);
                let b = ah.get(j, i);
                let g = gcd(h[i], h[j]);
                if g == 0 {
                    continue;
                }
                let da = h[j] / g;
                let db = h[i] / g;
                let (na, nb) = if want_up {
                    (a.saturating_add(da), b.saturating_add(db))
                } else {
                    (a.saturating_sub(da), b.saturating_sub(db))
                };
                if na == a && nb == b {
                    continue;
                }
                if na > cap(i, j, h) || nb > cap(j, i, h) {
                    continue;
                }
                if na.saturating_mul(h[i]) != nb.saturating_mul(h[j]) {
                    continue;
                }
                let err = ((na as f64) - target[i][j]).abs()
                    + ((nb as f64) - target[j][i]).abs();
                let score = -((err * 1000.0) as i64);
                let key = (score, i, j, na as i32 - a as i32);
                if best.map(|b| key > (b.0, b.1, b.2, b.3)).unwrap_or(true) {
                    best = Some(key);
                }
            }
        }
        match best {
            None => return,
            Some((_, i, j, delta)) => {
                let mut data = ah.rows().to_vec();
                if i == j {
                    let step = if h[i] % 2 == 1 { 2 } else { 1 };
                    if delta > 0 {
                        data[i][i] = data[i][i].saturating_add(step);
                    } else {
                        data[i][i] = data[i][i].saturating_sub(step);
                    }
                } else {
                    let g = gcd(h[i], h[j]);
                    let da = h[j] / g;
                    let db = h[i] / g;
                    if delta > 0 {
                        data[i][j] = data[i][j].saturating_add(da);
                        data[j][i] = data[j][i].saturating_add(db);
                    } else {
                        data[i][j] = data[i][j].saturating_sub(da);
                        data[j][i] = data[j][i].saturating_sub(db);
                    }
                }
                *ah = AhMatrix::new(data).expect("square");
            }
        }
    }
}

/// Linear congruential generator for layer-preserving switches.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn gen_range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }
}

/// One layer-preserving 2-swap (paper: switches that preserve `T`).
///
/// Intra-cell: two edges inside the same cell are replaced by another pairing
/// of their four endpoints. Inter-cell: two edges of `E(C_i, C_j)` are
/// crossed. Returns whether a swap was applied.
pub fn layer_preserving_switch(
    graph: &mut SimpleGraph,
    partition: &[usize],
    rng: &mut impl FnMut(usize) -> usize,
) -> bool {
    let n = graph.n();
    if partition.len() != n || graph.m() < 2 {
        return false;
    }
    let k = partition.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    // Bucket edges by unordered cell pair.
    let mut layers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); k * k];
    for &(u, v) in graph.edges() {
        let a = partition[u];
        let b = partition[v];
        let (c, d) = if a <= b { (a, b) } else { (b, a) };
        layers[c * k + d].push((u, v));
    }
    let nonempty: Vec<usize> = layers
        .iter()
        .enumerate()
        .filter(|(_, es)| es.len() >= 2)
        .map(|(i, _)| i)
        .collect();
    if nonempty.is_empty() {
        return false;
    }
    let li = nonempty[rng(nonempty.len())];
    let es = &layers[li];
    let i1 = rng(es.len());
    let mut i2 = rng(es.len().saturating_sub(1));
    if i2 >= i1 {
        i2 += 1;
    }
    if i2 >= es.len() {
        return false;
    }
    let (u, v) = es[i1];
    let (x, y) = es[i2];
    let c = li / k;
    let d = li % k;
    if c == d {
        // Intra-cell: four distinct endpoints, try the other two matchings.
        let mut verts = [u, v, x, y];
        verts.sort_unstable();
        if verts[0] == verts[1] || verts[1] == verts[2] || verts[2] == verts[3] {
            return false;
        }
        let pairs = [((u, x), (v, y)), ((u, y), (v, x))];
        let choice = pairs[rng(2)];
        let (p, q) = choice;
        if p.0 == p.1 || q.0 == q.1 {
            return false;
        }
        if graph.has_edge(p.0, p.1) || graph.has_edge(q.0, q.1) {
            return false;
        }
        graph.remove_edge(u, v);
        graph.remove_edge(x, y);
        let _ = graph.add_edge(p.0, p.1);
        let _ = graph.add_edge(q.0, q.1);
        true
    } else {
        // Bipartite layer: (left, right) × (left, right) → crossed pair.
        let (lu, ru) = if partition[u] == c { (u, v) } else { (v, u) };
        let (lx, rx) = if partition[x] == c { (x, y) } else { (y, x) };
        if lu == lx || ru == rx {
            return false;
        }
        if graph.has_edge(lu, rx) || graph.has_edge(lx, ru) {
            return false;
        }
        graph.remove_edge(u, v);
        graph.remove_edge(x, y);
        let _ = graph.add_edge(lu, rx);
        let _ = graph.add_edge(lx, ru);
        true
    }
}

/// Apply up to `n_swaps` successful layer-preserving 2-swaps.
pub fn randomize_ep(graph: &mut SimpleGraph, partition: &[usize], n_swaps: usize, seed: u64) {
    let mut lcg = Lcg(seed | 1);
    let mut applied = 0usize;
    let mut attempts = 0usize;
    while applied < n_swaps && attempts < n_swaps.saturating_mul(32) + 32 {
        attempts += 1;
        let mut rng = |n: usize| lcg.gen_range(n);
        if layer_preserving_switch(graph, partition, &mut rng) {
            applied += 1;
        }
    }
}

/// Algorithm 1 plus a handful of EP-preserving switches (so two draws can differ).
pub fn realize_sampled(
    ah: &AhMatrix,
    h: &[usize],
    n_swaps: usize,
    seed: u64,
) -> Result<SimpleGraph, Error> {
    let mut g = realize(ah, h)?;
    let p = partition_from_h(h);
    randomize_ep(&mut g, &p, n_swaps, seed);
    Ok(g)
}
