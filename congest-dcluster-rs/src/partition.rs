//! The parts of Sections 3.1 and 4.3 that the paper writes out in full.
//!
//! What is here:
//!
//! * [`measure_alpha_beta`], the two conditions of Definition 3.1, measured on
//!   a partition and an edge set,
//! * [`sample_incident_edges`], step 2 of Algorithm Partition, run on the
//!   simulated network,
//! * [`is_i_moderate`], [`is_admissible`] and [`class_threshold`], the
//!   `i`-moderate test and conditions (C1) and (C2) from Section 4.3.
//!
//! What is not here: steps 3 and 4 of Algorithm Partition. Step 3 is `MPX(n^l)`
//! from Miller, Peng and Xu, step 4 calls `OutDetect(f)` from the outdetect
//! labeling scheme, and step 1 needs the tree splitting of Lemma 4.3 from
//! Izumi, Kitamura, Naruse and Schwartzman. The paper takes all three as black
//! boxes, quoting their guarantees as Theorems 2.3 and 2.1 and Lemma 4.3
//! without pseudocode, so there is nothing to transcribe and this crate does
//! not invent a version.
//!
//! Where Section 4.3 writes `log n` without a base, this module reads it as the
//! natural log, matching the `ln n` in the Chernoff and Markov bounds it sits
//! next to.

use crate::congest::{Network, Payload, Tag};
use crate::dcluster::{delta, Clustering};
use crate::graph::Graph;
use crate::rng::SplitMix64;
use std::collections::VecDeque;

/// The measurement of Definition 3.1 for one pair `(P, F)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlphaBeta {
    /// `alpha = max over P of D_{H_F[P]}`, condition 1. `None` when some part
    /// is disconnected in `(V, F)`, which leaves its diameter undefined.
    pub alpha: Option<u32>,
    /// `beta = sum over open P of (D_{H_F[P]} + 1)`, condition 2. `None` for
    /// the same reason.
    pub beta: Option<u64>,
    pub open_parts: usize,
    pub closed_parts: usize,
}

/// Measure `alpha` and `beta` for the partition `part_of` and the edge set `f`.
///
/// A part `P` is closed with respect to `F` when every edge of `G` with exactly
/// one endpoint in `P` is in `F`, and open otherwise (Section 3.1). Diameters
/// are taken inside `H_F = (V, F)` restricted to `P`, by breadth-first search
/// from each vertex of the part.
pub fn measure_alpha_beta(g: &Graph, part_of: &[usize], part_count: usize, f: &[(u32, u32)]) -> AlphaBeta {
    let n = g.n();
    let mut f_sorted: Vec<(u32, u32)> = f.to_vec();
    f_sorted.sort_unstable();
    f_sorted.dedup();
    let mut f_adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(u, v) in &f_sorted {
        f_adj[u as usize].push(v);
        f_adj[v as usize].push(u);
    }

    let mut members: Vec<Vec<u32>> = vec![Vec::new(); part_count];
    for v in 0..n {
        members[part_of[v]].push(v as u32);
    }

    let mut alpha: Option<u32> = Some(0);
    let mut beta: Option<u64> = Some(0);
    let mut open_parts = 0;
    let mut closed_parts = 0;

    for (idx, part) in members.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        let diameter = induced_diameter(&f_adj, part, part_of, idx);

        // Open or closed with respect to F, from Section 3.1: P is closed when
        // every edge of G leaving P is in F.
        let mut closed = true;
        'outer: for &v in part {
            for &w in g.neighbors(v as usize) {
                if part_of[w as usize] != idx {
                    let e = if v < w { (v, w) } else { (w, v) };
                    if f_sorted.binary_search(&e).is_err() {
                        closed = false;
                        break 'outer;
                    }
                }
            }
        }
        if closed {
            closed_parts += 1;
        } else {
            open_parts += 1;
        }

        match diameter {
            Some(d) => {
                // Condition 1.
                alpha = alpha.map(|a| a.max(d));
                // Condition 2, over the open parts only.
                if !closed {
                    beta = beta.map(|b| b + u64::from(d) + 1);
                }
            }
            None => {
                alpha = None;
                if !closed {
                    beta = None;
                }
            }
        }
    }

    AlphaBeta { alpha, beta, open_parts, closed_parts }
}

/// Diameter of the subgraph of `(V, F)` induced by one part, by breadth-first
/// search from every vertex of the part. `None` if the part is disconnected in
/// `(V, F)`.
pub fn induced_diameter(f_adj: &[Vec<u32>], part: &[u32], part_of: &[usize], idx: usize) -> Option<u32> {
    let n = f_adj.len();
    let mut dist: Vec<i64> = vec![-1; n];
    let mut worst = 0u32;
    for &src in part {
        for &v in part {
            dist[v as usize] = -1;
        }
        dist[src as usize] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(src);
        let mut reached = 1usize;
        while let Some(u) = queue.pop_front() {
            let du = dist[u as usize];
            for &w in &f_adj[u as usize] {
                if part_of[w as usize] == idx && dist[w as usize] < 0 {
                    dist[w as usize] = du + 1;
                    worst = worst.max((du + 1) as u32);
                    reached += 1;
                    queue.push_back(w);
                }
            }
        }
        if reached != part.len() {
            return None;
        }
    }
    Some(worst)
}

/// Read a [`Clustering`] as a partition: `part_of[v]` is `v`'s cluster id, and
/// each leftover vertex becomes a singleton part of its own.
///
/// The leftovers are the failure case of Lemma 4.1(1). Giving them singleton
/// parts keeps `measure_alpha_beta` total instead of making it panic on a run
/// where the sampling missed a vertex.
pub fn partition_from_clustering(c: &Clustering) -> (Vec<usize>, usize) {
    let mut part_of = vec![usize::MAX; c.n];
    let mut next = c.cluster_count();
    for v in 0..c.n {
        part_of[v] = match c.cluster_of[v] {
            Some(idx) => idx,
            None => {
                next += 1;
                next - 1
            }
        };
    }
    (part_of, next)
}

/// Step 2 of Algorithm Partition.
///
/// Each vertex in a cluster of `C^(i)` samples its incident edges independently
/// with probability `q_i = min{1, 3 n^k log n / Delta_i}`, and `J` is the set of
/// edges sampled by at least one endpoint. In the simulation the sampling
/// vertex sends one message along each edge it picks, so both endpoints agree
/// on `J` and the cost is one message per sample.
pub fn sample_incident_edges<'g>(
    net: &mut Network<'g>,
    clustering: &Clustering,
    kappa: f64,
    seed: u64,
) -> Vec<(u32, u32)> {
    let g = net.graph();
    let n = g.n();
    let mut master = SplitMix64::seed(seed);
    let mut rngs = master.split(n);

    for u in 0..n {
        let q = match clustering.cluster_of[u] {
            Some(idx) => edge_sample_probability(n, clustering.clusters[idx].class, kappa),
            None => continue,
        };
        for &v in g.neighbors(u) {
            if rngs[u].bernoulli(q) {
                net.send(u, v as usize, Payload::new(Tag::EdgeSampled, &[]))
                    .expect("edge sample notice is within the bandwidth limit");
            }
        }
    }
    net.advance_round();

    let mut j: Vec<(u32, u32)> = Vec::new();
    for v in 0..n {
        for msg in net.inbox(v) {
            if msg.payload.tag == Tag::EdgeSampled {
                let (a, b) = (msg.from, v as u32);
                j.push(if a < b { (a, b) } else { (b, a) });
            }
        }
    }
    j.sort_unstable();
    j.dedup();
    j
}

/// `q_i = min{1, 3 n^k log n / Delta_i}` from step 2 of Algorithm Partition.
pub fn edge_sample_probability(n: usize, i: usize, kappa: f64) -> f64 {
    let n = n as f64;
    (3.0 * n.powf(kappa) * n.ln() / delta(i) as f64).min(1.0)
}

/// `t = floor(log(n^{k+l} log n))`, so `Delta_i > n^{k+l} log n` exactly when
/// `i > t`. The high-degree clusters `C^H` are those of class `i > t`.
pub fn class_threshold(n: usize, kappa: f64, lambda: f64) -> i64 {
    let x = (n as f64).powf(kappa + lambda) * (n as f64).ln();
    if x <= 0.0 {
        return -1;
    }
    x.log2().floor() as i64
}

/// `X` is `i`-moderate when `X` is contained in `C^(i)` and
/// `|X| <= n^{2k} log n / Delta_i` (Section 4.3).
pub fn is_i_moderate(size: usize, i: usize, n: usize, kappa: f64) -> bool {
    let n = n as f64;
    (size as f64) <= n.powf(2.0 * kappa) * n.ln() / delta(i) as f64
}

/// A part is admissible when it satisfies (C1) and (C2) of Section 4.3.
///
/// * (C1) no boundary cluster of the part is in `C^H`, that is of class
///   `i > t`,
/// * (C2) `P^(i)` is `i`-moderate for every `i`.
///
/// `part_classes` gives the class of each cluster in the part, and
/// `boundary_classes` the classes of the clusters of the part that carry an
/// edge leaving it.
pub fn is_admissible(
    part_classes: &[usize],
    boundary_classes: &[usize],
    n: usize,
    kappa: f64,
    lambda: f64,
) -> bool {
    let t = class_threshold(n, kappa, lambda);
    // (C1).
    if boundary_classes.iter().any(|&i| (i as i64) > t) {
        return false;
    }
    // (C2).
    let mut per_class = std::collections::BTreeMap::new();
    for &i in part_classes {
        *per_class.entry(i).or_insert(0usize) += 1;
    }
    per_class.iter().all(|(&i, &count)| is_i_moderate(count, i, n, kappa))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dcluster::dcluster;
    use crate::graph;

    #[test]
    fn a_path_measured_as_one_part() {
        // P = {V}, F = E(P_5). The single part is closed, since no edge leaves
        // it, and its diameter in (V, F) is 4.
        let g = graph::path(5);
        let f = g.edges();
        let m = measure_alpha_beta(&g, &[0, 0, 0, 0, 0], 1, &f);
        assert_eq!(m.alpha, Some(4));
        assert_eq!(m.beta, Some(0));
        assert_eq!(m.closed_parts, 1);
        assert_eq!(m.open_parts, 0);
    }

    #[test]
    fn open_parts_are_the_ones_missing_a_crossing_edge() {
        // Two parts of P_4, and F holds only the two inside edges. The edge
        // {1, 2} crosses and is not in F, so both parts are open, each with
        // diameter 1, giving beta = (1 + 1) + (1 + 1) = 4.
        let g = graph::path(4);
        let f = vec![(0, 1), (2, 3)];
        let m = measure_alpha_beta(&g, &[0, 0, 1, 1], 2, &f);
        assert_eq!(m.alpha, Some(1));
        assert_eq!(m.beta, Some(4));
        assert_eq!(m.open_parts, 2);

        // Put the crossing edge in F and both parts close, so beta drops to 0.
        let f = vec![(0, 1), (1, 2), (2, 3)];
        let m = measure_alpha_beta(&g, &[0, 0, 1, 1], 2, &f);
        assert_eq!(m.beta, Some(0));
        assert_eq!(m.closed_parts, 2);
    }

    #[test]
    fn a_disconnected_part_has_no_diameter() {
        let g = graph::path(4);
        let m = measure_alpha_beta(&g, &[0, 0, 0, 0], 1, &[(0, 1)]);
        assert_eq!(m.alpha, None);
    }

    #[test]
    fn moderate_and_admissible_follow_the_definitions() {
        let n = 1024;
        let kappa = 1.0 / 3.0;
        let lambda = 1.0 / 6.0;
        // n^{2/3} ln n at n = 1024 is about 704.19, so a class-0 part may
        // hold up to 704 clusters and a class-8 part up to 704.19 / 256, which
        // is 2.
        assert!(is_i_moderate(704, 0, n, kappa));
        assert!(!is_i_moderate(705, 0, n, kappa));
        assert!(is_i_moderate(2, 8, n, kappa));
        assert!(!is_i_moderate(3, 8, n, kappa));

        // t = floor(log2(n^{1/2} ln n)) at n = 1024 is floor(log2(221.8)) = 7.
        assert_eq!(class_threshold(n, kappa, lambda), 7);
        // (C1) rejects a boundary cluster of class 8.
        assert!(!is_admissible(&[0, 0], &[8], n, kappa, lambda));
        assert!(is_admissible(&[0, 0], &[7], n, kappa, lambda));
        // (C2) rejects a part with too many class-8 clusters.
        assert!(!is_admissible(&[8, 8, 8], &[0], n, kappa, lambda));
    }

    #[test]
    fn sampled_edges_are_real_edges_and_deterministic() {
        let g = graph::erdos_renyi(96, 0.1, 4);
        let c = dcluster(&g, 4);
        let all: std::collections::BTreeSet<(u32, u32)> = g.edges().into_iter().collect();

        let mut net = Network::new(&g);
        let j = sample_incident_edges(&mut net, &c, 1.0 / 3.0, 90);
        assert!(j.iter().all(|e| all.contains(e)), "J must be a subset of E(G)");
        assert!(net.messages() >= j.len() as u64);

        let mut net2 = Network::new(&g);
        let j2 = sample_incident_edges(&mut net2, &c, 1.0 / 3.0, 90);
        assert_eq!(j, j2);
        assert_eq!(net.messages(), net2.messages());
    }

    #[test]
    fn every_edge_is_sampled_once_the_probability_saturates() {
        // kappa = 1/2 on a small graph pushes q_i to 1 for every class, so J is
        // all of E(G).
        let g = graph::clique(16);
        let c = dcluster(&g, 6);
        assert!((0..16).all(|v| c.cluster_of[v].is_some()));
        let mut net = Network::new(&g);
        let j = sample_incident_edges(&mut net, &c, 1.0, 5);
        assert_eq!(j, g.edges());
    }
}
