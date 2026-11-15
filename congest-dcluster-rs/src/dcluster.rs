//! Algorithm 1, DCluster, Section 4.1.
//!
//! DCluster outputs an edge set `F` contained in `E(G)` and the partition `C`
//! of `V(G)` induced by the connected components of `(V(G), F)`. The partition
//! comes sorted into `imax + 1` collections `C^(0), ..., C^(imax)` with
//! `imax = ceil(log2 n) - 1` and `Delta_i = 2^i`, where `C^(i)` holds the
//! clusters centered at the vertices sampled in iteration `i`.
//!
//! Lemma 4.1 states the four output properties, with probability
//! `1 - O(log n / n^2)`:
//!
//! 1. the collections `C^(i)` are mutually disjoint,
//! 2. `deg_G(v) <= 2 Delta_i` for every `v` in a cluster of `C^(i)`,
//! 3. the subgraph of `(V(G), F)` induced by any cluster has diameter at most
//!    `2 ceil(log2 n) - 2`,
//! 4. `|C^(i)| <= 12 n ln n / Delta_i`.
//!
//! Lemma 4.2 bounds the message complexity by `48 n ln n ceil(log2 n)` with the
//! same probability. See [`message_bound`].
//!
//! # How the pseudocode maps onto rounds
//!
//! The paper states Algorithm 1 centrally and then says which lines need
//! communication: lines 7 to 9 and lines 11 to 14. Iteration `i` runs in four
//! rounds, matching the paragraph after the pseudocode.
//!
//! * round 1, lines 5 and 7: each vertex sampled into `W_i` announces "I'm a
//!   center" to its neighbors in `G_i`,
//! * round 2, lines 8, 9 and 11: a vertex that heard a center replies to it and
//!   joins; a vertex that heard none probes the neighbors it sampled from
//!   `N_G(v)`,
//! * round 3, line 12: every probed vertex replies with whether it has left
//!   `G_i`, and with its cluster center if it has one,
//! * round 4, lines 13 and 14: the prober tells the vertex it picked that the
//!   edge is now in `F`, so both endpoints of every edge of `F` know about it.
//!
//! So the round count is exactly `4 (imax + 1) = 4 ceil(log2 n)`, independent of
//! the input graph.
//!
//! Two details the pseudocode leaves open are pinned down here. "Choose any"
//! on lines 8 and 13 takes the smallest identifier, which keeps a run
//! reproducible from its seed. And the probe reply carries the responder's
//! cluster center along with its removal bit, so that a vertex that attaches
//! itself on line 14 knows which cluster it joined without another round; the
//! center is one identifier, so the message stays inside the bandwidth limit.

use crate::congest::{Network, Payload, Tag};
use crate::graph::Graph;
use crate::rng::SplitMix64;

/// `ceil(log2 n)`, with `ceil(log2 1) = 0`.
pub fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

/// `imax = ceil(log2 n) - 1`, the first iteration index of Algorithm 1.
pub fn i_max(n: usize) -> usize {
    ceil_log2(n).saturating_sub(1)
}

/// `Delta_i = 2^i`.
pub fn delta(i: usize) -> u64 {
    1u64 << i
}

/// The sampling probability of line 5 and line 11, `min{1, 6 ln n / Delta_i}`.
pub fn sample_probability(n: usize, i: usize) -> f64 {
    (6.0 * (n as f64).ln() / delta(i) as f64).min(1.0)
}

/// Lemma 4.2, `48 n ln n ceil(log2 n)`.
pub fn message_bound(n: usize) -> f64 {
    48.0 * n as f64 * (n as f64).ln() * ceil_log2(n) as f64
}

/// Lemma 4.1(4), `12 n ln n / Delta_i`.
pub fn cluster_count_bound(n: usize, i: usize) -> f64 {
    12.0 * n as f64 * (n as f64).ln() / delta(i) as f64
}

/// Lemma 4.1(3), `2 ceil(log2 n) - 2`.
pub fn diameter_bound(n: usize) -> u32 {
    (2 * ceil_log2(n)).saturating_sub(2) as u32
}

/// One cluster of the output partition.
#[derive(Clone, Debug)]
pub struct Cluster {
    /// The vertex of `W_i` this cluster is centered at.
    pub center: u32,
    /// The class `i`, so this cluster belongs to `C^(i)`.
    pub class: usize,
    /// Members, sorted.
    pub members: Vec<u32>,
}

/// What DCluster returns.
#[derive(Clone, Debug)]
pub struct Clustering {
    pub n: usize,
    /// `F`, a subset of `E(G)`, smaller endpoint first.
    pub f: Vec<(u32, u32)>,
    /// The clusters, indexed by cluster id.
    pub clusters: Vec<Cluster>,
    /// `classes[i]` lists the cluster ids in `C^(i)`.
    pub classes: Vec<Vec<usize>>,
    /// Cluster id of each vertex, `None` for a vertex left without one.
    pub cluster_of: Vec<Option<usize>>,
    /// The vertex a vertex attached itself to on line 9 or line 14, if any.
    pub parent: Vec<Option<u32>>,
    /// Vertices removed from `G_i` that never reached a cluster. Lemma 4.1
    /// says this is empty with probability `1 - O(log n / n^2)`; it is not a
    /// guarantee, so the field exists rather than being asserted away.
    pub orphans: Vec<u32>,
    pub rounds: u64,
    pub messages: u64,
}

impl Clustering {
    /// Every vertex sits in exactly one cluster, which is Lemma 4.1(1) read as
    /// a statement about the partition.
    pub fn is_total(&self) -> bool {
        self.orphans.is_empty() && self.cluster_of.iter().all(|c| c.is_some())
    }

    /// Total number of clusters.
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }
}

/// Run DCluster on `graph` with the given seed.
pub fn dcluster(graph: &Graph, seed: u64) -> Clustering {
    let n = graph.n();
    assert!(n >= 2, "DCluster needs at least 2 vertices: 6 ln n is 0 at n = 1");
    let imax = i_max(n);

    let mut master = SplitMix64::seed(seed);
    // Each vertex draws from its own stream, so its coin flips do not depend on
    // the order the simulator walks the vertex set.
    let mut rngs = master.split(n);

    let mut net = Network::new(graph);

    // Per-vertex state. Everything here is local to a vertex.
    let mut removed = vec![false; n]; // v is no longer in G_i
    let mut is_center = vec![false; n];
    let mut center_of: Vec<Option<u32>> = vec![None; n];
    let mut center_class = vec![0usize; n]; // meaningful when is_center
    let mut parent: Vec<Option<u32>> = vec![None; n];
    let mut f: Vec<(u32, u32)> = Vec::new();

    // Line 4: for i <- imax downto 0.
    for i in (0..=imax).rev() {
        let p = sample_probability(n, i);
        let delta_i = delta(i);

        // Line 5: W_i <- sample(V(G_i), min{1, 6 ln n / Delta_i}).
        let mut in_w = vec![false; n];
        for v in 0..n {
            if !removed[v] && rngs[v].bernoulli(p) {
                in_w[v] = true;
                is_center[v] = true;
                center_of[v] = Some(v as u32);
                center_class[v] = i;
            }
        }

        // Round 1, line 7. Lemma 4.2 charges this step |W_i| * 2 Delta_i
        // messages: a center writes to its neighbors in G_i, and G_i holds no
        // vertex of X_{i+1}, so those neighbors have degree at most 2 Delta_i.
        for v in 0..n {
            if in_w[v] {
                for &u in graph.neighbors(v) {
                    if !removed[u as usize] {
                        net.send(v, u as usize, Payload::new(Tag::CenterAnnounce, &[]))
                            .expect("center announcement is within the bandwidth limit");
                    }
                }
            }
        }
        net.advance_round();

        // Line 6: for all v in X_i \ W_i. X_i = {v : deg_G(v) > Delta_i}, and
        // the paragraph after the pseudocode and the proof of Lemma 4.2 both
        // read this as the vertices of X_i still present in G_i; the rest
        // already have a cluster.
        let active: Vec<usize> = (0..n)
            .filter(|&v| graph.degree(v) as u64 > delta_i && !removed[v] && !in_w[v])
            .collect();

        // Round 2, lines 8, 9 and 11.
        let mut probed_someone = vec![false; n];
        for &v in &active {
            // Line 7: is N_G(v) intersect W_i nonempty? Every center in that
            // intersection wrote to v in round 1.
            let chosen = net
                .inbox(v)
                .iter()
                .filter(|msg| msg.payload.tag == Tag::CenterAnnounce)
                .map(|msg| msg.from)
                .min();
            match chosen {
                Some(w) => {
                    // Lines 8 and 9.
                    net.send(v, w as usize, Payload::new(Tag::Join, &[]))
                        .expect("join is within the bandwidth limit");
                    parent[v] = Some(w);
                    center_of[v] = Some(w);
                    f.push(ordered(v as u32, w));
                }
                None => {
                    // Line 11: Y <- sample(N_G(v), min{1, 6 ln n / Delta_i}).
                    // Sampled from N_G(v), not from N_{G_i}(v), which is the
                    // whole point: the probe is looking for neighbors that
                    // already left.
                    for &y in graph.neighbors(v) {
                        if rngs[v].bernoulli(p) {
                            net.send(v, y as usize, Payload::new(Tag::Probe, &[]))
                                .expect("probe is within the bandwidth limit");
                            probed_someone[v] = true;
                        }
                    }
                }
            }
        }
        net.advance_round();

        // Round 3, line 12: a probed vertex answers with its own removal bit
        // and its cluster center.
        for y in 0..n {
            let probers: Vec<u32> = net
                .inbox(y)
                .iter()
                .filter(|msg| msg.payload.tag == Tag::Probe)
                .map(|msg| msg.from)
                .collect();
            if probers.is_empty() {
                continue;
            }
            let gone = u64::from(removed[y]);
            let (has_center, center) = match center_of[y] {
                Some(c) => (1u64, c as u64),
                None => (0u64, 0u64),
            };
            for v in probers {
                net.send(y, v as usize, Payload::new(Tag::ProbeReply, &[gone, has_center, center]))
                    .expect("probe reply is within the bandwidth limit");
            }
        }
        net.advance_round();

        // Round 4, lines 12 to 14: pick a y in Y \ V(G_i) and add {v, y} to F.
        for &v in &active {
            if !probed_someone[v] {
                continue;
            }
            let mut best: Option<(u32, Option<u32>)> = None;
            for msg in net.inbox(v) {
                if msg.payload.tag != Tag::ProbeReply || msg.payload.word(0) != 1 {
                    continue;
                }
                let center = if msg.payload.word(1) == 1 { Some(msg.payload.word(2) as u32) } else { None };
                if best.map_or(true, |(y, _)| msg.from < y) {
                    best = Some((msg.from, center));
                }
            }
            if let Some((y, center)) = best {
                net.send(v, y as usize, Payload::new(Tag::Link, &[]))
                    .expect("link is within the bandwidth limit");
                parent[v] = Some(y);
                // Line 14 attaches v to y's cluster. y left G_i in an earlier
                // iteration, so with high probability it has one; if the
                // sampling failed for y then v inherits no cluster either and
                // both show up in `orphans`.
                center_of[v] = center;
                f.push(ordered(v as u32, y));
            }
        }
        net.advance_round();

        // Line 16: G_{i-1} <- G_i \ (X_i union W_i). Every vertex of X_i leaves
        // whether or not it found a cluster, which is where Lemma 4.1(1) can
        // fail.
        for v in 0..n {
            if !removed[v] && (graph.degree(v) as u64 > delta_i || in_w[v]) {
                removed[v] = true;
            }
        }
    }

    f.sort_unstable();
    f.dedup();

    // Lines 17 and 18, plus the observation in the proof of Lemma 4.1(3) that
    // each cluster holds exactly one vertex of W. This part reads the vertices'
    // local state from outside the protocol; it sends nothing and is not
    // charged.
    let mut cluster_index: Vec<Option<usize>> = vec![None; n];
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut classes: Vec<Vec<usize>> = vec![Vec::new(); imax + 1];
    for v in 0..n {
        if is_center[v] {
            cluster_index[v] = Some(clusters.len());
            classes[center_class[v]].push(clusters.len());
            clusters.push(Cluster { center: v as u32, class: center_class[v], members: Vec::new() });
        }
    }

    let mut cluster_of: Vec<Option<usize>> = vec![None; n];
    let mut orphans = Vec::new();
    for v in 0..n {
        match center_of[v].and_then(|c| cluster_index[c as usize]) {
            Some(idx) => {
                cluster_of[v] = Some(idx);
                clusters[idx].members.push(v as u32);
            }
            None => orphans.push(v as u32),
        }
    }
    for c in clusters.iter_mut() {
        c.members.sort_unstable();
    }

    Clustering {
        n,
        f,
        clusters,
        classes,
        cluster_of,
        parent,
        orphans,
        rounds: net.rounds(),
        messages: net.messages(),
    }
}

fn ordered(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph;

    #[test]
    fn log_helpers_match_the_paper() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(1024), 10);
        assert_eq!(ceil_log2(1025), 11);
        assert_eq!(i_max(1024), 9);
        assert_eq!(delta(0), 1);
        assert_eq!(delta(9), 512);
        assert_eq!(diameter_bound(1024), 18);
    }

    #[test]
    fn probability_saturates_on_the_small_classes() {
        // 6 ln 64 is about 24.9, so every class with Delta_i <= 16 samples
        // every remaining vertex.
        assert_eq!(sample_probability(64, 4), 1.0);
        assert!(sample_probability(64, 5) < 1.0);
    }

    #[test]
    fn same_seed_gives_the_same_run() {
        let g = graph::erdos_renyi(64, 0.1, 11);
        let a = dcluster(&g, 777);
        let b = dcluster(&g, 777);
        assert_eq!(a.f, b.f);
        assert_eq!(a.messages, b.messages);
        assert_eq!(a.rounds, b.rounds);
        assert_eq!(a.cluster_of, b.cluster_of);
    }

    #[test]
    fn round_count_is_four_per_iteration() {
        // Hand count on a 4-cycle. n = 4, so ceil(log2 4) = 2 and imax = 1.
        // Algorithm 1 runs iterations i = 1 and i = 0, four rounds each, so 8.
        let g = graph::cycle(4);
        let c = dcluster(&g, 3);
        assert_eq!(c.rounds, 8);

        // And the general form, 4 ceil(log2 n).
        for &n in &[2usize, 5, 16, 17, 100] {
            let g = graph::cycle(n);
            assert_eq!(dcluster(&g, 3).rounds, 4 * ceil_log2(n) as u64, "n = {n}");
        }
    }

    #[test]
    fn a_four_cycle_clusters_by_hand() {
        // Every vertex of C_4 has degree 2. 6 ln 4 is about 8.3, so
        // p_1 = min{1, 8.3 / 2} = 1: at i = 1 the whole vertex set is sampled
        // into W_1, X_1 = {v : deg > 2} is empty, and every vertex ends up as
        // the center of its own singleton cluster. Nothing is left for i = 0.
        let g = graph::cycle(4);
        let c = dcluster(&g, 12345);
        assert!(c.f.is_empty());
        assert_eq!(c.cluster_count(), 4);
        assert_eq!(c.classes[1].len(), 4);
        assert!(c.classes[0].is_empty());
        assert!(c.is_total());
        // Four announcements per center along both incident edges.
        assert_eq!(c.messages, 8);
    }

    #[test]
    fn f_is_a_forest_of_in_trees() {
        let g = graph::preferential_attachment(128, 3, 8);
        let c = dcluster(&g, 21);
        // Every vertex adds at most one edge, on line 9 or line 14, so F has
        // one edge per vertex that has a parent and no more.
        let with_parent = c.parent.iter().filter(|p| p.is_some()).count();
        assert_eq!(c.f.len(), with_parent);
        assert!(c.f.len() < c.n);
    }
}
