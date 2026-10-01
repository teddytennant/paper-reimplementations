//! Connected dominating set on semi-ladder-free graphs.
//!
//! Implements the compact representation of inclusion-wise minimal set covers
//! (Solve / Algorithm 1), connected set cover via that representation and a
//! group Steiner tree, the grouping core, and Conn-DS by a group Steiner tree
//! in the input graph. Theorem 4's lossy compression is not implemented.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

/// Undirected simple graph on vertices `0..n-1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    adj: Vec<Vec<usize>>,
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self {
            adj: vec![Vec::new(); n],
        }
    }

    pub fn from_edges<I>(n: usize, edges: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let mut g = Self::new(n);
        for (u, v) in edges {
            g.add_edge(u, v);
        }
        g
    }

    pub fn n(&self) -> usize {
        self.adj.len()
    }

    pub fn add_vertex(&mut self) -> usize {
        self.adj.push(Vec::new());
        self.adj.len() - 1
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        if u == v || u >= self.n() || v >= self.n() {
            return;
        }
        insert_sorted(&mut self.adj[u], v);
        insert_sorted(&mut self.adj[v], u);
    }

    pub fn neighbors(&self, v: usize) -> &[usize] {
        &self.adj[v]
    }

    pub fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    pub fn is_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.n() || v >= self.n() {
            return false;
        }
        self.adj[u].binary_search(&v).is_ok()
    }

    pub fn vertices(&self) -> impl Iterator<Item = usize> {
        0..self.n()
    }

    /// Induced subgraph on `verts`, remapped to `0..verts.len()` in the given order.
    pub fn induced(&self, verts: &[usize]) -> (Graph, Vec<usize>) {
        let mut g = Graph::new(verts.len());
        let mut old_to_new = HashMap::new();
        for (i, &v) in verts.iter().enumerate() {
            old_to_new.insert(v, i);
        }
        for (i, &v) in verts.iter().enumerate() {
            if v >= self.n() {
                continue;
            }
            for &u in &self.adj[v] {
                if let Some(&j) = old_to_new.get(&u) {
                    if i < j {
                        g.add_edge(i, j);
                    }
                }
            }
        }
        (g, verts.to_vec())
    }
}

fn insert_sorted(xs: &mut Vec<usize>, v: usize) {
    match xs.binary_search(&v) {
        Ok(_) => {}
        Err(i) => xs.insert(i, v),
    }
}

/// Universe `0..universe` and an indexed family. The index is the identity of a set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetSystem {
    pub universe: usize,
    pub sets: Vec<BTreeSet<usize>>,
}

impl SetSystem {
    pub fn new(universe: usize, sets: Vec<BTreeSet<usize>>) -> Self {
        Self { universe, sets }
    }

    pub fn from_slices(universe: usize, sets: &[Vec<usize>]) -> Self {
        let sets = sets
            .iter()
            .map(|s| s.iter().copied().filter(|&x| x < universe).collect())
            .collect();
        Self { universe, sets }
    }
}

/// A semi-ladder: set indices `S_1..S_d` and elements `x_1..x_d`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness {
    pub set_indices: Vec<usize>,
    pub elements: Vec<usize>,
}

/// An `ℓ`-tuple of subfamilies. Each subfamily is a list of indices into `F`.
pub type Tuple = Vec<Vec<usize>>;

/// `true` when no `d` sets and `d` elements form a semi-ladder.
///
/// `x_i ∈ S_j` whenever `i < j`, and `x_i ∉ S_i`. Indices are 1-based in the
/// definition and 0-based in the witness vectors.
pub fn is_d_semi_ladder_free(system: &SetSystem, d: usize) -> bool {
    semi_ladder_witness(system, d).is_none()
}

pub fn semi_ladder_witness(system: &SetSystem, d: usize) -> Option<Witness> {
    if d == 0 {
        return None;
    }
    let m = system.sets.len();
    let n = system.universe;
    if d > m || d > n {
        return None;
    }
    let mut set_indices = Vec::with_capacity(d);
    let mut elements = Vec::with_capacity(d);
    let mut used_sets = vec![false; m];
    let mut used_elems = vec![false; n];
    if search_ladder(
        system,
        d,
        &mut set_indices,
        &mut elements,
        &mut used_sets,
        &mut used_elems,
    ) {
        Some(Witness {
            set_indices,
            elements,
        })
    } else {
        None
    }
}

fn search_ladder(
    system: &SetSystem,
    d: usize,
    set_indices: &mut Vec<usize>,
    elements: &mut Vec<usize>,
    used_sets: &mut [bool],
    used_elems: &mut [bool],
) -> bool {
    if set_indices.len() == d {
        return true;
    }
    let m = system.sets.len();
    let n = system.universe;
    for si in 0..m {
        if used_sets[si] {
            continue;
        }
        let set = &system.sets[si];
        if elements.iter().any(|&x| !set.contains(&x)) {
            continue;
        }
        for x in 0..n {
            if used_elems[x] || set.contains(&x) {
                continue;
            }
            used_sets[si] = true;
            used_elems[x] = true;
            set_indices.push(si);
            elements.push(x);
            if search_ladder(system, d, set_indices, elements, used_sets, used_elems) {
                return true;
            }
            set_indices.pop();
            elements.pop();
            used_sets[si] = false;
            used_elems[x] = false;
        }
    }
    false
}

pub fn closed_neighborhood_system(g: &Graph) -> SetSystem {
    let n = g.n();
    let mut sets = Vec::with_capacity(n);
    for v in 0..n {
        let mut s = BTreeSet::new();
        s.insert(v);
        s.extend(g.neighbors(v).iter().copied());
        sets.push(s);
    }
    SetSystem { universe: n, sets }
}

/// Smallest `q` such that the closed-neighborhood family is `(q+1)`-semi-ladder-free.
/// `None` only for the empty search cap, which this implementation does not hit:
/// the index is at most `n`.
pub fn semi_ladder_index(g: &Graph) -> Option<usize> {
    Some(set_semi_ladder_index(&closed_neighborhood_system(g)))
}

pub fn set_semi_ladder_index(system: &SetSystem) -> usize {
    let cap = system.universe.min(system.sets.len());
    let mut lo = 0usize;
    let mut hi = cap;
    // largest d that is NOT d-semi-ladder-free; index equals that d (0 if 1-free).
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if semi_ladder_witness(system, mid).is_some() {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

pub fn encapsulates(tuple: &[Vec<usize>], sequence: &[usize]) -> bool {
    if tuple.len() != sequence.len() {
        return false;
    }
    tuple
        .iter()
        .zip(sequence.iter())
        .all(|(fam, s)| fam.binary_search(s).is_ok() || fam.contains(s))
}

/// True when some ordering of `cover` is encapsulated by `tuple`.
pub fn encapsulates_cover(tuple: &[Vec<usize>], cover: &[usize]) -> bool {
    if tuple.len() != cover.len() {
        return false;
    }
    let mut order = cover.to_vec();
    order.sort_unstable();
    fn rec(tuple: &[Vec<usize>], order: &mut [usize], i: usize) -> bool {
        if i == order.len() {
            return encapsulates(tuple, order);
        }
        for j in i..order.len() {
            order.swap(i, j);
            if rec(tuple, order, i + 1) {
                return true;
            }
            order.swap(i, j);
        }
        false
    }
    // Dedup permutations of identical indices by sorting first and skipping duplicates.
    rec(tuple, &mut order, 0)
}

fn pow_bounded(base: usize, exp: usize) -> Option<usize> {
    if exp == 0 {
        return Some(1);
    }
    base.checked_pow(exp as u32)
}

/// Compact representation of all inclusion-wise minimal set covers of size at most `k`.
///
/// Runs Solve from the empty seed for every arity `ℓ ≤ k`, then canonicalizes
/// permutation-equivalent tuples and drops tuples subsumed by another. Both
/// steps preserve covering, disjointness, and completeness.
pub fn compact_representation(system: &SetSystem, k: usize, _d: usize) -> Vec<Tuple> {
    if k == 0 || system.universe == 0 && system.sets.is_empty() {
        if system.universe == 0 {
            return vec![Vec::new()];
        }
        return Vec::new();
    }
    let mut all = Vec::new();
    for ell in 1..=k {
        let seeds = vec![BTreeSet::new(); ell];
        let mut memo = HashMap::new();
        let part = solve(&seeds, system, &mut memo);
        all.extend(part);
    }
    minimize_representation(all)
}

fn solve(
    seeds: &[BTreeSet<usize>],
    system: &SetSystem,
    memo: &mut HashMap<Vec<Vec<usize>>, Vec<Tuple>>,
) -> Vec<Tuple> {
    let key: Vec<Vec<usize>> = seeds.iter().map(|s| s.iter().copied().collect()).collect();
    if let Some(hit) = memo.get(&key) {
        return hit.clone();
    }
    // Insert a placeholder so a cyclic dependency cannot recurse forever.
    // The recursion strictly grows a seed, so cycles do not occur; the placeholder
    // also caps accidental re-entry.
    memo.insert(key.clone(), Vec::new());

    let mut covered = BTreeSet::new();
    for s in seeds {
        covered.extend(s.iter().copied());
    }
    let result = if covered.len() == system.universe {
        vec![base_tuple(seeds, system)]
    } else {
        let e = (0..system.universe).find(|x| !covered.contains(x)).unwrap();
        let mut acc: Vec<Tuple> = Vec::new();
        for j in 0..seeds.len() {
            let mut need = seeds[j].clone();
            need.insert(e);
            let members: Vec<usize> = system
                .sets
                .iter()
                .enumerate()
                .filter(|(_, s)| need.iter().all(|x| s.contains(x)))
                .map(|(i, _)| i)
                .collect();
            if members.is_empty() {
                continue;
            }
            let mut ije = system.sets[members[0]].clone();
            for &idx in &members[1..] {
                ije = ije.intersection(&system.sets[idx]).copied().collect();
            }
            let mut next = seeds.to_vec();
            next[j] = ije;
            let sub = solve(&next, system, memo);
            acc.extend(sub);
        }
        dedup_tuples(acc)
    };
    memo.insert(key, result.clone());
    result
}

fn base_tuple(seeds: &[BTreeSet<usize>], system: &SetSystem) -> Tuple {
    let mut families: Vec<Vec<usize>> = seeds
        .iter()
        .map(|seed| {
            let mut fam: Vec<usize> = system
                .sets
                .iter()
                .enumerate()
                .filter(|(_, s)| seed.iter().all(|x| s.contains(x)))
                .map(|(i, _)| i)
                .collect();
            fam.sort_unstable();
            fam
        })
        .collect();
    let mut count = vec![0usize; system.sets.len()];
    for fam in &families {
        for &idx in fam {
            count[idx] += 1;
        }
    }
    for fam in &mut families {
        fam.retain(|&idx| count[idx] == 1);
    }
    families
}

fn dedup_tuples(mut tuples: Vec<Tuple>) -> Vec<Tuple> {
    for t in &mut tuples {
        canonicalize_tuple(t);
    }
    tuples.sort();
    tuples.dedup();
    tuples
}

fn canonicalize_tuple(t: &mut Tuple) {
    for fam in t.iter_mut() {
        fam.sort_unstable();
        fam.dedup();
    }
    t.sort();
}

fn minimize_representation(tuples: Vec<Tuple>) -> Vec<Tuple> {
    let mut tuples = dedup_tuples(tuples);
    tuples.retain(|t| t.iter().all(|f| !f.is_empty()) || t.is_empty());
    let mut drop = vec![false; tuples.len()];
    for i in 0..tuples.len() {
        if tuples[i].is_empty() {
            continue;
        }
        for j in 0..tuples.len() {
            if i == j || tuples[i].len() != tuples[j].len() {
                continue;
            }
            if subsumes(&tuples[j], &tuples[i]) && tuples[i] != tuples[j] {
                drop[i] = true;
                break;
            }
        }
    }
    tuples
        .into_iter()
        .zip(drop)
        .filter_map(|(t, d)| if d { None } else { Some(t) })
        .collect()
}

fn subsumes(big: &Tuple, small: &Tuple) -> bool {
    if big.len() != small.len() {
        return false;
    }
    let mut used = vec![false; big.len()];
    fn bt(i: usize, big: &Tuple, small: &Tuple, used: &mut [bool]) -> bool {
        if i == small.len() {
            return true;
        }
        for j in 0..big.len() {
            if used[j] {
                continue;
            }
            if is_subset(&small[i], &big[j]) {
                used[j] = true;
                if bt(i + 1, big, small, used) {
                    return true;
                }
                used[j] = false;
            }
        }
        false
    }
    bt(0, big, small, &mut used)
}

fn is_subset(a: &[usize], b: &[usize]) -> bool {
    a.iter().all(|x| b.binary_search(x).is_ok())
}

/// Minimum-vertex group Steiner tree. Groups need not be handled specially by
/// the caller when the optimum uses a Steiner vertex outside every group: the
/// search connects dummy terminals attached to each group, so outside vertices
/// are eligible, then the dummies are dropped. On the star with groups `{0}`
/// and `{2}` this returns `{0, 1, 2}`.
pub fn group_steiner_tree(g: &Graph, groups: &[Vec<usize>]) -> Option<Vec<usize>> {
    if groups.is_empty() {
        return Some(Vec::new());
    }
    let mut cleaned = Vec::with_capacity(groups.len());
    for group in groups {
        let mut gs: Vec<usize> = group.iter().copied().filter(|&v| v < g.n()).collect();
        gs.sort_unstable();
        gs.dedup();
        if gs.is_empty() {
            return None;
        }
        cleaned.push(gs);
    }
    if cleaned.len() == 1 {
        return Some(vec![cleaned[0][0]]);
    }
    if cleaned.len() > 20 {
        return steiner_fallback(g, &cleaned);
    }
    let mut augmented = g.clone();
    let mut terminals = Vec::with_capacity(cleaned.len());
    for group in &cleaned {
        let t = augmented.add_vertex();
        for &v in group {
            augmented.add_edge(t, v);
        }
        terminals.push(t);
    }
    // Dummy terminals are leaves only. Routing through a dummy would treat two
    // of its group members as adjacent even when G has no such edge.
    let mut leaf_only = vec![false; augmented.n()];
    for &t in &terminals {
        leaf_only[t] = true;
    }
    if let Some(tree) = steiner_vertices(&augmented, &terminals, &leaf_only) {
        let mut real: Vec<usize> = tree.into_iter().filter(|&v| v < g.n()).collect();
        real.sort_unstable();
        real.dedup();
        if group_tree_ok(g, &cleaned, &real) {
            return Some(real);
        }
    }
    steiner_fallback(g, &cleaned)
}

fn group_tree_ok(g: &Graph, groups: &[Vec<usize>], verts: &[usize]) -> bool {
    if verts.is_empty() || !set_is_connected(g, verts) {
        return false;
    }
    let chosen: BTreeSet<usize> = verts.iter().copied().collect();
    groups
        .iter()
        .all(|group| group.iter().any(|v| chosen.contains(v)))
}

fn steiner_fallback(g: &Graph, groups: &[Vec<usize>]) -> Option<Vec<usize>> {
    // Representative enumeration is only used past the DP cap. Tests stay under it.
    let ell = groups.len();
    if ell > 8 {
        return None;
    }
    let mut best: Option<Vec<usize>> = None;
    let mut choice = vec![0usize; ell];
    fn rec(
        g: &Graph,
        groups: &[Vec<usize>],
        i: usize,
        choice: &mut [usize],
        best: &mut Option<Vec<usize>>,
    ) {
        if i == groups.len() {
            let terminals: Vec<usize> = choice.to_vec();
            if let Some(tree) = steiner_vertices(g, &terminals, &[]) {
                if best
                    .as_ref()
                    .map(|b| tree.len() < b.len() || (tree.len() == b.len() && tree < *b))
                    .unwrap_or(true)
                {
                    *best = Some(tree);
                }
            }
            return;
        }
        for &v in &groups[i] {
            choice[i] = v;
            rec(g, groups, i + 1, choice, best);
        }
    }
    rec(g, groups, 0, &mut choice, &mut best);
    best
}

fn steiner_vertices(g: &Graph, terminals: &[usize], leaf_only: &[bool]) -> Option<Vec<usize>> {
    let n = g.n();
    let ell = terminals.len();
    if ell == 0 {
        return Some(Vec::new());
    }
    if ell == 1 {
        return if terminals[0] < n {
            Some(vec![terminals[0]])
        } else {
            None
        };
    }
    if ell > 20 {
        return None;
    }
    for &t in terminals {
        if t >= n {
            return None;
        }
    }
    let states = 1usize << ell;
    let inf = u32::MAX / 4;
    let mut dp = vec![vec![inf; n]; states];
    let mut kind = vec![vec![0u8; n]; states];
    let mut arg = vec![vec![0usize; n]; states];
    for (i, &t) in terminals.iter().enumerate() {
        let mask = 1usize << i;
        dp[mask][t] = 0;
        kind[mask][t] = 1;
    }
    for mask in 1..states {
        if mask.count_ones() >= 2 {
            let mut sub = mask;
            loop {
                sub = (sub - 1) & mask;
                if sub == 0 {
                    break;
                }
                let other = mask ^ sub;
                if other == 0 || sub > other {
                    continue;
                }
                for v in 0..n {
                    if leaf_only.get(v).copied().unwrap_or(false) {
                        continue;
                    }
                    let cost = dp[sub][v].saturating_add(dp[other][v]);
                    if cost < dp[mask][v] {
                        dp[mask][v] = cost;
                        kind[mask][v] = 2;
                        arg[mask][v] = sub;
                    }
                }
            }
        }
        let mut dist = dp[mask].clone();
        let mut via = vec![usize::MAX; n];
        let mut heap = BinaryHeap::new();
        for v in 0..n {
            if dist[v] < inf {
                heap.push((Reverse(dist[v]), v));
            }
        }
        let mut seen = vec![false; n];
        while let Some((Reverse(d), v)) = heap.pop() {
            if seen[v] || d != dist[v] {
                continue;
            }
            seen[v] = true;
            for &u in g.neighbors(v) {
                if leaf_only.get(u).copied().unwrap_or(false) {
                    continue;
                }
                let nd = d.saturating_add(1);
                if nd < dist[u] {
                    dist[u] = nd;
                    via[u] = v;
                    heap.push((Reverse(nd), u));
                }
            }
        }
        for v in 0..n {
            if dist[v] < dp[mask][v] {
                dp[mask][v] = dist[v];
                kind[mask][v] = 3;
                arg[mask][v] = via[v];
            }
        }
    }
    let full = states - 1;
    let mut best_v = None;
    let mut best_c = inf;
    for v in 0..n {
        if leaf_only.get(v).copied().unwrap_or(false) {
            continue;
        }
        if dp[full][v] < best_c {
            best_c = dp[full][v];
            best_v = Some(v);
        }
    }
    let root = best_v?;
    if best_c >= inf {
        return None;
    }
    let mut verts = BTreeSet::new();
    reconstruct(full, root, &kind, &arg, &mut verts);
    if verts.is_empty() {
        return None;
    }
    Some(verts.into_iter().collect())
}

fn reconstruct(
    mask: usize,
    v: usize,
    kind: &[Vec<u8>],
    arg: &[Vec<usize>],
    verts: &mut BTreeSet<usize>,
) {
    match kind[mask][v] {
        1 => {
            verts.insert(v);
        }
        2 => {
            let sub = arg[mask][v];
            reconstruct(sub, v, kind, arg, verts);
            reconstruct(mask ^ sub, v, kind, arg, verts);
        }
        3 => {
            let u = arg[mask][v];
            verts.insert(v);
            if u != v {
                reconstruct(mask, u, kind, arg, verts);
            }
        }
        _ => {}
    }
}

/// A connected set cover of size at most `k`, as indices into `F`.
pub fn connected_set_cover(system: &SetSystem, k: usize, d: usize) -> Option<Vec<usize>> {
    if system.universe == 0 {
        return Some(Vec::new());
    }
    if k == 0 {
        return None;
    }
    let rep = compact_representation(system, k, d);
    let ig = intersection_graph(system);
    let mut best: Option<Vec<usize>> = None;
    for tuple in &rep {
        if tuple.is_empty() || tuple.iter().any(|f| f.is_empty()) {
            continue;
        }
        if tuple.len() > k {
            continue;
        }
        let groups = tuple.clone();
        if let Some(tree) = group_steiner_tree(&ig, &groups) {
            if tree.len() <= k && is_connected_set_cover(system, &tree) {
                if best
                    .as_ref()
                    .map(|b| tree.len() < b.len() || (tree.len() == b.len() && tree < *b))
                    .unwrap_or(true)
                {
                    best = Some(tree);
                }
            }
        }
    }
    best
}

fn intersection_graph(system: &SetSystem) -> Graph {
    let m = system.sets.len();
    let mut g = Graph::new(m);
    for i in 0..m {
        for j in (i + 1)..m {
            if system.sets[i]
                .intersection(&system.sets[j])
                .next()
                .is_some()
            {
                g.add_edge(i, j);
            }
        }
    }
    g
}

fn is_connected_set_cover(system: &SetSystem, idxs: &[usize]) -> bool {
    if !covers_universe(system, idxs) {
        return false;
    }
    let ig = intersection_graph(system);
    set_is_connected(&ig, idxs)
}

pub fn covers_universe(system: &SetSystem, idxs: &[usize]) -> bool {
    let mut seen = vec![false; system.universe];
    let mut count = 0usize;
    for &i in idxs {
        if i >= system.sets.len() {
            continue;
        }
        for &x in &system.sets[i] {
            if x < system.universe && !seen[x] {
                seen[x] = true;
                count += 1;
            }
        }
    }
    count == system.universe
}

/// Replace `S` by one vertex whose open neighborhood is the union of the open
/// neighborhoods of `S`. Empty or singleton `S` leaves the graph unchanged.
pub fn group(g: &Graph, s: &[usize]) -> Graph {
    let sset: BTreeSet<usize> = s.iter().copied().filter(|&v| v < g.n()).collect();
    if sset.len() <= 1 {
        return g.clone();
    }
    let survivors: Vec<usize> = (0..g.n()).filter(|v| !sset.contains(v)).collect();
    let mut ng = Graph::new(survivors.len() + 1);
    let mut old_to_new = HashMap::new();
    for (i, &v) in survivors.iter().enumerate() {
        old_to_new.insert(v, i);
    }
    let w = survivors.len();
    for (i, &u) in survivors.iter().enumerate() {
        for &v in g.neighbors(u) {
            if let Some(&j) = old_to_new.get(&v) {
                if i < j {
                    ng.add_edge(i, j);
                }
            }
        }
    }
    let mut neigh = BTreeSet::new();
    for &u in &sset {
        for &v in g.neighbors(u) {
            if let Some(&j) = old_to_new.get(&v) {
                neigh.insert(j);
            }
        }
    }
    for v in neigh {
        ng.add_edge(w, v);
    }
    ng
}

/// Spec FindCore: while the graph is larger than `k^d`, group a largest member
/// of a maximal collection of pairwise disjoint small intersections of at most
/// `d` closed neighborhoods. Returns the vertex set of the final graph.
pub fn find_core(g: &Graph, k: usize, d: usize) -> Vec<usize> {
    let mut cur = g.clone();
    for _ in 0..=g.n() {
        let n = cur.n();
        let threshold = pow_bounded(k, d).unwrap_or(usize::MAX);
        if n <= threshold {
            break;
        }
        let cands = small_neighborhood_intersections(&cur, d);
        let collection = maximal_disjoint(&cands);
        if collection.is_empty() {
            break;
        }
        let s_max = collection
            .iter()
            .max_by(|a, b| a.len().cmp(&b.len()).then_with(|| b.cmp(a)))
            .cloned()
            .unwrap_or_default();
        if s_max.len() < 2 {
            break;
        }
        let next = group(&cur, &s_max);
        if next.n() >= cur.n() {
            break;
        }
        cur = next;
    }
    (0..cur.n()).collect()
}

fn small_neighborhood_intersections(g: &Graph, d: usize) -> Vec<Vec<usize>> {
    let n = g.n();
    if d == 0 || n == 0 {
        return Vec::new();
    }
    let closed = closed_masks(g);
    let mut found = BTreeSet::new();
    let limit = d.min(n).min(12);
    let mut chosen = Vec::new();
    fn rec(
        start: usize,
        n: usize,
        limit: usize,
        closed: &[u64],
        d: usize,
        chosen: &mut Vec<usize>,
        found: &mut BTreeSet<Vec<usize>>,
    ) {
        if !chosen.is_empty() {
            let mut bits = u64::MAX;
            for &v in chosen.iter() {
                bits &= closed[v];
            }
            let set = bits_to_vec(bits, n);
            if !set.is_empty() && set.len() <= d {
                found.insert(set);
            }
        }
        if chosen.len() == limit {
            return;
        }
        for v in start..n {
            chosen.push(v);
            rec(v + 1, n, limit, closed, d, chosen, found);
            chosen.pop();
        }
    }
    if n <= 64 {
        rec(0, n, limit, &closed, d, &mut chosen, &mut found);
    } else {
        // Fallback for large n: only single closed neighborhoods of size ≤ d.
        for v in 0..n {
            let mut s: Vec<usize> = g.neighbors(v).to_vec();
            s.push(v);
            s.sort_unstable();
            if s.len() <= d {
                found.insert(s);
            }
        }
    }
    found.into_iter().collect()
}

fn closed_masks(g: &Graph) -> Vec<u64> {
    let n = g.n().min(64);
    let mut masks = vec![0u64; g.n()];
    for v in 0..n {
        let mut m = 1u64 << v;
        for &u in g.neighbors(v) {
            if u < 64 {
                m |= 1u64 << u;
            }
        }
        masks[v] = m;
    }
    masks
}

fn bits_to_vec(bits: u64, n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..n.min(64) {
        if bits & (1u64 << i) != 0 {
            out.push(i);
        }
    }
    out
}

fn maximal_disjoint(cands: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by(|&i, &j| {
        cands[j]
            .len()
            .cmp(&cands[i].len())
            .then_with(|| cands[i].cmp(&cands[j]))
    });
    let mut picked: Vec<Vec<usize>> = Vec::new();
    let mut used = BTreeSet::new();
    for i in order {
        if cands[i].iter().any(|v| used.contains(v)) {
            continue;
        }
        for &v in &cands[i] {
            used.insert(v);
        }
        picked.push(cands[i].clone());
    }
    picked
}

// --- Graph predicates and domination numbers ---------------------------------

pub fn is_complete(g: &Graph) -> bool {
    let n = g.n();
    if n <= 1 {
        return true;
    }
    for v in 0..n {
        if g.degree(v) != n - 1 {
            return false;
        }
    }
    true
}

pub fn count_components(g: &Graph) -> usize {
    let mut seen = vec![false; g.n()];
    let mut comps = 0usize;
    for s in 0..g.n() {
        if seen[s] {
            continue;
        }
        comps += 1;
        let mut q = std::collections::VecDeque::new();
        q.push_back(s);
        seen[s] = true;
        while let Some(v) = q.pop_front() {
            for &u in g.neighbors(v) {
                if !seen[u] {
                    seen[u] = true;
                    q.push_back(u);
                }
            }
        }
    }
    comps
}

pub fn is_connected(g: &Graph) -> bool {
    g.n() <= 1 || count_components(g) == 1
}

pub fn is_semi_ladder_free(g: &Graph, d: usize) -> bool {
    is_d_semi_ladder_free(&closed_neighborhood_system(g), d)
}

pub fn is_semi_ladder_free_with_witness(g: &Graph, d: usize) -> Option<Witness> {
    semi_ladder_witness(&closed_neighborhood_system(g), d)
}

fn normalized_set(set: &[usize], n: usize) -> BTreeSet<usize> {
    set.iter().copied().filter(|&v| v < n).collect()
}

pub fn is_dominating_set(g: &Graph, set: &[usize]) -> bool {
    let s = normalized_set(set, g.n());
    if g.n() == 0 {
        return true;
    }
    let mut dom = vec![false; g.n()];
    for &v in &s {
        dom[v] = true;
        for &u in g.neighbors(v) {
            dom[u] = true;
        }
    }
    dom.into_iter().all(|b| b)
}

pub fn set_is_connected(g: &Graph, set: &[usize]) -> bool {
    let s = normalized_set(set, g.n());
    if s.len() <= 1 {
        return true;
    }
    let start = *s.iter().next().unwrap();
    let mut seen = BTreeSet::new();
    let mut q = std::collections::VecDeque::new();
    q.push_back(start);
    seen.insert(start);
    while let Some(v) = q.pop_front() {
        for &u in g.neighbors(v) {
            if s.contains(&u) && seen.insert(u) {
                q.push_back(u);
            }
        }
    }
    seen.len() == s.len()
}

pub fn is_connected_dominating_set(g: &Graph, set: &[usize]) -> bool {
    let s = normalized_set(set, g.n());
    if g.n() == 0 {
        return s.is_empty();
    }
    if !is_connected(g) {
        return false;
    }
    !s.is_empty()
        && set_is_connected(g, &s.iter().copied().collect::<Vec<_>>())
        && is_dominating_set(g, &s.iter().copied().collect::<Vec<_>>())
}

pub fn dominating_set_size(g: &Graph, set: &[usize]) -> Option<usize> {
    if is_dominating_set(g, set) {
        Some(normalized_set(set, g.n()).len())
    } else {
        None
    }
}

pub fn connected_dominating_set_size(g: &Graph, set: &[usize]) -> Option<usize> {
    if is_connected_dominating_set(g, set) {
        Some(normalized_set(set, g.n()).len())
    } else {
        None
    }
}

pub fn domination_number(g: &Graph) -> Option<usize> {
    if g.n() == 0 {
        return Some(0);
    }
    if g.n() > 22 {
        return domination_number_backtrack(g);
    }
    let n = g.n();
    let closed = vertex_closed_mask(g);
    let full = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    let mut best = n;
    for mask in 0..(1u64 << n) {
        let bits = mask.count_ones() as usize;
        if bits >= best {
            continue;
        }
        let mut cov = 0u64;
        let mut m = mask;
        while m != 0 {
            let b = m.trailing_zeros() as usize;
            cov |= closed[b];
            m &= m - 1;
        }
        if cov == full {
            best = bits;
        }
    }
    Some(best)
}

fn vertex_closed_mask(g: &Graph) -> Vec<u64> {
    let n = g.n();
    let mut closed = vec![0u64; n];
    for v in 0..n {
        let mut m = 1u64 << v;
        for &u in g.neighbors(v) {
            if u < 64 {
                m |= 1u64 << u;
            }
        }
        closed[v] = m;
    }
    closed
}

fn domination_number_backtrack(g: &Graph) -> Option<usize> {
    let n = g.n();
    let mut best = n;
    let mut chosen = Vec::new();
    fn rec(g: &Graph, start: usize, best: &mut usize, chosen: &mut Vec<usize>) {
        if chosen.len() >= *best {
            return;
        }
        if is_dominating_set(g, chosen) {
            *best = chosen.len();
            return;
        }
        for v in start..g.n() {
            chosen.push(v);
            rec(g, v + 1, best, chosen);
            chosen.pop();
        }
    }
    rec(g, 0, &mut best, &mut chosen);
    Some(best)
}

pub fn minimum_dominating_set(g: &Graph) -> Option<Vec<usize>> {
    let target = domination_number(g)?;
    if g.n() == 0 {
        return Some(Vec::new());
    }
    let mut chosen = Vec::new();
    fn rec(g: &Graph, start: usize, target: usize, chosen: &mut Vec<usize>) -> bool {
        if chosen.len() == target {
            return is_dominating_set(g, chosen);
        }
        if chosen.len() > target || start >= g.n() {
            return false;
        }
        for v in start..g.n() {
            if chosen.len() + (g.n() - v) < target {
                break;
            }
            chosen.push(v);
            if rec(g, v + 1, target, chosen) {
                return true;
            }
            chosen.pop();
        }
        false
    }
    if rec(g, 0, target, &mut chosen) {
        Some(chosen)
    } else {
        None
    }
}

pub fn connected_domination_number(g: &Graph) -> Option<usize> {
    minimum_connected_dominating_set(g).map(|s| s.len())
}

/// Connected dominating set of size at most `k`.
///
/// Closed neighborhoods are the set system. Each tuple of the compact
/// representation is solved as a group Steiner tree in `G` itself, not in the
/// intersection graph. Among trees of size at most `k`, the smallest is
/// returned; ties break toward the lexicographically smallest vertex list.
pub fn connected_dominating_set(g: &Graph, k: usize) -> Option<Vec<usize>> {
    if g.n() == 0 {
        return Some(Vec::new());
    }
    if k == 0 {
        return None;
    }
    if g.n() >= 2 && !is_connected(g) {
        return None;
    }
    for v in 0..g.n() {
        if g.degree(v) + 1 == g.n() {
            return Some(vec![v]);
        }
    }
    let system = closed_neighborhood_system(g);
    // Solve does not branch on d. `n` stands in for `semi_ladder_index(g) + 1`.
    let rep = compact_representation(&system, k, g.n());
    let mut best: Option<Vec<usize>> = None;
    for tuple in &rep {
        if tuple.is_empty() || tuple.iter().any(|fam| fam.is_empty()) {
            continue;
        }
        let Some(mut tree) = group_steiner_tree(g, tuple) else {
            continue;
        };
        tree.sort_unstable();
        tree.dedup();
        if tree.is_empty() || tree.len() > k || !is_connected_dominating_set(g, &tree) {
            continue;
        }
        let replace = match &best {
            None => true,
            Some(prev) => tree.len() < prev.len() || (tree.len() == prev.len() && tree < *prev),
        };
        if replace {
            best = Some(tree);
        }
    }
    best
}

/// Minimum connected dominating set, via the paper's decision procedure.
///
/// `None` when the graph is disconnected and has at least two vertices.
pub fn minimum_connected_dominating_set(g: &Graph) -> Option<Vec<usize>> {
    connected_dominating_set(g, g.n())
}

pub fn is_minimum_dominating_set(g: &Graph, set: &[usize]) -> bool {
    match (dominating_set_size(g, set), domination_number(g)) {
        (Some(sz), Some(opt)) => sz == opt,
        _ => false,
    }
}

pub fn is_minimum_connected_dominating_set(g: &Graph, set: &[usize]) -> bool {
    match (
        connected_dominating_set_size(g, set),
        connected_domination_number(g),
    ) {
        (Some(sz), Some(opt)) => sz == opt,
        _ => false,
    }
}

/// Deletion core from the task statement.
///
/// Returns `None` when `g` is not `d`-semi-ladder-free. A complete graph yields
/// the minimum-degree vertex (smallest index on ties). Otherwise deletable
/// vertices are removed in increasing index order until none remain. A vertex
/// is deletable when it is not a cut vertex and its closed neighborhood is not
/// a minimal separator.
pub fn deletion_core(g: &Graph, d: usize) -> Option<Vec<usize>> {
    if !is_semi_ladder_free(g, d) {
        return None;
    }
    if is_complete(g) {
        if g.n() == 0 {
            return Some(Vec::new());
        }
        let mut best = 0usize;
        for v in 1..g.n() {
            if g.degree(v) < g.degree(best) {
                best = v;
            }
        }
        return Some(vec![best]);
    }
    let mut alive: BTreeSet<usize> = (0..g.n()).collect();
    loop {
        let mut victim = None;
        let current = induced_on(g, &alive);
        if is_complete(&current.graph) {
            // Stay with the stated rule: completeness is checked on the input,
            // not rechecked. Still, a complete induced subgraph has no cut
            // vertex and closed neighborhoods are not separators once the
            // remaining graph is complete, so deletion continues.
        }
        for &v in &alive {
            if is_deletable(&current, v) {
                victim = Some(v);
                break;
            }
        }
        let Some(v) = victim else {
            break;
        };
        alive.remove(&v);
        if alive.is_empty() {
            break;
        }
    }
    Some(alive.into_iter().collect())
}

struct Induced {
    graph: Graph,
    new_of: HashMap<usize, usize>,
}

fn induced_on(g: &Graph, alive: &BTreeSet<usize>) -> Induced {
    let old_of: Vec<usize> = alive.iter().copied().collect();
    let (graph, _) = g.induced(&old_of);
    let new_of = old_of.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    Induced { graph, new_of }
}

fn is_deletable(ind: &Induced, old_v: usize) -> bool {
    let Some(&v) = ind.new_of.get(&old_v) else {
        return false;
    };
    if is_cut_vertex(&ind.graph, v) {
        return false;
    }
    let mut closed = vec![v];
    closed.extend(ind.graph.neighbors(v).iter().copied());
    closed.sort_unstable();
    !is_minimal_separator(&ind.graph, &closed)
}

fn is_cut_vertex(g: &Graph, v: usize) -> bool {
    let before = count_components(g);
    let mut h = g.clone();
    // Isolate v by clearing its edges, then ignore v in the component count.
    let nbs = h.neighbors(v).to_vec();
    for u in nbs {
        remove_edge(&mut h, v, u);
    }
    let after = count_components_ignoring(&h, v);
    after > before
}

fn remove_edge(g: &mut Graph, u: usize, v: usize) {
    if let Ok(i) = g.adj[u].binary_search(&v) {
        g.adj[u].remove(i);
    }
    if let Ok(i) = g.adj[v].binary_search(&u) {
        g.adj[v].remove(i);
    }
}

fn count_components_ignoring(g: &Graph, ignore: usize) -> usize {
    let mut seen = vec![false; g.n()];
    seen[ignore] = true;
    let mut comps = 0usize;
    for s in 0..g.n() {
        if seen[s] {
            continue;
        }
        comps += 1;
        let mut q = std::collections::VecDeque::new();
        q.push_back(s);
        seen[s] = true;
        while let Some(v) = q.pop_front() {
            for &u in g.neighbors(v) {
                if !seen[u] {
                    seen[u] = true;
                    q.push_back(u);
                }
            }
        }
    }
    comps
}

/// `S` is a separator when `G - S` has at least two components. It is minimal
/// when no `S \ {x}` is a separator.
fn is_minimal_separator(g: &Graph, s: &[usize]) -> bool {
    if !is_separator(g, s) {
        return false;
    }
    for i in 0..s.len() {
        let mut sub = Vec::with_capacity(s.len() - 1);
        sub.extend(s[..i].iter().copied());
        sub.extend(s[i + 1..].iter().copied());
        if is_separator(g, &sub) {
            return false;
        }
    }
    true
}

fn is_separator(g: &Graph, s: &[usize]) -> bool {
    let ban: BTreeSet<usize> = s.iter().copied().collect();
    if ban.len() >= g.n() {
        return false;
    }
    let mut seen = vec![false; g.n()];
    for &v in &ban {
        if v < g.n() {
            seen[v] = true;
        }
    }
    let mut comps = 0usize;
    for start in 0..g.n() {
        if seen[start] {
            continue;
        }
        comps += 1;
        if comps >= 2 {
            return true;
        }
        let mut q = std::collections::VecDeque::new();
        q.push_back(start);
        seen[start] = true;
        while let Some(v) = q.pop_front() {
            for &u in g.neighbors(v) {
                if !seen[u] {
                    seen[u] = true;
                    q.push_back(u);
                }
            }
        }
    }
    false
}
