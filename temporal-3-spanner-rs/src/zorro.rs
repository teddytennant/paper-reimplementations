//! Algorithm 1, `Zorro(G, k)`.

use crate::graph::TemporalBiclique;

/// What `Zorro` returns, plus the two hub vertices, which the caller does not
/// need but the tests do.
#[derive(Clone, Debug)]
pub struct ZorroOut {
    /// `S_1`, of size exactly `k`.
    pub s1: Vec<usize>,
    /// `T_2`, of size exactly `n - k`.
    pub t2: Vec<usize>,
    /// `Z`, the union of the two stars, of size at most `n`.
    pub z: Vec<(usize, usize)>,
    /// The target hub `t*`.
    pub t_star: usize,
    /// The source hub `s*`.
    pub s_star: usize,
}

/// Rank the targets of `s` by `(label, target index)`.
///
/// The tie break matters. Line 2 takes the `k` smallest neighbors of `s` and
/// line 6 takes the `n - k` largest neighbors of `s*`; the proof of Lemma 5
/// needs `lambda({s*, t*}) <= lambda({s*, t})` for `t in T_2`, which holds
/// because the two lines cut the same total order at the same point.
fn ranked(g: &TemporalBiclique, s: usize, targets: &[usize], buf: &mut Vec<usize>) {
    buf.clear();
    buf.extend_from_slice(targets);
    buf.sort_unstable_by_key(|&t| (g.label(s, t), t));
}

/// `Zorro(G[sources, targets], k)` as in Algorithm 1.
///
/// The biclique has to be balanced, `|S| = |T| = n`, and `k in {1, ..., n-1}`.
/// By Lemma 5 the result satisfies `|S_1| = k`, `|T_2| = n - k`, `|Z| <= n`, and
/// `Z` covers `S_1 x T_2` with paths of at most 3 hops.
pub fn zorro(g: &TemporalBiclique, sources: &[usize], targets: &[usize], k: usize) -> ZorroOut {
    let n = sources.len();
    assert_eq!(n, targets.len(), "Zorro needs a balanced biclique");
    assert!(
        (1..n).contains(&k),
        "k must lie in 1..=n-1, got k={k} n={n}"
    );

    // Lines 1 and 2, with the counting implementation from the proof of Lemma 5:
    // one counter and one list per target.
    let mut count = vec![0usize; n];
    let mut lists: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut buf: Vec<usize> = Vec::with_capacity(n);
    let mut pos = vec![0usize; g.n_targets()];
    for (i, &t) in targets.iter().enumerate() {
        pos[t] = i;
    }
    for &s in sources {
        ranked(g, s, targets, &mut buf);
        for &t in &buf[..k] {
            count[pos[t]] += 1;
            lists[pos[t]].push(s);
        }
    }

    // Line 3. The pigeonhole argument in Lemma 5 says the maximum counter is at
    // least k, since the k smallest neighbors over all sources are k*n edges
    // spread over n targets.
    let mut best = 0usize;
    for i in 1..n {
        if count[i] > count[best] {
            best = i;
        }
    }
    assert!(
        count[best] >= k,
        "pigeonhole failed: no target is among the k smallest for k sources"
    );
    let t_star = targets[best];

    // Line 4.
    let s1: Vec<usize> = lists[best][..k].to_vec();

    // Line 5: s* is the largest neighbor of t* inside S_1.
    let mut s_star = s1[0];
    for &s in &s1[1..] {
        if g.label(s, t_star) > g.label(s_star, t_star) {
            s_star = s;
        }
    }

    // Line 6: T_2 is the n-k largest neighbors of s*, i.e. everything after the
    // k smallest in the same order used on line 2.
    ranked(g, s_star, targets, &mut buf);
    let t2: Vec<usize> = buf[k..].to_vec();

    // Line 7.
    let mut z = Vec::with_capacity(n);
    for &s in &s1 {
        z.push((s, t_star));
    }
    for &t in &t2 {
        z.push((s_star, t));
    }
    z.sort_unstable();
    z.dedup();

    ZorroOut {
        s1,
        t2,
        z,
        t_star,
        s_star,
    }
}
