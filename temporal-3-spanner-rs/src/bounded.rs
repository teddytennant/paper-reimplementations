//! Section 4: temporal bicliques of bounded lifetime, Lemmas 9, 10, 11 and
//! Theorem 12, giving a 3-bispanner of size at most `12 L n`.

use crate::edges::EdgeSet;
use crate::graph::{Label, TemporalBiclique};
use crate::util::difference;

/// A biclique seen either as it is, or with the two sides swapped and every
/// label `i` replaced by `L + 1 - i`.
///
/// Lemmas 10 and 11 both come with a symmetric statement that the paper gets by
/// "interchanging S and T, replacing each label i by L+1-i, and reversing the
/// resulting paths". Reversing a path that is nondecreasing in the mirrored
/// labels gives a path that is nondecreasing in the real ones, so one
/// implementation covers both directions. In the swapped view, `left` holds real
/// targets and `right` holds real sources, and covering `left x right` in the
/// view means covering `right x left` as real source-target pairs.
struct View<'a> {
    g: &'a TemporalBiclique,
    swapped: bool,
    lifetime: Label,
}

impl View<'_> {
    /// The label of the edge between `a` on the left side and `b` on the right.
    #[inline]
    fn lab(&self, a: usize, b: usize) -> Label {
        if self.swapped {
            self.lifetime + 1 - self.g.label(b, a)
        } else {
            self.g.label(a, b)
        }
    }

    #[inline]
    fn emit(&self, out: &mut EdgeSet, a: usize, b: usize) {
        if self.swapped {
            out.insert(b, a);
        } else {
            out.insert(a, b);
        }
    }
}

/// Lemma 9. Finds `s`, `t` maximizing `|S_st| + |T_st|`, writes the two stars
/// into `out`, and returns `(S', T') = (S_st, T_st)`.
///
/// `S_st = {s' : lambda(s',t) <= lambda(s,t)}` and
/// `T_st = {t' : lambda(s,t') >= lambda(s,t)}`, so for `s' in S_st` and
/// `t' in T_st` the path `(s', t, s, t')` is temporal. The averaging argument in
/// the paper gives `|S'| + |T'| >= n + n/L`.
fn lemma9(
    g: &TemporalBiclique,
    sources: &[usize],
    targets: &[usize],
    out: &mut EdgeSet,
) -> (Vec<usize>, Vec<usize>) {
    let mut best_score = 0usize;
    let mut best = (sources[0], targets[0]);
    for &s in sources {
        for &t in targets {
            let lam = g.label(s, t);
            let mut score = 0usize;
            for &s2 in sources {
                if g.label(s2, t) <= lam {
                    score += 1;
                }
            }
            for &t2 in targets {
                if g.label(s, t2) >= lam {
                    score += 1;
                }
            }
            if score > best_score {
                best_score = score;
                best = (s, t);
            }
        }
    }
    let (s, t) = best;
    let lam = g.label(s, t);
    let s_prime: Vec<usize> = sources
        .iter()
        .copied()
        .filter(|&s2| g.label(s2, t) <= lam)
        .collect();
    let t_prime: Vec<usize> = targets
        .iter()
        .copied()
        .filter(|&t2| g.label(s, t2) >= lam)
        .collect();
    for &s2 in &s_prime {
        out.insert(s2, t);
    }
    for &t2 in &t_prime {
        out.insert(s, t2);
    }
    (s_prime, t_prime)
}

/// Lemma 10, in the orientation of `v`. Needs `|right| = |left| + r` and
/// `r < |left| / (L - 1)`.
///
/// Picks `s*` on the left maximizing the sum of its labels, takes `T_0` to be
/// the `r` largest neighbors of `s*`, and for each `s` finds some
/// `t_s` outside `T_0` with `lambda(s,t_s) <= lambda(s*,t_s)`. The returned
/// edges cover `left x T_0` through the paths `(s, t_s, s*, t)`, and number at
/// most `2|left| + r`.
fn lemma10(v: &View, left: &[usize], right: &[usize], r: usize, out: &mut EdgeSet) -> Vec<usize> {
    let n = left.len();
    debug_assert_eq!(right.len(), n + r);
    if r == 0 {
        return Vec::new();
    }
    if n == 0 {
        return right[..r].to_vec();
    }

    let mut s_star = left[0];
    let mut best: u64 = right.iter().map(|&b| v.lab(left[0], b) as u64).sum();
    for &a in &left[1..] {
        let total: u64 = right.iter().map(|&b| v.lab(a, b) as u64).sum();
        if total > best {
            best = total;
            s_star = a;
        }
    }

    let mut ord = right.to_vec();
    ord.sort_unstable_by_key(|&b| (v.lab(s_star, b), b));
    let (rest, t0) = ord.split_at(n);

    for &a in left {
        // The counting argument in the proof says such a t_s exists; if it did
        // not, s* would not have had the largest label sum.
        let ts = rest
            .iter()
            .copied()
            .find(|&b| v.lab(a, b) <= v.lab(s_star, b))
            .expect("Lemma 10: no valid t_s, the precondition r < n/(L-1) must have been broken");
        v.emit(out, a, ts);
        v.emit(out, s_star, ts);
    }
    for &b in t0 {
        v.emit(out, s_star, b);
    }
    t0.to_vec()
}

/// Lemma 11, in the orientation of `v`. Handles any `r = |right| - |left|`, and
/// returns a set of `r` elements of `right` that the emitted edges cover
/// against all of `left`. At most `3|left| + (4L+1)r` edges.
fn lemma11(v: &View, left: &[usize], right: &[usize], out: &mut EdgeSet) -> Vec<usize> {
    let n = left.len();
    let r = right.len() - n;
    let l = v.lifetime as usize;
    assert!(l >= 2, "Lemma 11 assumes lifetime at least 2");
    if r == 0 {
        return Vec::new();
    }
    // r < n/(L-1): Lemma 10 applies directly.
    if r * (l - 1) < n {
        return lemma10(v, left, right, r, out);
    }
    // Small left side: take everything, at most L*r edges.
    if n < l {
        let t0 = right[..r].to_vec();
        for &a in left {
            for &b in &t0 {
                v.emit(out, a, b);
            }
        }
        return t0;
    }

    // Otherwise peel r elements off in batches of q, each batch small enough for
    // Lemma 10 to apply.
    let q = (n - 1) / (l - 1);
    debug_assert!(q >= 1);
    let t_prime = &right[..r];
    let mut t_star: Vec<usize> = right[r..].to_vec();
    let mut t0: Vec<usize> = Vec::with_capacity(r);
    for batch in t_prime.chunks(q) {
        let mut cur = t_star.clone();
        cur.extend_from_slice(batch);
        let removed = lemma10(v, left, &cur, batch.len(), out);
        t_star = difference(&cur, &removed);
        t0.extend_from_slice(&removed);
    }
    debug_assert_eq!(t0.len(), r);
    t0
}

/// The size bound of Theorem 12.
pub fn bounded_size_bound(n: usize, lifetime: Label) -> usize {
    12 * (lifetime.max(1) as usize) * n
}

/// Theorem 12. A 3-bispanner of a balanced temporal biclique of lifetime at most
/// `lifetime`, of size at most `12 * lifetime * n`.
///
/// Each round applies Lemma 9 to get `S'`, `T'` covering `S' x T'` with `n + r`
/// edges where `r >= n/L`, then Lemma 11 twice to cover `S_0 x (T \ T')` and
/// `(S \ S') x T_0` for some `S_0` and `T_0` of size `r`. Everything touching
/// `S_0` or `T_0` is then covered, and the recursion continues on the balanced
/// biclique of size `n - r` that is left.
pub fn bounded_span(g: &TemporalBiclique, lifetime: Label) -> EdgeSet {
    assert!(
        g.is_balanced(),
        "the bounded lifetime construction needs a balanced biclique"
    );
    assert!(lifetime >= 1, "lifetime must be at least 1");
    assert!(
        g.lifetime() <= lifetime,
        "the graph has a label above the declared lifetime"
    );
    let n_all = g.n_sources();
    let mut out = EdgeSet::new(n_all, g.n_targets());
    if n_all == 0 {
        return out;
    }

    // With L = 1 every label is the same, so a star at one source together with
    // a star at one target is a 3-bispanner of size 2n-1.
    if lifetime <= 1 {
        for s in 0..n_all {
            out.insert(s, 0);
        }
        for t in 0..g.n_targets() {
            out.insert(0, t);
        }
        return out;
    }

    let mut sources = g.all_sources();
    let mut targets = g.all_targets();
    while !sources.is_empty() {
        let n = sources.len();
        let (s_prime, t_prime) = lemma9(g, &sources, &targets, &mut out);
        let r = s_prime.len() + t_prime.len() - n;
        debug_assert!(
            r >= 1 && r * (lifetime as usize) >= n,
            "Lemma 9 missed its |S'|+|T'| bound"
        );

        // Lemma 11 on G[S', T \ T'], where the source side is the larger one by
        // r, so it runs in the swapped view.
        let mirror = View {
            g,
            swapped: true,
            lifetime,
        };
        let t_not_prime = difference(&targets, &t_prime);
        let s0 = lemma11(&mirror, &t_not_prime, &s_prime, &mut out);

        // Lemma 11 on G[S \ S', T'], target side larger by r.
        let direct = View {
            g,
            swapped: false,
            lifetime,
        };
        let s_not_prime = difference(&sources, &s_prime);
        let t0 = lemma11(&direct, &s_not_prime, &t_prime, &mut out);

        sources = difference(&sources, &s0);
        targets = difference(&targets, &t0);
        debug_assert_eq!(sources.len(), targets.len());
    }
    out
}
