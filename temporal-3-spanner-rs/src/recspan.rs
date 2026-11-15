//! Algorithm 2, `RecSpan_f(G)`, and the choice of `f` from equation (3).

use crate::edges::EdgeSet;
use crate::graph::TemporalBiclique;
use crate::util::difference;
use crate::zorro::zorro;

/// `f(n) = ceil(n * e^(-sqrt(ln n)))`, equation (3).
///
/// This is the choice that turns the recurrence of Lemma 6 into the bound of
/// Lemma 7. The paper notes `2 <= f(n) <= n-1` for every `n >= 3`; the clamp is
/// there so a rounding wobble at the low end cannot hand `Zorro` an out of range
/// `k`.
pub fn default_f(n: usize) -> usize {
    assert!(n >= 3);
    let x = n as f64;
    let k = (x * (-(x.ln().sqrt())).exp()).ceil() as usize;
    k.clamp(1, n - 1)
}

/// The size bound of Theorem 4, `n * e^(2 sqrt(ln n)) = n^(1 + 2/sqrt(ln n))`.
pub fn size_bound(n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let x = n as f64;
    x * (2.0 * x.ln().sqrt()).exp()
}

fn rec<F: Fn(usize) -> usize>(
    g: &TemporalBiclique,
    sources: &[usize],
    targets: &[usize],
    f: &F,
    out: &mut EdgeSet,
) {
    let n = sources.len();
    debug_assert_eq!(n, targets.len());

    // Lines 1 and 2.
    if n <= 2 {
        for &s in sources {
            for &t in targets {
                out.insert(s, t);
            }
        }
        return;
    }

    // Line 3.
    let k = f(n);
    assert!(
        (1..n).contains(&k),
        "f(n) must lie in 1..=n-1, got f({n})={k}"
    );

    // Lines 4 and 5. The second call is the same Zorro on the same graph with
    // the complementary parameter, so |S_1'| = n-k and |T_2'| = k.
    let a = zorro(g, sources, targets, k);
    let b = zorro(g, sources, targets, n - k);
    for &(s, t) in a.z.iter().chain(b.z.iter()) {
        out.insert(s, t);
    }

    // Line 6. Z covers S_1 x T_2 and Z' covers S_1' x T_2', so what is left is
    // s in S_1 with t outside T_2, t in T_2' with s outside S_1', and the rest.
    let t_not_2 = difference(targets, &a.t2);
    let s_not_1p = difference(sources, &b.s1);
    let s_not_1 = difference(sources, &a.s1);
    let t_not_2p = difference(targets, &b.t2);

    rec(g, &a.s1, &t_not_2, f, out);
    rec(g, &s_not_1p, &b.t2, f, out);
    rec(g, &s_not_1, &t_not_2p, f, out);
}

/// `RecSpan_f(G)` on a balanced temporal biclique, with a caller supplied `f`.
pub fn rec_span_with<F: Fn(usize) -> usize>(g: &TemporalBiclique, f: F) -> EdgeSet {
    assert!(g.is_balanced(), "RecSpan needs a balanced biclique");
    let mut out = EdgeSet::new(g.n_sources(), g.n_targets());
    rec(g, &g.all_sources(), &g.all_targets(), &f, &mut out);
    out
}

/// `RecSpan_f(G)` with the `f` of equation (3). This is Theorem 4.
pub fn rec_span(g: &TemporalBiclique) -> EdgeSet {
    rec_span_with(g, default_f)
}
