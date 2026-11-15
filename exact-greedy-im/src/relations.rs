//! Bit-mask separator reachability relations.
//!
//! A relation on `k` ordered vertices is a `u64` whose bit `i * k + j` is set
//! iff `i ⇝ j`. Reachability is reflexive (identity is always included) and we
//! close under transitivity, matching the paper's `⇝` / `TC`.

/// Maximum number of vertices in an augmented bag (`w + 1` original plus `ρ`).
/// Width `w ≤ 2` gives bags of size at most 4.
pub const MAX_AUG_BAG: usize = 4;

#[inline]
pub fn bit(i: usize, j: usize, k: usize) -> u64 {
    1u64 << (i * k + j)
}

#[inline]
pub fn reaches(rel: u64, i: usize, j: usize, k: usize) -> bool {
    rel & bit(i, j, k) != 0
}

/// Reflexive identity relation on `k` vertices.
pub fn identity(k: usize) -> u64 {
    debug_assert!(k <= 8);
    let mut r = 0u64;
    for i in 0..k {
        r |= bit(i, i, k);
    }
    r
}

/// Reflexive-transitive closure (Floyd–Warshall on bits).
pub fn transitive_closure(mut rel: u64, k: usize) -> u64 {
    debug_assert!(k <= 8);
    rel |= identity(k);
    for mid in 0..k {
        for i in 0..k {
            if reaches(rel, i, mid, k) {
                for j in 0..k {
                    if reaches(rel, mid, j, k) {
                        rel |= bit(i, j, k);
                    }
                }
            }
        }
    }
    rel
}

pub fn union_then_tc(a: u64, b: u64, k: usize) -> u64 {
    transitive_closure(a | b, k)
}

pub fn index_of(verts: &[usize], v: usize) -> usize {
    verts
        .iter()
        .position(|&x| x == v)
        .unwrap_or_else(|| panic!("vertex {v} not in {verts:?}"))
}

/// Restrict a relation on `from` to the subset `to` (order of `to` is kept).
pub fn restrict(rel: u64, from: &[usize], to: &[usize]) -> u64 {
    let k = from.len();
    let s = to.len();
    let mut out = 0u64;
    for i in 0..s {
        let ii = index_of(from, to[i]);
        for j in 0..s {
            if reaches(rel, ii, index_of(from, to[j]), k) {
                out |= bit(i, j, s);
            }
        }
    }
    transitive_closure(out, s)
}

/// Embed a relation on `src` into a superset `dest`, filling identity on `dest`.
pub fn embed(rel: u64, src: &[usize], dest: &[usize]) -> u64 {
    let s = src.len();
    let k = dest.len();
    let mut out = identity(k);
    for i in 0..s {
        let ii = index_of(dest, src[i]);
        for j in 0..s {
            if reaches(rel, i, j, s) {
                out |= bit(ii, index_of(dest, src[j]), k);
            }
        }
    }
    out
}

/// Reachability relation induced by a list of directed edges on `verts`.
pub fn from_edges(verts: &[usize], edges: &[(usize, usize)]) -> u64 {
    let k = verts.len();
    let mut rel = identity(k);
    for &(u, v) in edges {
        if verts.contains(&u) && verts.contains(&v) {
            rel |= bit(index_of(verts, u), index_of(verts, v), k);
        }
    }
    transitive_closure(rel, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_tc_adds_skip() {
        let verts = [0, 1, 2];
        let rel = from_edges(&verts, &[(0, 1), (1, 2)]);
        assert!(reaches(rel, 0, 2, 3));
        assert!(reaches(rel, 0, 0, 3));
        assert!(!reaches(rel, 2, 0, 3));
    }

    #[test]
    fn restrict_and_embed_roundtrip_on_subset() {
        let bag = [0, 1, 2];
        let sep = [0, 2];
        let rel = from_edges(&bag, &[(0, 1), (1, 2)]);
        let rsep = restrict(rel, &bag, &sep);
        assert!(reaches(rsep, 0, 1, 2)); // 0 ⇝ 2 on the separator
        let lifted = embed(rsep, &sep, &bag);
        assert!(reaches(lifted, 0, 2, 3));
        assert!(!reaches(lifted, 0, 1, 3));
    }
}
