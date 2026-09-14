//! SCSS decision via the ear-decomposition DP of Section 3.1.

use crate::distance::{add_cost, INF};

/// Minimum cost of a strong spanning subgraph of the single-cost digraph
/// on `n` vertices with edges `(u, v, cost)`. `n <= 1` costs 0.
pub fn min_scss_cost(n: usize, edges: &[(usize, usize, u64)]) -> u64 {
    if n <= 1 {
        return 0;
    }
    assert!(n <= 12, "unweighted ear DP supports n<=12 terminals");
    let nmask = 1usize << n;
    let mut edge = vec![vec![INF; n]; n];
    for &(u, v, c) in edges {
        if u < n && v < n && u != v && c < edge[u][v] {
            edge[u][v] = c;
        }
    }

    // P[u][v][X]: min cost of a u-to-v path with internal vertices exactly X.
    let mut p = vec![vec![vec![INF; nmask]; n]; n];
    for u in 0..n {
        for v in 0..n {
            if u != v {
                p[u][v][0] = edge[u][v];
            }
        }
    }
    for mask in 1..nmask {
        for u in 0..n {
            if mask & (1 << u) != 0 {
                continue;
            }
            for v in 0..n {
                if u == v || mask & (1 << v) != 0 {
                    continue;
                }
                let mut best = INF;
                let mut bits = mask;
                while bits != 0 {
                    let x = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let sub = mask ^ (1 << x);
                    let c = add_cost(p[u][x][sub], edge[x][v]);
                    if c < best {
                        best = c;
                    }
                }
                p[u][v][mask] = best;
            }
        }
    }

    // cycle[u][X] = P[u, u, X]: cycle through X union {u} starting at u.
    let mut cycle = vec![vec![INF; nmask]; n];
    for u in 0..n {
        for mask in 0..nmask {
            if mask & (1 << u) != 0 {
                continue;
            }
            if mask == 0 {
                continue;
            }
            let mut best = INF;
            let mut bits = mask;
            while bits != 0 {
                let x = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let sub = mask ^ (1 << x);
                let c = add_cost(p[u][x][sub], edge[x][u]);
                if c < best {
                    best = c;
                }
            }
            cycle[u][mask] = best;
        }
    }

    // D[q][X]: min cost of a q-ear decomposition on vertex set X.
    let mut d = vec![vec![INF; nmask]; n];
    for mask in 1..nmask {
        let mut best = INF;
        let mut bits = mask;
        while bits != 0 {
            let u = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let internals = mask ^ (1 << u);
            if cycle[u][internals] < best {
                best = cycle[u][internals];
            }
        }
        d[1][mask] = best;
    }

    for q in 2..n {
        for mask in 1..nmask {
            let mut best = INF;
            let mut y = mask;
            while y != 0 {
                if y != mask {
                    let rest = mask ^ y;
                    let mut add = INF;
                    let mut r1 = rest;
                    while r1 != 0 {
                        let u = r1.trailing_zeros() as usize;
                        r1 &= r1 - 1;
                        let mut r2 = rest;
                        while r2 != 0 {
                            let v = r2.trailing_zeros() as usize;
                            r2 &= r2 - 1;
                            let pc = if u == v {
                                cycle[u][y]
                            } else {
                                p[u][v][y]
                            };
                            if pc < add {
                                add = pc;
                            }
                        }
                    }
                    let c = add_cost(d[q - 1][rest], add);
                    if c < best {
                        best = c;
                    }
                }
                y = (y - 1) & mask;
            }
            d[q][mask] = best;
        }
    }

    let full = nmask - 1;
    let mut best = INF;
    for q in 1..n {
        if d[q][full] < best {
            best = d[q][full];
        }
    }
    best
}

pub fn scss_unweighted(n: usize, edges: &[(usize, usize, u64)], k: u64) -> bool {
    min_scss_cost(n, edges) <= k
}

/// Minimum second cost of an SCSS whose first costs sum to at most `k`.
pub fn min_scss_weight(n: usize, edges: &[(usize, usize, u64, u64)], k: usize) -> u64 {
    if n <= 1 {
        return 0;
    }
    assert!(n <= 10, "weighted ear DP supports n<=10 terminals");
    let nmask = 1usize << n;
    let hk = k + 1;
    let mut cw = vec![vec![vec![INF; hk]; n]; n];
    for &(u, v, ell, w) in edges {
        let h = ell as usize;
        if u < n && v < n && u != v && h <= k && w < cw[u][v][h] {
            cw[u][v][h] = w;
        }
    }

    // P[u][v][X][h]
    let mut p = vec![vec![vec![vec![INF; hk]; nmask]; n]; n];
    for u in 0..n {
        for v in 0..n {
            if u != v {
                for h in 0..hk {
                    p[u][v][0][h] = cw[u][v][h];
                }
            }
        }
    }
    for mask in 1..nmask {
        for u in 0..n {
            if mask & (1 << u) != 0 {
                continue;
            }
            for v in 0..n {
                if u == v || mask & (1 << v) != 0 {
                    continue;
                }
                let mut bits = mask;
                while bits != 0 {
                    let x = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let sub = mask ^ (1 << x);
                    for h1 in 0..hk {
                        let c1 = p[u][x][sub][h1];
                        if c1 >= INF {
                            continue;
                        }
                        for ell in 0..=(k - h1) {
                            let w = cw[x][v][ell];
                            if w >= INF {
                                continue;
                            }
                            let h = h1 + ell;
                            let tot = add_cost(c1, w);
                            if tot < p[u][v][mask][h] {
                                p[u][v][mask][h] = tot;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut cycle = vec![vec![vec![INF; hk]; nmask]; n];
    for u in 0..n {
        for mask in 1..nmask {
            if mask & (1 << u) != 0 {
                continue;
            }
            let mut bits = mask;
            while bits != 0 {
                let x = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let sub = mask ^ (1 << x);
                for h1 in 0..hk {
                    let c1 = p[u][x][sub][h1];
                    if c1 >= INF {
                        continue;
                    }
                    for ell in 0..=(k - h1) {
                        let w = cw[x][u][ell];
                        if w >= INF {
                            continue;
                        }
                        let h = h1 + ell;
                        let tot = add_cost(c1, w);
                        if tot < cycle[u][mask][h] {
                            cycle[u][mask][h] = tot;
                        }
                    }
                }
            }
        }
    }

    let mut d = vec![vec![vec![INF; hk]; nmask]; n];
    for mask in 1..nmask {
        let mut bits = mask;
        while bits != 0 {
            let u = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let internals = mask ^ (1 << u);
            for h in 0..hk {
                let c = cycle[u][internals][h];
                if c < d[1][mask][h] {
                    d[1][mask][h] = c;
                }
            }
        }
    }

    for q in 2..n {
        for mask in 1..nmask {
            let mut y = mask;
            while y != 0 {
                if y != mask {
                    let rest = mask ^ y;
                    let mut rbits = Vec::new();
                    let mut r = rest;
                    while r != 0 {
                        rbits.push(r.trailing_zeros() as usize);
                        r &= r - 1;
                    }
                    for &u in &rbits {
                        for &v in &rbits {
                            for h1 in 0..hk {
                                let dprev = d[q - 1][rest][h1];
                                if dprev >= INF {
                                    continue;
                                }
                                for h2 in 0..=(k - h1) {
                                    let pc = if u == v {
                                        cycle[u][y][h2]
                                    } else {
                                        p[u][v][y][h2]
                                    };
                                    if pc >= INF {
                                        continue;
                                    }
                                    let h = h1 + h2;
                                    let tot = add_cost(dprev, pc);
                                    if tot < d[q][mask][h] {
                                        d[q][mask][h] = tot;
                                    }
                                }
                            }
                        }
                    }
                }
                y = (y - 1) & mask;
            }
        }
    }

    let full = nmask - 1;
    let mut best = INF;
    for q in 1..n {
        for h in 0..hk {
            if d[q][full][h] < best {
                best = d[q][full][h];
            }
        }
    }
    best
}

pub fn scss_weighted(n: usize, edges: &[(usize, usize, u64, u64)], k: usize, t: u64) -> bool {
    min_scss_weight(n, edges, k) <= t
}
