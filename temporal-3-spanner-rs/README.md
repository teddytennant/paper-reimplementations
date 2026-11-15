# temporal-3-spanner-rs

Temporal 3-spanners of temporal cliques and bicliques, from arXiv:2609.02851.
Zorro (Algorithm 1), the recursion RecSpan_f (Algorithm 2) with
f(n) = ceil(n e^(-sqrt(ln n))), the O(nL) bounded-lifetime construction of
Theorem 12, and the clique-to-biclique reduction of Observation 3.

## build

    cargo build --release
    cargo test --release

## run

    cargo run --release --bin spanner -- 200 8 0

Arguments are n, lifetime and seed. It prints the spanner size for a random
temporal clique next to n*e^(2 sqrt(ln n)) and 12Ln.

## what is checked

The size bounds are theorems in the paper; the tests only assert them on the
instances they generate, so passing is evidence and not a proof. The spanner
property is checked directly: a separate temporal BFS computes minimum-hop
distances from the edge list alone, and every ordered pair has to come out at
3 hops or fewer. Every edge carries exactly one label, following Section 2, so
multi-labelled input has to be thinned before it goes in.
