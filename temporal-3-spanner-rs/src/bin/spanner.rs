//! Build a 3-spanner of a random temporal clique and print its size against the
//! bounds from the paper.
//!
//! Usage: spanner [n] [lifetime] [seed]

use std::env;
use temporal_spanner::{
    bounded_size_bound, clique_spanner, clique_spanner_bounded, random_clique, size_bound,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let arg = |i: usize, d: u64| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let n = arg(1, 200) as usize;
    let lifetime = arg(2, 8) as u32;
    let seed = arg(3, 0);

    let g = random_clique(n, lifetime, seed);
    let l = g.lifetime();
    let trivial = n * n.saturating_sub(1) / 2;

    let general = clique_spanner(&g);
    println!("n = {n}, lifetime = {l}, seed = {seed}");
    println!("clique edges                {trivial}");
    println!(
        "RecSpan_f                   {} (bound n*e^(2 sqrt ln n) = {:.0})",
        general.len(),
        size_bound(n)
    );

    let bounded = clique_spanner_bounded(&g, l);
    println!(
        "bounded lifetime            {} (bound 12*L*n = {})",
        bounded.len(),
        bounded_size_bound(n, l)
    );
}
