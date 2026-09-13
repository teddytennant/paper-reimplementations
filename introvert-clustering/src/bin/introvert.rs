use std::env;
use std::process;

use introvert_clustering::{layered_decompose, Graph};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 1 {
        eprintln!("usage: introvert <n>");
        process::exit(2);
    }
    let n: usize = args[0].parse().unwrap_or_else(|_| {
        eprintln!("could not parse vertex count {}", args[0]);
        process::exit(2);
    });

    let mut g = Graph::new(n);
    if n >= 2 {
        for i in 0..n {
            let j = (i + 1) % n;
            if let Err(err) = g.add_edge(i, j) {
                eprintln!("{err}");
                process::exit(2);
            }
        }
    }

    let eps = 0.1;
    let beta = 0.5;
    let max_layers = ((20.0 * (n.max(2) as f64).ln()) as usize).max(1);
    let mut rng = StdRng::seed_from_u64(1);
    let decomp = layered_decompose(&g, eps, beta, max_layers, &mut rng).unwrap_or_else(|err| {
        eprintln!("{err}");
        process::exit(1);
    });

    let mut introvert = true;
    for layer in &decomp.layers {
        let gi = Graph::from_edges(g.n(), &layer.remaining).unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        });
        if !layer.clustering.introvert_ok(&gi, eps) {
            introvert = false;
            break;
        }
    }

    println!("n = {n}");
    println!("m = {}", g.m());
    println!("layers = {}", decomp.layers.len());
    println!("leftover = {}", decomp.leftover.len());
    println!("introvert_ok = {introvert}");
}
