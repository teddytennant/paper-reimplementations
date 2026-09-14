use std::env;
use std::io::{self, BufRead};
use std::process;

use sca_scss::{sca_unweighted, sca_weighted, DiGraph};

fn demo() {
    let mut g = DiGraph::new(3);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    let links = [(2, 0)];
    let yes = sca_unweighted(&g, &links, 1);
    println!("demo: chain 0->1->2 with link 2->0, k=1");
    println!("{}", if yes { "yes" } else { "no" });
}

fn parse_stdin() -> io::Result<()> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mode = match lines.next() {
        Some(l) => l?,
        None => {
            demo();
            return Ok(());
        }
    };
    let mode = mode.trim().to_lowercase();
    let header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header"))??;
    let nums: Vec<u64> = header
        .split_whitespace()
        .map(|s| s.parse().unwrap_or(0))
        .collect();
    if nums.len() < 3 {
        eprintln!("header: n m k [t]");
        process::exit(2);
    }
    let n = nums[0] as usize;
    let m = nums[1] as usize;
    let k = nums[2] as usize;
    let mut g = DiGraph::new(n);
    for _ in 0..m {
        let line = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing edge"))??;
        let mut it = line.split_whitespace();
        let u: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let v: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        g.add_edge(u, v);
    }
    let p_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing p"))??;
    let p: usize = p_line.trim().parse().unwrap_or(0);
    if mode.starts_with('w') {
        let t = if nums.len() >= 4 { nums[3] } else { 0 };
        let mut links = Vec::new();
        for _ in 0..p {
            let line = lines
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing link"))??;
            let mut it = line.split_whitespace();
            let u: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
            let v: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
            let w: u64 = it.next().unwrap_or("0").parse().unwrap_or(0);
            links.push((u, v, w));
        }
        let yes = sca_weighted(&g, &links, k, t);
        println!("{}", if yes { "yes" } else { "no" });
    } else {
        let mut links = Vec::new();
        for _ in 0..p {
            let line = lines
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing link"))??;
            let mut it = line.split_whitespace();
            let u: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
            let v: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
            links.push((u, v));
        }
        let yes = sca_unweighted(&g, &links, k);
        println!("{}", if yes { "yes" } else { "no" });
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        demo();
        return;
    }
    if args[0] == "-" {
        if let Err(err) = parse_stdin() {
            eprintln!("{err}");
            process::exit(1);
        }
        return;
    }
    eprintln!("usage: sca           # demo");
    eprintln!("       sca -         # stdin: unweighted|weighted, then n m k [t], edges, p, links");
    process::exit(2);
}
