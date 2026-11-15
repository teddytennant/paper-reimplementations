use std::env;
use std::process;

use polyamorous_scheduling::{
    deadline_driven_schedule, max_weighted_waiting_time, Graph, DEFAULT_MAX_DAYS,
};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: schedule <n> <u-v:g>...\n  example: schedule 3 0-1:1.0 0-2:0.5 1-2:0.5");
        process::exit(2);
    }
    let n: usize = args[0].parse().unwrap_or_else(|_| {
        eprintln!("could not parse vertex count {}", args[0]);
        process::exit(2);
    });
    let mut graph = Graph::new(n);
    for spec in &args[1..] {
        let (uv, gpart) = spec.split_once(':').unwrap_or_else(|| {
            eprintln!("expected u-v:g, got {spec}");
            process::exit(2);
        });
        let (u, v) = uv.split_once('-').unwrap_or_else(|| {
            eprintln!("expected u-v:g, got {spec}");
            process::exit(2);
        });
        let u: usize = u.parse().unwrap_or_else(|_| {
            eprintln!("bad vertex {u}");
            process::exit(2);
        });
        let v: usize = v.parse().unwrap_or_else(|_| {
            eprintln!("bad vertex {v}");
            process::exit(2);
        });
        let g: f64 = gpart.parse().unwrap_or_else(|_| {
            eprintln!("bad weight {gpart}");
            process::exit(2);
        });
        if let Err(err) = graph.add_edge(u, v, g) {
            eprintln!("{err}");
            process::exit(2);
        }
    }

    let g_star = graph.g_star();
    let schedule = deadline_driven_schedule(&graph, DEFAULT_MAX_DAYS).unwrap_or_else(|err| {
        eprintln!("{err}");
        process::exit(1);
    });
    let cost = max_weighted_waiting_time(&graph, &schedule.matchings);
    println!("G* = {g_star}");
    println!("period = {}", schedule.period());
    println!("transient = {}", schedule.transient);
    println!("paper_heat (online) = {}", schedule.max_heat);
    println!("ops_cost (periodic) = {cost}");
    if g_star > 0.0 {
        println!("online_heat / G* = {}", schedule.max_heat / g_star);
        println!("ops_cost / G* = {}", cost / g_star);
    }
    println!("schedule:");
    for (t, matching) in schedule.matchings.iter().enumerate() {
        let mut parts: Vec<String> = matching
            .iter()
            .map(|&e| {
                let edge = &graph.edges()[e];
                format!("{}-{}:{}", edge.u, edge.v, edge.g)
            })
            .collect();
        parts.sort();
        println!("  day {t}: {{{}}}", parts.join(", "));
    }
}
