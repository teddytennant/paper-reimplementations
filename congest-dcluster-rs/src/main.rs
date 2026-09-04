//! Run DCluster on a generated graph and print what it cost.

use congest_dcluster::congest::Network;
use congest_dcluster::dcluster::{
    ceil_log2, cluster_count_bound, dcluster, delta, diameter_bound, i_max, message_bound,
};
use congest_dcluster::graph::{self, Graph};
use congest_dcluster::partition::{measure_alpha_beta, partition_from_clustering, sample_incident_edges};

fn main() {
    let mut n = 512usize;
    let mut seed = 1u64;
    let mut family = String::from("er");
    let mut p = 0.25f64;
    let mut kappa = 1.0 / 3.0;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut k = 0;
    while k < args.len() {
        let take = |k: usize| -> String {
            args.get(k + 1).cloned().unwrap_or_else(|| {
                eprintln!("missing value for {}", args[k]);
                std::process::exit(2);
            })
        };
        match args[k].as_str() {
            "--n" => {
                n = take(k).parse().expect("--n wants an integer");
                k += 2;
            }
            "--seed" => {
                seed = take(k).parse().expect("--seed wants an integer");
                k += 2;
            }
            "--graph" => {
                family = take(k);
                k += 2;
            }
            "--p" => {
                p = take(k).parse().expect("--p wants a float");
                k += 2;
            }
            "--kappa" => {
                kappa = take(k).parse().expect("--kappa wants a float");
                k += 2;
            }
            "-h" | "--help" => {
                println!("dcluster [--n N] [--seed S] [--graph er|pa|path|cycle|star|clique|barbell] [--p P] [--kappa K]");
                return;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }

    let g: Graph = match family.as_str() {
        "er" => graph::erdos_renyi(n, p, seed),
        "pa" => graph::preferential_attachment(n, 3, seed),
        "path" => graph::path(n),
        "cycle" => graph::cycle(n),
        "star" => graph::star(n),
        "clique" => graph::clique(n),
        "barbell" => graph::barbell(n),
        other => {
            eprintln!("unknown graph family {other}");
            std::process::exit(2);
        }
    };

    let c = dcluster(&g, seed);
    let bound = message_bound(n);

    println!("graph      {family}, seed {seed}");
    println!("n          {}", g.n());
    println!("m          {}", g.m());
    println!("imax       {}  (ceil(log2 n) = {})", i_max(n), ceil_log2(n));
    println!("rounds     {}  (4 ceil(log2 n) = {})", c.rounds, 4 * ceil_log2(n));
    println!(
        "messages   {}  (Lemma 4.2 bound 48 n ln n ceil(log2 n) = {:.0}, ratio {:.4})",
        c.messages,
        bound,
        c.messages as f64 / bound
    );
    println!("clusters   {}", c.cluster_count());
    println!("|F|        {}", c.f.len());
    println!("unclustered {}  (Lemma 4.1(1) holds when this is 0)", c.orphans.len());

    println!("\nclusters by class, against the Lemma 4.1(4) bound 12 n ln n / Delta_i:");
    for i in (0..=i_max(n)).rev() {
        let count = c.classes[i].len();
        if count == 0 {
            continue;
        }
        println!(
            "  i = {i:2}  Delta_i = {:5}  |C^(i)| = {count:5}  bound {:9.1}  {}",
            delta(i),
            cluster_count_bound(n, i),
            if count as f64 <= cluster_count_bound(n, i) { "ok" } else { "OVER" }
        );
    }

    let (part_of, part_count) = partition_from_clustering(&c);
    let ab = measure_alpha_beta(&g, &part_of, part_count, &c.f);
    match ab.alpha {
        Some(a) => println!(
            "\nmax cluster diameter in (V, F): {a}  (Lemma 4.1(3) bound {})",
            diameter_bound(n)
        ),
        None => println!("\nsome cluster is disconnected in (V, F)"),
    }
    println!(
        "clustering read as a partition (Definition 3.1): alpha = {:?}, beta = {:?}, {} open parts, {} closed",
        ab.alpha, ab.beta, ab.open_parts, ab.closed_parts
    );

    let mut net = Network::new(&g);
    let j = sample_incident_edges(&mut net, &c, kappa, seed ^ 0x5eed);
    println!(
        "\nstep 2 of Algorithm Partition at kappa = {kappa:.3}: |J| = {} of {} edges, {} messages",
        j.len(),
        g.m(),
        net.messages()
    );
}
