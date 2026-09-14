use std::env;
use std::io::{self, Read};
use std::process;

use frc_chunking::{pack, FrcParams};

fn usage() {
    eprintln!("usage: frc [LEN ...] [--m MIN] [--M MAX] [--delta D] [--k K]");
    eprintln!("       lengths may also be whitespace-separated on stdin");
}

fn parse_f64(flag: &str, raw: Option<String>) -> f64 {
    let raw = raw.unwrap_or_else(|| {
        eprintln!("missing value for {flag}");
        process::exit(2);
    });
    raw.parse().unwrap_or_else(|_| {
        eprintln!("invalid {flag}: {raw}");
        process::exit(2);
    })
}

fn parse_usize(flag: &str, raw: Option<String>) -> usize {
    let raw = raw.unwrap_or_else(|| {
        eprintln!("missing value for {flag}");
        process::exit(2);
    });
    raw.parse().unwrap_or_else(|_| {
        eprintln!("invalid {flag}: {raw}");
        process::exit(2);
    })
}

fn main() {
    let mut params = FrcParams::default();
    let mut lengths: Vec<usize> = Vec::new();
    let mut args = env::args().skip(1).peekable();
    if args.peek().map(|s| s.as_str()) == Some("-h")
        || args.peek().map(|s| s.as_str()) == Some("--help")
    {
        usage();
        process::exit(0);
    }
    while let Some(a) = args.next() {
        match a.as_str() {
            "--m" => params.m = parse_f64("--m", args.next()),
            "--M" => params.M = parse_f64("--M", args.next()),
            "--delta" => params.delta = parse_f64("--delta", args.next()),
            "--k" => params.k = parse_usize("--k", args.next()),
            "--" => {
                for rest in args.by_ref() {
                    match rest.parse::<usize>() {
                        Ok(n) => lengths.push(n),
                        Err(_) => {
                            eprintln!("invalid length: {rest}");
                            process::exit(2);
                        }
                    }
                }
            }
            _ => match a.parse::<usize>() {
                Ok(n) => lengths.push(n),
                Err(_) => {
                    eprintln!("unrecognized argument: {a}");
                    usage();
                    process::exit(2);
                }
            },
        }
    }

    if lengths.is_empty() {
        let mut buf = String::new();
        if io::stdin().read_to_string(&mut buf).is_ok() {
            for tok in buf.split_whitespace() {
                match tok.parse::<usize>() {
                    Ok(n) => lengths.push(n),
                    Err(_) => {
                        eprintln!("invalid length on stdin: {tok}");
                        process::exit(2);
                    }
                }
            }
        }
    }

    if lengths.is_empty() {
        usage();
        process::exit(2);
    }

    match pack(&lengths, &params) {
        Ok(res) => {
            for c in &res.chunks {
                println!("{} {}", c.start, c.end);
            }
        }
        Err(err) => {
            eprintln!("{err}");
            process::exit(1);
        }
    }
}
