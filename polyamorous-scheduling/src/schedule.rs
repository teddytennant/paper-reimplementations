//! Deadline-driven matching and periodic schedule construction.
//!
//! Section 2 / Definition 1, unnormalized so that G* need not be 1:
//!   eligible when h_e(t) ≥ G*,
//!   D_t(e) = t + (2 G* − h_e(t)) / g_e.
//! With G* = 1 this is exactly the displayed deadline in Section 2.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

use crate::graph::Graph;

pub const DEFAULT_MAX_DAYS: usize = 100_000;

#[derive(Clone, Debug)]
pub struct Schedule {
    /// Edge-index matchings, one per day of the repeating cycle.
    pub matchings: Vec<Vec<usize>>,
    /// Days discarded before the age vector first repeats.
    pub transient: usize,
    /// Supremum of h_e(t) over the simulated horizon (paper heat).
    pub max_heat: f64,
}

impl Schedule {
    pub fn period(&self) -> usize {
        self.matchings.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    EmptyCycle,
    NoCycle { max_days: usize },
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScheduleError::EmptyCycle => write!(f, "cycle detection produced an empty period"),
            ScheduleError::NoCycle { max_days } => {
                write!(f, "no recurrent age vector within {max_days} days")
            }
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Section 2, unnormalized: D_t(e) = t + (2 G* − h_e(t)) / g_e.
///
/// Normalized form in the paper (G* = 1): D_t(e) = t + (2 − h_e(t)) / g_e.
pub fn deadline(t: f64, heat: f64, g: f64, g_star: f64) -> f64 {
    t + (2.0 * g_star - heat) / g
}

/// Definition 1: greedy matching of eligible edges in nondecreasing deadline
/// order, ties broken by edge index (a fixed total order).
pub fn deadline_driven_matching(graph: &Graph, heats: &[f64], t: f64, g_star: f64) -> Vec<usize> {
    debug_assert_eq!(heats.len(), graph.m());
    let mut eligible: Vec<usize> = (0..graph.m())
        .filter(|&e| heats[e] + 1e-12 >= g_star && g_star > 0.0)
        .collect();
    eligible.sort_by(|&a, &b| {
        let da = deadline(t, heats[a], graph.edges()[a].g, g_star);
        let db = deadline(t, heats[b], graph.edges()[b].g, g_star);
        match da.partial_cmp(&db).unwrap_or(Ordering::Equal) {
            Ordering::Equal => a.cmp(&b),
            other => other,
        }
    });
    let mut used = vec![false; graph.n()];
    let mut matching = Vec::new();
    for e in eligible {
        let edge = &graph.edges()[e];
        if !used[edge.u] && !used[edge.v] {
            used[edge.u] = true;
            used[edge.v] = true;
            matching.push(e);
        }
    }
    matching
}

/// Run Definition 1 from zero heat until the integer age vector repeats.
///
/// Heat at day t of an edge last served at time s is g_e · (t − s), which is
/// the quantity used in the witness calculation in Section 2
/// (h_{e_0}(s) = g_e s when last service is the fictitious time 0).
/// The paper writes h_e(t+1) = 0 after service; that is the reset at the
/// service instant. The next decision sees heat g_e.
pub fn deadline_driven_schedule(graph: &Graph, max_days: usize) -> Result<Schedule, ScheduleError> {
    let m = graph.m();
    if m == 0 {
        return Ok(Schedule {
            matchings: vec![Vec::new()],
            transient: 0,
            max_heat: 0.0,
        });
    }
    let g_star = graph.g_star();
    let mut last_service = vec![0usize; m];
    let mut history: Vec<Vec<usize>> = Vec::with_capacity(m.max(8));
    let mut seen: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut max_heat = 0.0;

    for t in 0..max_days {
        let ages: Vec<u32> = last_service.iter().map(|&s| (t - s) as u32).collect();
        if let Some(&t0) = seen.get(&ages) {
            if t0 == t {
                return Err(ScheduleError::EmptyCycle);
            }
            return Ok(Schedule {
                matchings: history[t0..t].to_vec(),
                transient: t0,
                max_heat,
            });
        }
        seen.insert(ages, t);

        let heats: Vec<f64> = (0..m)
            .map(|e| graph.edges()[e].g * (t - last_service[e]) as f64)
            .collect();
        for &h in &heats {
            if h > max_heat {
                max_heat = h;
            }
        }

        let matching = deadline_driven_matching(graph, &heats, t as f64, g_star);
        for &e in &matching {
            last_service[e] = t;
        }
        history.push(matching);
    }

    Err(ScheduleError::NoCycle { max_days })
}

/// True iff `edges` is a matching: no two share a vertex.
pub fn is_matching(graph: &Graph, edges: &[usize]) -> bool {
    let mut used = vec![false; graph.n()];
    for &e in edges {
        if e >= graph.m() {
            return false;
        }
        let edge = &graph.edges()[e];
        if used[edge.u] || used[edge.v] {
            return false;
        }
        used[edge.u] = true;
        used[edge.v] = true;
    }
    true
}

/// OPS objective of Gąsieniec–Smith–Wild: max_e g_e · r_S(e), where r_S(e) is
/// the maximum circular gap between consecutive occurrences of e.
/// Returns +∞ if some edge never appears.
pub fn max_weighted_waiting_time(graph: &Graph, matchings: &[Vec<usize>]) -> f64 {
    let p = matchings.len();
    if p == 0 {
        return if graph.m() == 0 { 0.0 } else { f64::INFINITY };
    }
    let mut appearances: Vec<Vec<usize>> = vec![Vec::new(); graph.m()];
    for (t, matching) in matchings.iter().enumerate() {
        for &e in matching {
            appearances[e].push(t);
        }
    }
    let mut cost: f64 = 0.0;
    for (e, times) in appearances.iter().enumerate() {
        if times.is_empty() {
            return f64::INFINITY;
        }
        let mut gap = 0usize;
        for w in times.windows(2) {
            gap = gap.max(w[1] - w[0]);
        }
        let wrap = times[0] + p - times[times.len() - 1];
        gap = gap.max(wrap);
        cost = cost.max(graph.edges()[e].g * gap as f64);
    }
    cost
}

/// Paper heat of a periodic schedule: same as weighted waiting time, because
/// h_e at a decision equals g_e times days since last service.
pub fn paper_heat(graph: &Graph, matchings: &[Vec<usize>]) -> f64 {
    max_weighted_waiting_time(graph, matchings)
}
