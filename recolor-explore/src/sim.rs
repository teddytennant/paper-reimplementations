//! Adversary-branching simulator for oblivious recolorable exploration.

use crate::ac4::{ac4, ColorC4};
use crate::atc3::{atc3, ColorTC3};
use crate::graph::Graph;
use std::fmt;

/// Destination after recoloring: stop, or request a neighbor of this color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dest<C> {
    Stop,
    Neighbor(C),
}

/// Observation that has no listed rule. Unreachable on a correct execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unreachable;

/// Why a maximal execution failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fail {
    UnreachableObservation {
        pos: usize,
        step: usize,
    },
    AbsentColor {
        pos: usize,
        step: usize,
    },
    StepBound {
        pos: usize,
        step: usize,
    },
    StopNotAtStart {
        pos: usize,
        step: usize,
    },
    StopUnvisited {
        pos: usize,
        step: usize,
        missing: Vec<usize>,
    },
}

impl fmt::Display for Fail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fail::UnreachableObservation { pos, step } => {
                write!(f, "unreachable observation at vertex {pos} on step {step}")
            }
            Fail::AbsentColor { pos, step } => {
                write!(f, "requested absent color at vertex {pos} on step {step}")
            }
            Fail::StepBound { pos, step } => {
                write!(f, "step bound exceeded at vertex {pos} on step {step}")
            }
            Fail::StopNotAtStart { pos, step } => {
                write!(f, "stopped at {pos} (not start) on step {step}")
            }
            Fail::StopUnvisited { pos, step, missing } => write!(
                f,
                "stopped at {pos} on step {step} with unvisited {missing:?}"
            ),
        }
    }
}

/// Explore `g` from `start` against every adversary.
///
/// On success, returns the maximum number of rule applications over all
/// adversarial branches. Every branch must visit every vertex, return to
/// `start`, and stop there, within `max_steps` rule applications.
pub fn explore<C, F>(
    g: &Graph,
    start: usize,
    init: C,
    mut rule: F,
    max_steps: usize,
) -> Result<usize, Fail>
where
    C: Copy + Eq,
    F: FnMut(C, &[C]) -> Result<(C, Dest<C>), Unreachable>,
{
    let n = g.n();
    let mut colors = vec![init; n];
    let mut visited = vec![false; n];
    visited[start] = true;
    let mut max_seen = 0usize;
    rec(
        g,
        start,
        start,
        &mut colors,
        &mut visited,
        1,
        0,
        max_steps,
        &mut rule,
        &mut max_seen,
        &mut Vec::new(),
    )?;
    Ok(max_seen)
}

fn rec<C, F>(
    g: &Graph,
    start: usize,
    pos: usize,
    colors: &mut [C],
    visited: &mut [bool],
    visited_count: usize,
    steps: usize,
    max_steps: usize,
    rule: &mut F,
    max_seen: &mut usize,
    scratch: &mut Vec<C>,
) -> Result<(), Fail>
where
    C: Copy + Eq,
    F: FnMut(C, &[C]) -> Result<(C, Dest<C>), Unreachable>,
{
    if steps >= max_steps {
        return Err(Fail::StepBound { pos, step: steps });
    }
    scratch.clear();
    for &u in g.neighbors(pos) {
        scratch.push(colors[u]);
    }
    let (new_c, dest) = match rule(colors[pos], scratch) {
        Ok(x) => x,
        Err(Unreachable) => {
            return Err(Fail::UnreachableObservation {
                pos,
                step: steps + 1,
            });
        }
    };
    colors[pos] = new_c;
    let steps = steps + 1;
    match dest {
        Dest::Stop => {
            if pos != start {
                return Err(Fail::StopNotAtStart { pos, step: steps });
            }
            if visited_count != g.n() {
                let missing: Vec<usize> = (0..g.n()).filter(|&v| !visited[v]).collect();
                return Err(Fail::StopUnvisited {
                    pos,
                    step: steps,
                    missing,
                });
            }
            *max_seen = (*max_seen).max(steps);
            Ok(())
        }
        Dest::Neighbor(d) => {
            let cands: Vec<usize> = g
                .neighbors(pos)
                .iter()
                .copied()
                .filter(|&u| colors[u] == d)
                .collect();
            if cands.is_empty() {
                return Err(Fail::AbsentColor { pos, step: steps });
            }
            if cands.len() == 1 {
                let u = cands[0];
                let was = visited[u];
                if !was {
                    visited[u] = true;
                }
                rec(
                    g,
                    start,
                    u,
                    colors,
                    visited,
                    visited_count + usize::from(!was),
                    steps,
                    max_steps,
                    rule,
                    max_seen,
                    scratch,
                )
            } else {
                let color_snap = colors.to_vec();
                let vis_snap = visited.to_vec();
                for &u in &cands {
                    colors.copy_from_slice(&color_snap);
                    visited.copy_from_slice(&vis_snap);
                    let was = visited[u];
                    if !was {
                        visited[u] = true;
                    }
                    rec(
                        g,
                        start,
                        u,
                        colors,
                        visited,
                        visited_count + usize::from(!was),
                        steps,
                        max_steps,
                        rule,
                        max_seen,
                        scratch,
                    )?;
                }
                Ok(())
            }
        }
    }
}

/// ATC3 from `start`.
pub fn explore_atc3(g: &Graph, start: usize, max_steps: usize) -> Result<usize, Fail> {
    explore(
        g,
        start,
        ColorTC3::Init,
        |c, m| atc3(c, m).map(|(nc, d, _)| (nc, d)),
        max_steps,
    )
}

/// AC4 from `start`.
pub fn explore_ac4(g: &Graph, start: usize, max_steps: usize) -> Result<usize, Fail> {
    explore(
        g,
        start,
        ColorC4::Init,
        |c, m| ac4(c, m).map(|(nc, d, _)| (nc, d)),
        max_steps,
    )
}

/// ATC3 from every start. Returns the max rule-count over starts and branches.
pub fn explore_atc3_all_starts(g: &Graph, max_steps: usize) -> Result<usize, Fail> {
    let mut mx = 0;
    for s in 0..g.n() {
        mx = mx.max(explore_atc3(g, s, max_steps)?);
    }
    Ok(mx)
}

/// AC4 from every start.
pub fn explore_ac4_all_starts(g: &Graph, max_steps: usize) -> Result<usize, Fail> {
    let mut mx = 0;
    for s in 0..g.n() {
        mx = mx.max(explore_ac4(g, s, max_steps)?);
    }
    Ok(mx)
}
