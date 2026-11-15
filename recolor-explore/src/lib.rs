//! Recolorable graph exploration in the BFUW model (arXiv:2609.14356).
//!
//! Implements Algorithm 1 (ATC3, three colors on trees and simple cycles)
//! and Algorithm 2 (AC4, four colors on graphs whose blocks are cycles or
//! complete bipartite graphs). The agent is oblivious: each step observes
//! the current color and the multiset of neighbor colors, recolors the
//! current vertex, and either stops or requests a neighbor color. An
//! adversary chooses the destination among neighbors of that color.

mod ac4;
mod atc3;
mod graph;
mod sim;

pub use ac4::{ac4, ColorC4, RuleC4};
pub use atc3::{atc3, ColorTC3, RuleTC3};
pub use graph::Graph;
pub use sim::{
    explore, explore_ac4, explore_ac4_all_starts, explore_atc3, explore_atc3_all_starts, Dest,
    Fail, Unreachable,
};
