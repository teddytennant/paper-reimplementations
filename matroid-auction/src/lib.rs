//! Auction algorithm for weighted matroid intersection.
//!
//! `auction` is Algorithm 1. `approximate` rescales the caller epsilon by 1/5.

mod auction;
mod matroid;

pub use auction::{
    approximate, auction, ceil_lambda, min_pi_circuit_element, num_phases, AuctionError, Solution,
};
pub use matroid::{IndependenceOracle, PartitionMatroid, TransversalMatroid, UniformMatroid};
