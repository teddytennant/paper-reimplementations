//! Degree-adaptive clustering from "Beating Quadratic Time-Message Trade-off in
//! Distributed Minimum Spanning Tree Construction" (Izumi, Kitamura,
//! Masuzawa, arXiv:2608.30299), on top of a simulated CONGEST-KT1 network.
//!
//! [`dcluster`] is Algorithm 1 of Section 4.1, line for line, running on the
//! [`congest::Network`] simulator, which counts rounds and messages and rejects
//! any message too wide for the model. [`partition`] holds the pieces of
//! Sections 3.1 and 4.3 the paper writes out: the two conditions of
//! Definition 3.1, step 2 of Algorithm Partition, and the `i`-moderate and
//! admissibility tests.
//!
//! Every random choice comes from a seeded [`rng::SplitMix64`] stream, so a run
//! is reproducible from `(graph, seed)` down to the message count.

pub mod congest;
pub mod dcluster;
pub mod graph;
pub mod partition;
pub mod rng;

pub use dcluster::{dcluster, Cluster, Clustering};
pub use graph::Graph;
