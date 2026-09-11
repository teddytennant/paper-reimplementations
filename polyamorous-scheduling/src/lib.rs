//! Deadline-driven 4G* algorithm for the Polyamorous Scheduling Problem
//! (Agarwal, arXiv:2609.10641).
//!
//! The paper has no numbered equations. Formulas are cited by section,
//! Definition 1, Lemma 1, and Theorem 1.

mod graph;
mod schedule;

pub use graph::{Edge, Graph, GraphError};
pub use schedule::{
    deadline, deadline_driven_matching, deadline_driven_schedule, is_matching,
    max_weighted_waiting_time, paper_heat, Schedule, ScheduleError, DEFAULT_MAX_DAYS,
};
