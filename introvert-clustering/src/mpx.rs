//! Miller-Peng-Xu low-diameter clustering (Theorem 7 / [MPX13]).

use rand::Rng;

use crate::graph::{validate_lambda, Clustering, Error, Graph};

/// Sample Exp(rate = `lambda`) via inverse CDF: -ln(U)/lambda with U ~ Unif(0, 1).
fn exp_sample<R: Rng>(rng: &mut R, lambda: f64) -> f64 {
    let u: f64 = rng.gen::<f64>().clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    -u.ln() / lambda
}

/// Theorem 7 / [MPX13]. Each vertex `u` draws an independent shift `delta_u ~ Exp(lambda)`
/// (rate lambda). Vertex `v` is assigned to argmin_u (dist(v, u) - delta_u), with ties
/// broken by smaller vertex id. This is the sequential all-pairs form of MPX:
/// E[number of inter-cluster edges] <= lambda |E|, and clusters have strong diameter
/// O(lambda^{-1} log n).
pub fn mpx_cluster<R: Rng>(g: &Graph, lambda: f64, rng: &mut R) -> Result<Clustering, Error> {
    validate_lambda(lambda)?;
    let n = g.n();
    if n == 0 {
        return Ok(Clustering::new(0));
    }

    let delta: Vec<f64> = (0..n).map(|_| exp_sample(rng, lambda)).collect();
    let dist: Vec<Vec<Option<usize>>> = (0..n).map(|s| g.bfs_dist(s)).collect();

    let mut assignment = vec![None; n];
    for v in 0..n {
        let mut best_u = v;
        let mut best_score = f64::INFINITY;
        for u in 0..n {
            let Some(d) = dist[v][u] else {
                continue;
            };
            let score = (d as f64) - delta[u];
            if score < best_score || (score == best_score && u < best_u) {
                best_score = score;
                best_u = u;
            }
        }
        assignment[v] = Some(best_u);
    }
    Ok(Clustering::from_assignment(assignment))
}
