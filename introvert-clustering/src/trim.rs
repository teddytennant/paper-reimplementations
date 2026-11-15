//! Lemma 1 trimming: drop clustered vertices that violate the introvert property.

use crate::graph::{din_in_cluster, validate_eps, Clustering, Error, Graph};

/// A vertex removed by Lemma 1, with din and deg at the moment of removal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemovedVertex {
    pub vertex: usize,
    pub din: usize,
    pub deg: usize,
}

#[derive(Clone, Debug)]
pub struct TrimTrace {
    pub clustering: Clustering,
    pub removed: Vec<RemovedVertex>,
}

/// Lemma 1. Repeatedly remove clustered vertices with
/// din < (1/2 - eps) deg. Removal order: sweep vertices in increasing id,
/// unclustering a violator as soon as it is seen; repeat sweeps until a
/// full pass removes nobody.
pub fn trim(g: &Graph, clustering: &Clustering, eps: f64) -> Result<Clustering, Error> {
    Ok(trim_with_trace(g, clustering, eps)?.clustering)
}

/// Lemma 1 with a trace of each removal (for Equation (4) checks).
pub fn trim_with_trace(
    g: &Graph,
    clustering: &Clustering,
    eps: f64,
) -> Result<TrimTrace, Error> {
    validate_eps(eps)?;
    let n = clustering.n().min(g.n());
    let mut c = clustering.clone();
    if clustering.n() > n {
        for v in n..clustering.n() {
            c.uncluster(v);
        }
    }
    let mut removed = Vec::new();
    loop {
        let mut changed = false;
        for v in 0..n {
            if c.cluster_of(v).is_none() {
                continue;
            }
            let deg = g.degree(v);
            let din = din_in_cluster(g, &c, v);
            // Lemma 1: remove if din < (1/2 - eps) deg_G(v).
            if (din as f64) < (0.5 - eps) * (deg as f64) {
                c.uncluster(v);
                removed.push(RemovedVertex {
                    vertex: v,
                    din,
                    deg,
                });
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(TrimTrace {
        clustering: c,
        removed,
    })
}
