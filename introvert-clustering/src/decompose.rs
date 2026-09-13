//! Lemma 2 introvert clustering and Theorem 3 layered decomposition.

use rand::Rng;

use crate::graph::{validate_beta, validate_eps, Clustering, Error, Graph};
use crate::mpx::mpx_cluster;
use crate::trim::trim;

/// One layer of a Definition 2 decomposition.
#[derive(Clone, Debug)]
pub struct Layer {
    pub clustering: Clustering,
    /// Intra-cluster edges of C_i in G_i (the layer assignment E_i).
    pub intra_edges: Vec<(usize, usize)>,
    /// Edges of G_i at the start of this layer (E_i ∪ ··· ∪ leftover).
    pub remaining: Vec<(usize, usize)>,
}

/// Theorem 3 output. `leftover` is E(G_{ℓ+1}), empty when every edge was assigned.
#[derive(Clone, Debug)]
pub struct LayeredDecomp {
    pub layers: Vec<Layer>,
    pub leftover: Vec<(usize, usize)>,
}

/// Lemma 2: MPX with λ = 2εβ, then Lemma 1 trim.
pub fn introvert_cluster<R: Rng>(
    g: &Graph,
    eps: f64,
    beta: f64,
    rng: &mut R,
) -> Result<Clustering, Error> {
    validate_eps(eps)?;
    validate_beta(beta)?;
    // Lemma 2: lambda = 2 * eps * beta, so E[U_0] <= 2 eps beta |E| and E[U] <= beta |E| after trim.
    let lambda = 2.0 * eps * beta;
    let mpx = mpx_cluster(g, lambda, rng)?;
    trim(g, &mpx, eps)
}

/// Theorem 3. Start from G_1 = G. At layer i apply Lemma 2 to G_i, set E_i to
/// the intra-cluster edges of C_i in G_i, and continue on the inter-cluster
/// remainder. At most `max_layers` layers; stop early if no edges remain, or
/// if a layer makes no progress (remainder goes to `leftover`).
pub fn layered_decompose<R: Rng>(
    g: &Graph,
    eps: f64,
    beta: f64,
    max_layers: usize,
    rng: &mut R,
) -> Result<LayeredDecomp, Error> {
    validate_eps(eps)?;
    validate_beta(beta)?;
    let mut remaining: Vec<(usize, usize)> = g.edges().to_vec();
    let mut layers = Vec::new();
    for _ in 0..max_layers {
        if remaining.is_empty() {
            break;
        }
        let gi = Graph::from_edges(g.n(), &remaining)?;
        let clustering = introvert_cluster(&gi, eps, beta, rng)?;
        let mut intra_edges = Vec::new();
        let mut next = Vec::new();
        for &(u, v) in &remaining {
            if clustering.is_intra_cluster(u, v) {
                intra_edges.push((u, v));
            } else {
                next.push((u, v));
            }
        }
        let no_progress = intra_edges.is_empty();
        layers.push(Layer {
            clustering,
            intra_edges,
            remaining: remaining.clone(),
        });
        remaining = next;
        if no_progress {
            break;
        }
    }
    Ok(LayeredDecomp {
        layers,
        leftover: remaining,
    })
}
