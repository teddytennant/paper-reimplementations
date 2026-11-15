Deterministic relative approximation of TV distance between mixtures of product distributions by zonotope compression (Fu, arXiv:2609.24235).

Run the tests and the demo with:

    cargo test
    cargo run --release

Markov-chain mixtures and latent-tree Ising are not included. Arithmetic is f64. The numerical rank cutoff is 1e-10 times the largest eigenvalue of G^T G. Net points equal within 1e-12 are deduplicated. The positioning map uses an SVD basis of the generator span (the paper allows any linear isomorphism). Lemma 3.4's displayed sqrt(2r) ball is not what the alpha=2 construction proves, and this code follows the construction and the delta formula in the proof. If an eigenvalue of a Lewis Gram matrix is non-positive from roundoff, it is replaced by 1e-14 times the largest eigenvalue before the inverse or inverse square root is formed. That replacement is not applied when every eigenvalue is positive.
