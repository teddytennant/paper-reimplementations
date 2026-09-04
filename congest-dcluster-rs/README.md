# congest-dcluster-rs

Rust implementation of the degree-adaptive clustering (Algorithm 1, DCluster)
from "Beating Quadratic Time-Message Trade-off in Distributed Minimum Spanning
Tree Construction" (arXiv:2608.30299).

## build

cargo build --release

## run

cargo run --release -- --n 512

This is Section 4.1 on a simulated CONGEST-KT1 network, not the paper's MST algorithm.
Algorithm Partition is not here: steps 3 and 4 call MPX and OutDetect and Lemma 4.4
calls Split, none of which the paper writes out. What Section 4.3 does write out is here:
step 2's edge sampling, the i-moderate and (C1)/(C2) tests, and Definition 3.1 measured
on a partition. The round and message counts come out of the simulator, not out of a proof.
