Rust implementation of the 4G* deadline-driven algorithm for Polyamorous Scheduling (https://arxiv.org/abs/2609.10641).

```
cargo test
cargo run --bin schedule -- 3 0-1:1.0 0-2:0.5 1-2:0.5
```

The paper specifies an infinite online rule (Definition 1), not a periodic construction. This crate returns the first recurrent cycle of that process. The source has no equation numbers; formulas are cited by section. Discrete heat is g_e times days since last service, matching the witness calculation in Section 2.
