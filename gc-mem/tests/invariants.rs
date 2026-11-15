//! Paper invariants: Eq. 4, Theorems 3.4, 3.5, 4.1, STM union, multi-factor crash.

use gc_mem::{
    count_vote, expected_valid_count, hypergeometric_draw, monte_carlo_mean_valid,
    multi_factor_score, proportional_attention, scored_dominance, temporal_dominance,
    timestamp_rerank, union_dedup, Chunk, CountingOracle, ExactFactOracle, Fact, MemoryStream,
    MultiFactorWeights, Retrieved, Stm, XorShift64,
};

const QUERY: &str = "alice";

fn fact(attr: &str, value: &str) -> Fact {
    Fact::new("alice", attr, value)
}

fn item(id: u64, attr: &str, value: &str, tau: u64) -> Retrieved<Fact> {
    Retrieved::new(Chunk::new(id, fact(attr, value), tau), 1.0)
}

fn ids(rk: &[Retrieved<Fact>]) -> Vec<u64> {
    let mut v: Vec<u64> = rk.iter().map(|r| r.chunk.id).collect();
    v.sort_unstable();
    v
}

fn values_for<'a>(rk: &'a [Retrieved<Fact>], attr: &'a str) -> Vec<&'a str> {
    rk.iter()
        .filter(|r| r.chunk.content.attribute == attr)
        .map(|r| r.chunk.content.value.as_str())
        .collect()
}

/// Eq. 4: one new-state chunk plus older contradictions → only the new state
/// (and any non-contradicting history) survive Φ_T.
#[test]
fn eq4_new_state_excises_old_contradictions() {
    let oracle = ExactFactOracle;
    let mut rk = vec![
        item(1, "job", "engineer", 1),
        item(2, "job", "engineer", 2),
        item(3, "job", "engineer", 3),
        item(4, "job", "engineer", 4),
        item(5, "job", "chef", 10),
        item(6, "diet", "vegan", 0),
    ];
    let kept = temporal_dominance(&rk, &oracle, QUERY);
    let jobs = values_for(&kept, "job");
    assert_eq!(jobs, vec!["chef"]);
    let diets = values_for(&kept, "diet");
    assert_eq!(diets, vec!["vegan"]);
    assert_eq!(ids(&kept), vec![5, 6]);

    rk.pop();
    let only_job = temporal_dominance(&rk, &oracle, QUERY);
    assert_eq!(ids(&only_job), vec![5]);
}

/// Theorem 4.1: if S_new ∈ R_k, no older κ-contradictory chunk survives.
/// If S_new is absent, older chunks remain (do not invent a valid state).
#[test]
fn thm41_state_recovery() {
    let oracle = ExactFactOracle;
    let with_new = vec![
        item(1, "job", "engineer", 1),
        item(2, "job", "engineer", 5),
        item(3, "job", "chef", 9),
    ];
    let kept = temporal_dominance(&with_new, &oracle, QUERY);
    assert_eq!(ids(&kept), vec![3]);
    assert!(kept.iter().all(|r| r.chunk.content.value == "chef"));
    assert!(kept.iter().all(|r| r.chunk.content.value != "engineer"));

    let without_new = vec![item(1, "job", "engineer", 1), item(2, "job", "teacher", 5)];
    let leftover = temporal_dominance(&without_new, &oracle, QUERY);
    assert!(!leftover.is_empty(), "must not invent S_new");
    assert!(
        leftover.iter().all(|r| r.chunk.content.value != "chef"),
        "absent S_new is not fabricated"
    );
    assert_eq!(ids(&leftover), vec![2]);
    assert_eq!(leftover[0].chunk.content.value, "teacher");
}

/// Non-contradicting long-term facts survive Φ_T even if old.
/// Timestamp re-rank drops them; GC-Mem keeps them.
#[test]
fn non_contradicting_history_survives() {
    let oracle = ExactFactOracle;
    let rk = vec![
        item(1, "diet", "vegan", 1),
        item(2, "job", "engineer", 10),
        item(3, "job", "engineer", 11),
        item(4, "job", "engineer", 12),
        item(5, "job", "engineer", 13),
        item(6, "job", "chef", 20),
    ];

    let gc = temporal_dominance(&rk, &oracle, QUERY);
    assert!(
        gc.iter()
            .any(|r| r.chunk.content.attribute == "diet" && r.chunk.content.value == "vegan"),
        "GC-Mem must keep the old non-contradicting diet fact"
    );
    assert!(gc
        .iter()
        .any(|r| r.chunk.content.attribute == "job" && r.chunk.content.value == "chef"));
    assert!(!gc
        .iter()
        .any(|r| r.chunk.content.attribute == "job" && r.chunk.content.value == "engineer"));

    let reranked = timestamp_rerank(&rk, 3);
    assert!(
        !reranked.iter().any(|r| r.chunk.content.attribute == "diet"),
        "timestamp re-rank drops the old valid diet fact"
    );
    assert!(reranked.iter().any(|r| r.chunk.id == 6));
}

/// Chained mutations A→B→C: Φ_T collapses to terminal C when C is present (Def 3.1).
#[test]
fn chained_mutations_collapse_to_terminal() {
    let oracle = ExactFactOracle;
    let rk = vec![
        item(1, "status", "A", 1),
        item(2, "status", "B", 2),
        item(3, "status", "C", 3),
    ];
    let kept = temporal_dominance(&rk, &oracle, QUERY);
    assert_eq!(ids(&kept), vec![3]);
    assert_eq!(kept[0].chunk.content.value, "C");

    let without_c = vec![item(1, "status", "A", 1), item(2, "status", "B", 2)];
    let mid = temporal_dominance(&without_c, &oracle, QUERY);
    assert_eq!(ids(&mid), vec![2]);
    assert_eq!(mid[0].chunk.content.value, "B");
}

/// Theorem 3.4 closed form E[X] = k M / (N+M). Monte Carlo mean is close.
/// E[X] → 0 as N → ∞ for fixed k, M.
#[test]
fn thm34_closed_form_and_monte_carlo() {
    let k = 10;
    let n = 90;
    let m = 10;
    let closed = expected_valid_count(k, n, m);
    assert!((closed - k as f64 * m as f64 / (n + m) as f64).abs() < 1e-12);

    let mut rng = XorShift64::seed(0xC0FFEE);
    let mc = monte_carlo_mean_valid(k, n, m, 20_000, &mut rng);
    assert!(
        (mc - closed).abs() < 0.05,
        "MC mean {mc} vs closed form {closed}"
    );

    let mut prev = f64::INFINITY;
    for n_stale in [10usize, 100, 1_000, 10_000, 1_000_000] {
        let e = expected_valid_count(5, n_stale, 2);
        assert!(e < prev, "E[X] must decrease as N grows");
        prev = e;
    }
    let tail = expected_valid_count(5, 1_000_000, 2);
    assert!(tail < 1e-4, "E[X] → 0 as N → ∞, got {tail}");

    let mut rng = XorShift64::seed(7);
    let mut x_sum = 0usize;
    for _ in 0..2_000 {
        x_sum += hypergeometric_draw(5, 95, 5, &mut rng);
    }
    let mean = x_sum as f64 / 2_000.0;
    let expect = expected_valid_count(5, 95, 5);
    assert!((mean - expect).abs() < 0.08, "{mean} vs {expect}");
}

/// Theorem 3.5: uniform/count vote over 1 valid + k-1 stale yields P(stale)=(k-1)/k.
/// After Φ_T, P(new) → 1 if the valid chunk was retrieved.
#[test]
fn thm35_majority_vote_trap() {
    let k = 8;
    let mut rk = Vec::new();
    for i in 0..(k - 1) {
        rk.push(item(i as u64, "job", "engineer", i as u64 + 1));
    }
    rk.push(item(k as u64, "job", "chef", k as u64 + 10));

    let mass = proportional_attention(&rk, |c: &Fact| c.value.clone());
    let p_stale = mass.get("engineer").copied().unwrap_or(0.0);
    let p_new = mass.get("chef").copied().unwrap_or(0.0);
    let expect_stale = (k - 1) as f64 / k as f64;
    assert!((p_stale - expect_stale).abs() < 1e-12);
    assert!((p_new - 1.0 / k as f64).abs() < 1e-12);

    let vote = count_vote(&rk, |c: &Fact| c.value.clone());
    assert_eq!(vote.as_deref(), Some("engineer"));

    let oracle = ExactFactOracle;
    let gc = temporal_dominance(&rk, &oracle, QUERY);
    let mass_gc = proportional_attention(&gc, |c: &Fact| c.value.clone());
    let p_new_gc = mass_gc.get("chef").copied().unwrap_or(0.0);
    assert!(
        (p_new_gc - 1.0).abs() < 1e-12,
        "after Φ_T, P(new) → 1, got {p_new_gc}"
    );
    assert!(!mass_gc.contains_key("engineer"));
    assert_eq!(
        count_vote(&gc, |c: &Fact| c.value.clone()).as_deref(),
        Some("chef")
    );
}

/// Multi-factor scoring with usage/stability that grow with age prefers stale
/// chunks and can retain S_old, violating Theorem 4.1. Pure Φ_T does not.
#[test]
fn multi_factor_violates_thm41() {
    let oracle = ExactFactOracle;
    let now = 100;
    let old = item(1, "job", "engineer", 1).with_usage_stability(50.0, 50.0);
    let new = item(2, "job", "chef", 100).with_usage_stability(1.0, 1.0);
    let mut old = old;
    let mut new = new;
    old.relevance = 0.95;
    new.relevance = 0.95;
    let rk = vec![old, new];

    let phi = temporal_dominance(&rk, &oracle, QUERY);
    assert_eq!(ids(&phi), vec![2], "pure Φ_T keeps S_new");
    assert_eq!(phi[0].chunk.content.value, "chef");

    let crashed = scored_dominance(&rk, &oracle, QUERY, now, &MultiFactorWeights::full());
    assert!(
        crashed.iter().any(|r| r.chunk.content.value == "engineer"),
        "usage/stability retain S_old"
    );
    assert!(
        !crashed.iter().any(|r| r.chunk.content.value == "chef"),
        "S_new is dominated by the high-usage stale chunk"
    );

    let ts_only = scored_dominance(
        &rk,
        &oracle,
        QUERY,
        now,
        &MultiFactorWeights::timestamp_only(),
    );
    assert_eq!(ids(&ts_only), vec![2]);

    assert!(
        multi_factor_score(&rk[0], now, &MultiFactorWeights::full())
            > multi_factor_score(&rk[1], now, &MultiFactorWeights::full())
    );
}

/// STM: recent-N FIFO unioned with top-k, duplicates removed before Φ_T.
#[test]
fn stm_union_dedup_before_phi() {
    let mut stm = Stm::new(3);
    let c1 = Chunk::new(1, fact("job", "engineer"), 1);
    let c2 = Chunk::new(2, fact("hobby", "chess"), 2);
    let c3 = Chunk::new(3, fact("job", "engineer"), 3);
    let c4 = Chunk::new(4, fact("diet", "vegan"), 4);
    let c5 = Chunk::new(5, fact("job", "chef"), 5);
    for c in [c1.clone(), c2.clone(), c3.clone(), c4.clone(), c5.clone()] {
        stm.push(c);
    }
    let stm_ids: Vec<u64> = stm.iter().map(|c| c.id).collect();
    assert_eq!(stm_ids, vec![3, 4, 5], "FIFO keeps last N=3");

    let mut bag = MemoryStream::new();
    for c in [c1, c2, c3, c4.clone(), c5.clone()] {
        bag.append(c);
    }
    let mtm = bag.top_k(2, |f: &Fact| if f.attribute == "job" { 1.0 } else { 0.0 });
    assert_eq!(mtm.len(), 2);
    assert!(mtm.iter().any(|r| r.chunk.id == 5));

    let rk = union_dedup(&mtm, &stm.chunks());
    let mut uniq = ids(&rk);
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(uniq.len(), rk.len(), "duplicates removed");
    assert!(
        rk.iter().any(|r| r.chunk.id == 4),
        "STM-only diet is unioned"
    );
    assert!(rk.iter().any(|r| r.chunk.id == 5));
    assert_eq!(
        rk.iter().filter(|r| r.chunk.id == 5).count(),
        1,
        "id 5 from MTM and STM is kept once"
    );

    let oracle = ExactFactOracle;
    let kept = temporal_dominance(&rk, &oracle, QUERY);
    assert!(kept
        .iter()
        .any(|r| r.chunk.content.attribute == "job" && r.chunk.content.value == "chef"));
    assert!(!kept
        .iter()
        .any(|r| r.chunk.content.attribute == "job" && r.chunk.content.value == "engineer"));
    assert!(kept.iter().any(|r| r.chunk.content.attribute == "diet"));
}

/// κ is pairwise O(|R_k|^2): oracle calls ≤ |R_k|(|R_k|-1) (or ≤ |R_k|^2).
#[test]
fn kappa_pairwise_quadratic() {
    let inner = ExactFactOracle;
    let oracle = CountingOracle::new(inner);
    let n = 7;
    let mut rk = Vec::new();
    for i in 0..n {
        rk.push(item(i as u64, "job", "engineer", i as u64));
    }
    rk.push(item(n as u64, "hobby", "chess", n as u64));
    let m = rk.len();
    let _ = temporal_dominance(&rk, &oracle, QUERY);
    let calls = oracle.calls();
    let n2 = (m * m) as u64;
    let n_nm1 = (m * (m - 1)) as u64;
    assert!(
        calls <= n_nm1,
        "calls {calls} exceed |R_k|(|R_k|-1) = {n_nm1}"
    );
    assert!(calls <= n2);
    assert!(
        calls >= n_nm1 / 4,
        "expected a quadratic number of pairwise checks, got {calls} for |R_k|={m}"
    );
}

/// Bag retriever is an in-memory rank, not a dense model.
#[test]
fn memory_stream_topk_is_a_bag() {
    let mut bag = MemoryStream::new();
    bag.append(Chunk::new(1, fact("job", "engineer"), 1));
    bag.append(Chunk::new(2, fact("diet", "vegan"), 2));
    bag.append(Chunk::new(3, fact("job", "chef"), 3));
    let top = bag.top_k(1, |f: &Fact| if f.attribute == "diet" { 2.0 } else { 0.1 });
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].chunk.id, 2);
    assert_eq!(bag.len(), 3);
}
