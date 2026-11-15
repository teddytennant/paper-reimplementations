//! CPU reference for the TopoEP two-stage expert placement and token routing solver.
//!
//! Stage 1 places at most one cross-domain replica per expert per domain.
//! Tokens are then routed, preferring instances in the source domain.
//! Stage 2 moves load only inside a domain, from the current busiest rank.

#![forbid(unsafe_code)]

/// Two-stage TopoEP load-balancing instance.
///
/// `dom[r]` is the domain of rank `r`. `main[e]` is the fixed main rank of
/// expert `e`. `n_slot` is the number of non-main replica slots on each rank.
/// `omega[s][e]` is the token count from source rank `s` to expert `e`.
/// `cost[s][r]` is a communication-time tie-break (lower is better). The
/// diagonal is 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    pub n_ranks: usize,
    pub n_experts: usize,
    pub dom: Vec<usize>,
    pub main: Vec<usize>,
    pub n_slot: usize,
    pub w_bytes: Vec<u64>,
    pub s_tok: u64,
    pub omega: Vec<Vec<u64>>,
    pub cost: Vec<Vec<u64>>,
}

/// Placement and routing produced by [`solve`].
///
/// `x[e][r]` is true when expert `e` is hosted on rank `r`.
/// `q[s][e][r]` is the number of `(s, e)` assignments executed on rank `r`.
/// `rank_load[r]` is the sum of `q` over sources and experts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub x: Vec<Vec<bool>>,
    pub q: Vec<Vec<Vec<u64>>>,
    pub rank_load: Vec<u64>,
}

/// Solve inter-node placement, route tokens, then refine inside each domain.
///
/// Deterministic. Equal keys break toward the lower index.
///
/// The result keeps these constraints: token conservation (C1), assignments
/// only on hosts (C2), at most `n_slot` non-main instances per rank (C3),
/// a cross-domain replica only when that domain's benefit is positive (C4),
/// and every main instance present (C5).
pub fn solve(input: &Input) -> Plan {
    validate(input);
    let mut plan = place_and_route(input);
    refine_plan(input, &mut plan);
    debug_assert_eq!(plan.rank_load, loads_from_q(&plan.q));
    plan
}

fn validate(input: &Input) {
    assert_eq!(input.dom.len(), input.n_ranks, "dom length");
    assert_eq!(input.main.len(), input.n_experts, "main length");
    assert_eq!(input.w_bytes.len(), input.n_experts, "w_bytes length");
    assert_eq!(input.omega.len(), input.n_ranks, "omega rows");
    assert_eq!(input.cost.len(), input.n_ranks, "cost rows");
    for expert in 0..input.n_experts {
        assert!(input.main[expert] < input.n_ranks, "main rank out of range");
    }
    for source in 0..input.n_ranks {
        assert_eq!(input.omega[source].len(), input.n_experts, "omega cols");
        assert_eq!(input.cost[source].len(), input.n_ranks, "cost cols");
        assert_eq!(input.cost[source][source], 0, "cost diagonal must be 0");
    }
}

/// `2 * demand * s_tok - w_bytes`, computed in `i128` and narrowed to `i64`.
fn benefit(demand: u64, s_tok: u64, w_bytes: u64) -> i64 {
    let value = 2i128 * i128::from(demand) * i128::from(s_tok) - i128::from(w_bytes);
    i64::try_from(value).expect("benefit exceeds i64")
}

fn ranks_by_domain(dom: &[usize]) -> Vec<Vec<usize>> {
    let n_dom = dom.iter().copied().max().map(|d| d + 1).unwrap_or(0);
    let mut ranks = vec![Vec::new(); n_dom];
    for (rank, &domain) in dom.iter().enumerate() {
        ranks[domain].push(rank);
    }
    ranks
}

fn domain_demand(input: &Input, ranks_by_dom: &[Vec<usize>]) -> Vec<Vec<u64>> {
    let mut demand = vec![vec![0u64; input.n_experts]; ranks_by_dom.len()];
    for (domain, ranks) in ranks_by_dom.iter().enumerate() {
        for &source in ranks {
            for expert in 0..input.n_experts {
                demand[domain][expert] += input.omega[source][expert];
            }
        }
    }
    demand
}

fn place(input: &Input) -> Vec<Vec<bool>> {
    let mut x = vec![vec![false; input.n_ranks]; input.n_experts];
    for expert in 0..input.n_experts {
        x[expert][input.main[expert]] = true;
    }

    let mut occupied = vec![0usize; input.n_ranks];
    let ranks_by_dom = ranks_by_domain(&input.dom);
    let demand = domain_demand(input, &ranks_by_dom);

    for (domain, ranks) in ranks_by_dom.iter().enumerate() {
        if ranks.is_empty() {
            continue;
        }
        let mut candidates: Vec<(i64, usize)> = Vec::new();
        for expert in 0..input.n_experts {
            if input.dom[input.main[expert]] == domain {
                continue;
            }
            let gain = benefit(demand[domain][expert], input.s_tok, input.w_bytes[expert]);
            if gain > 0 {
                candidates.push((gain, expert));
            }
        }
        candidates.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));

        for &(_, expert) in &candidates {
            let rank = least_occupied(ranks, &occupied, input.n_slot);
            if let Some(rank) = rank {
                x[expert][rank] = true;
                occupied[rank] += 1;
            }
        }
    }
    x
}

/// Lowest occupied rank in `ranks` that still has a free non-main slot.
/// `ranks` must be in increasing rank order so an equal occupation keeps the
/// lower index.
fn least_occupied(ranks: &[usize], occupied: &[usize], n_slot: usize) -> Option<usize> {
    let mut best: Option<usize> = None;
    for &rank in ranks {
        if occupied[rank] >= n_slot {
            continue;
        }
        if best
            .map(|current| occupied[rank] < occupied[current])
            .unwrap_or(true)
        {
            best = Some(rank);
        }
    }
    best
}

fn route(input: &Input, x: &[Vec<bool>]) -> Vec<Vec<Vec<u64>>> {
    let n_ranks = input.n_ranks;
    let mut q = vec![vec![vec![0u64; n_ranks]; input.n_experts]; n_ranks];
    for source in 0..n_ranks {
        let home = input.dom[source];
        for expert in 0..input.n_experts {
            let tokens = input.omega[source][expert];
            if tokens == 0 {
                continue;
            }
            let mut local = Vec::new();
            let mut deployed = Vec::new();
            for rank in 0..n_ranks {
                if x[expert][rank] {
                    deployed.push(rank);
                    if input.dom[rank] == home {
                        local.push(rank);
                    }
                }
            }
            let chosen = if local.is_empty() { &deployed } else { &local };
            assert!(
                !chosen.is_empty(),
                "expert {expert} has no deployed instance"
            );
            distribute(tokens, chosen, &mut q[source][expert]);
        }
    }
    q
}

/// Split `tokens` as evenly as possible. The remainder goes to the earlier
/// entries of `ranks`, which are in increasing rank order.
fn distribute(tokens: u64, ranks: &[usize], dest: &mut [u64]) {
    let width = ranks.len() as u64;
    let base = tokens / width;
    let remainder = (tokens % width) as usize;
    for (index, &rank) in ranks.iter().enumerate() {
        dest[rank] = base + u64::from(index < remainder);
    }
}

fn loads_from_q(q: &[Vec<Vec<u64>>]) -> Vec<u64> {
    if q.is_empty() {
        return Vec::new();
    }
    let mut load = vec![0u64; q.len()];
    for by_source in q {
        for by_expert in by_source {
            for (rank, &amount) in by_expert.iter().enumerate() {
                load[rank] += amount;
            }
        }
    }
    load
}

fn place_and_route(input: &Input) -> Plan {
    let x = place(input);
    let q = route(input, &x);
    let rank_load = loads_from_q(&q);
    Plan { x, q, rank_load }
}

fn refine_plan(input: &Input, plan: &mut Plan) {
    let n_ranks = input.n_ranks;
    let n_experts = input.n_experts;
    let mut assigned = vec![vec![0u64; n_ranks]; n_experts];
    for source in 0..n_ranks {
        for expert in 0..n_experts {
            for rank in 0..n_ranks {
                assigned[expert][rank] += plan.q[source][expert][rank];
            }
        }
    }
    let mut slots_used = vec![0usize; n_ranks];
    for expert in 0..n_experts {
        for rank in 0..n_ranks {
            if plan.x[expert][rank] && input.main[expert] != rank {
                slots_used[rank] += 1;
            }
        }
    }

    let ranks_by_dom = ranks_by_domain(&input.dom);
    for ranks in &ranks_by_dom {
        refine_domain(input, plan, &mut assigned, &mut slots_used, ranks);
    }
}

/// Drain the busiest rank in one domain until it has no positive move.
///
/// Targets stay in the same domain, so a token's destination domain does not
/// change and no new cross-domain route is created. A replica is added only
/// in a domain that already hosts the expert, because the busiest rank already
/// executes some of that expert's assignments.
///
/// Each committed move strictly decreases the sum of squared rank loads, so
/// the loop terminates.
fn refine_domain(
    input: &Input,
    plan: &mut Plan,
    assigned: &mut [Vec<u64>],
    slots_used: &mut [usize],
    ranks: &[usize],
) {
    if ranks.len() < 2 {
        return;
    }
    let total: u64 = ranks.iter().map(|&rank| plan.rank_load[rank]).sum();
    let mut remaining_steps = total.saturating_mul(total).saturating_add(1);
    loop {
        assert!(
            remaining_steps > 0,
            "intra-node refinement did not terminate"
        );
        remaining_steps -= 1;

        let busy = busiest(ranks, &plan.rank_load);
        let Some(mv) = best_move(
            input,
            &plan.x,
            assigned,
            slots_used,
            &plan.rank_load,
            ranks,
            busy,
        ) else {
            break;
        };

        if !plan.x[mv.expert][mv.target] {
            debug_assert!(ranks.iter().any(|&rank| plan.x[mv.expert][rank]));
            debug_assert_ne!(input.main[mv.expert], mv.target);
            debug_assert!(slots_used[mv.target] < input.n_slot);
            plan.x[mv.expert][mv.target] = true;
            slots_used[mv.target] += 1;
        }
        debug_assert_eq!(input.dom[busy], input.dom[mv.target]);
        move_tokens(&mut plan.q, mv.expert, busy, mv.target, mv.delta);
        assigned[mv.expert][busy] -= mv.delta;
        assigned[mv.expert][mv.target] += mv.delta;
        plan.rank_load[busy] -= mv.delta;
        plan.rank_load[mv.target] += mv.delta;
    }
}

/// Highest load in `ranks`. `ranks` is increasing, so a strict `>` keeps the
/// lower index on a tie.
fn busiest(ranks: &[usize], rank_load: &[u64]) -> usize {
    let mut best = ranks[0];
    for &rank in &ranks[1..] {
        if rank_load[rank] > rank_load[best] {
            best = rank;
        }
    }
    best
}

struct Candidate {
    delta: u64,
    target_load: u64,
    cost: u64,
    expert: usize,
    target: usize,
}

fn best_move(
    input: &Input,
    x: &[Vec<bool>],
    assigned: &[Vec<u64>],
    slots_used: &[usize],
    rank_load: &[u64],
    ranks: &[usize],
    busy: usize,
) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    for expert in 0..input.n_experts {
        let available = assigned[expert][busy];
        if available == 0 {
            continue;
        }
        for &target in ranks {
            if target == busy {
                continue;
            }
            let hosts = x[expert][target];
            if !hosts && slots_used[target] >= input.n_slot {
                continue;
            }
            if rank_load[target] >= rank_load[busy] {
                continue;
            }
            let delta = available.min((rank_load[busy] - rank_load[target]) / 2);
            if delta == 0 {
                continue;
            }
            let candidate = Candidate {
                delta,
                target_load: rank_load[target],
                cost: input.cost[busy][target],
                expert,
                target,
            };
            if best
                .as_ref()
                .map(|current| move_outranks(&candidate, current))
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }
    }
    best
}

/// Larger delta, then lower target load, then lower cost, then lower expert
/// index, then lower target index.
fn move_outranks(candidate: &Candidate, current: &Candidate) -> bool {
    if candidate.delta != current.delta {
        return candidate.delta > current.delta;
    }
    if candidate.target_load != current.target_load {
        return candidate.target_load < current.target_load;
    }
    if candidate.cost != current.cost {
        return candidate.cost < current.cost;
    }
    if candidate.expert != current.expert {
        return candidate.expert < current.expert;
    }
    candidate.target < current.target
}

/// Move `delta` assignments of `expert` from `from` to `to`.
/// Sources are drained in increasing source index.
fn move_tokens(q: &mut [Vec<Vec<u64>>], expert: usize, from: usize, to: usize, mut delta: u64) {
    for source in q.iter_mut() {
        if delta == 0 {
            return;
        }
        let avail = source[expert][from];
        if avail == 0 {
            continue;
        }
        let amount = avail.min(delta);
        source[expert][from] = avail - amount;
        source[expert][to] += amount;
        delta -= amount;
    }
    assert_eq!(delta, 0, "moved fewer assignments than delta");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost_matrix(n: usize, off: u64) -> Vec<Vec<u64>> {
        let mut cost = vec![vec![off; n]; n];
        for rank in 0..n {
            cost[rank][rank] = 0;
        }
        cost
    }

    fn instance(
        dom: Vec<usize>,
        main: Vec<usize>,
        n_slot: usize,
        w_bytes: Vec<u64>,
        s_tok: u64,
        omega: Vec<Vec<u64>>,
        cost: Vec<Vec<u64>>,
    ) -> Input {
        Input {
            n_ranks: dom.len(),
            n_experts: main.len(),
            dom,
            main,
            n_slot,
            w_bytes,
            s_tok,
            omega,
            cost,
        }
    }

    fn assert_invariants(input: &Input, plan: &Plan) {
        let n_ranks = input.n_ranks;
        let n_experts = input.n_experts;
        assert_eq!(plan.x.len(), n_experts);
        assert_eq!(plan.q.len(), n_ranks);
        assert_eq!(plan.rank_load.len(), n_ranks);

        for expert in 0..n_experts {
            assert_eq!(plan.x[expert].len(), n_ranks);
            assert!(
                plan.x[expert][input.main[expert]],
                "main instance missing for expert {expert}"
            );
        }

        for source in 0..n_ranks {
            assert_eq!(plan.q[source].len(), n_experts);
            for expert in 0..n_experts {
                assert_eq!(plan.q[source][expert].len(), n_ranks);
                let sum: u64 = plan.q[source][expert].iter().sum();
                assert_eq!(
                    sum, input.omega[source][expert],
                    "token conservation for source {source} expert {expert}"
                );
                for rank in 0..n_ranks {
                    if plan.q[source][expert][rank] > 0 {
                        assert!(
                            plan.x[expert][rank],
                            "assignment on a rank that does not host the expert"
                        );
                    }
                }
            }
        }

        for rank in 0..n_ranks {
            let non_main = (0..n_experts)
                .filter(|&expert| plan.x[expert][rank] && input.main[expert] != rank)
                .count();
            assert!(
                non_main <= input.n_slot,
                "rank {rank} hosts {non_main} non-main instances"
            );
            let load: u64 = (0..n_ranks)
                .map(|source| {
                    (0..n_experts)
                        .map(|expert| plan.q[source][expert][rank])
                        .sum::<u64>()
                })
                .sum();
            assert_eq!(plan.rank_load[rank], load, "rank load");
        }

        let grouped = ranks_by_domain(&input.dom);
        let demand = domain_demand(input, &grouped);
        for expert in 0..n_experts {
            let home = input.dom[input.main[expert]];
            for rank in 0..n_ranks {
                if plan.x[expert][rank] && input.dom[rank] != home {
                    let gain = benefit(
                        demand[input.dom[rank]][expert],
                        input.s_tok,
                        input.w_bytes[expert],
                    );
                    assert!(
                        gain > 0,
                        "cross-domain replica of expert {expert} on rank {rank} has benefit {gain}"
                    );
                }
            }
        }

        for source in 0..n_ranks {
            let home = input.dom[source];
            for expert in 0..n_experts {
                let local =
                    (0..n_ranks).any(|rank| plan.x[expert][rank] && input.dom[rank] == home);
                if !local {
                    continue;
                }
                for rank in 0..n_ranks {
                    if input.dom[rank] != home {
                        assert_eq!(
                            plan.q[source][expert][rank], 0,
                            "source {source} expert {expert} left its domain while a local instance exists"
                        );
                    }
                }
            }
        }
    }

    fn cross_domain_tokens(input: &Input, plan: &Plan) -> u64 {
        let mut total = 0u64;
        for source in 0..input.n_ranks {
            for expert in 0..input.n_experts {
                for rank in 0..input.n_ranks {
                    if input.dom[source] != input.dom[rank] {
                        total += plan.q[source][expert][rank];
                    }
                }
            }
        }
        total
    }

    fn lmax(plan: &Plan) -> u64 {
        plan.rank_load.iter().copied().max().unwrap_or(0)
    }

    #[test]
    fn token_conservation_reachability_capacity_main_and_benefit() {
        // Three positive-benefit experts, one slot per rank, two ranks in the
        // source domain. Highest benefit takes rank 0, next takes rank 1,
        // the third is rejected for lack of a slot.
        let input = instance(
            vec![0, 0, 1],
            vec![2, 2, 2],
            1,
            vec![1, 1, 1],
            1,
            vec![vec![10, 8, 6], vec![0, 0, 0], vec![0, 0, 0]],
            cost_matrix(3, 1),
        );
        let plan = solve(&input);
        assert_invariants(&input, &plan);
        assert_eq!(plan.x[0], vec![true, false, true]);
        assert_eq!(plan.x[1], vec![false, true, true]);
        assert_eq!(plan.x[2], vec![false, false, true]);
        assert_eq!(plan.q[0][0], vec![10, 0, 0]);
        assert_eq!(plan.q[0][1], vec![0, 8, 0]);
        assert_eq!(plan.q[0][2], vec![0, 0, 6]);
        assert_eq!(plan.rank_load, vec![10, 8, 6]);

        // Equal benefit places the lower expert index and leaves the other unplaced.
        let tied = instance(
            vec![0, 1],
            vec![1, 1],
            1,
            vec![1, 1],
            1,
            vec![vec![5, 5], vec![0, 0]],
            cost_matrix(2, 1),
        );
        let tied_plan = solve(&tied);
        assert_invariants(&tied, &tied_plan);
        assert_eq!(tied_plan.x[0], vec![true, true]);
        assert_eq!(tied_plan.x[1], vec![false, true]);
        assert_eq!(tied_plan.q[0][0], vec![5, 0]);
        assert_eq!(tied_plan.q[0][1], vec![0, 5]);
    }

    #[test]
    fn nonpositive_benefit_creates_no_cross_domain_replica() {
        let negative = instance(
            vec![0, 1],
            vec![1],
            1,
            vec![1000],
            1,
            vec![vec![10], vec![0]],
            cost_matrix(2, 1),
        );
        let plan = solve(&negative);
        assert_invariants(&negative, &plan);
        assert_eq!(plan.x[0], vec![false, true]);
        assert_eq!(plan.q[0][0], vec![0, 10]);
        assert!(plan.x[0].iter().filter(|&&hosted| hosted).count() == 1);

        // 2 * 4 * 2 - 16 = 0, which must not place a cross-domain replica.
        let zero = instance(
            vec![0, 1],
            vec![1],
            2,
            vec![16],
            2,
            vec![vec![4], vec![0]],
            cost_matrix(2, 1),
        );
        let zero_plan = solve(&zero);
        assert_invariants(&zero, &zero_plan);
        assert!(!zero_plan.x[0][0]);
        assert!(zero_plan.x[0][1]);
        assert_eq!(zero_plan.q[0][0], vec![0, 4]);
    }

    #[test]
    fn positive_benefit_places_replica_and_prefers_source_domain() {
        let input = instance(
            vec![0, 1],
            vec![1],
            1,
            vec![1],
            1,
            vec![vec![8], vec![0]],
            cost_matrix(2, 3),
        );
        let gain = benefit(8, 1, 1);
        assert!(gain > 0);
        let plan = solve(&input);
        assert_invariants(&input, &plan);
        assert_eq!(plan.x[0], vec![true, true]);
        assert_eq!(plan.q[0][0], vec![8, 0]);
        assert_eq!(plan.rank_load, vec![8, 0]);
    }

    #[test]
    fn refinement_holds_lmax_and_cross_domain_tokens() {
        // Expert 0 is replicated into domain 0. Expert 1 stays remote, so some
        // tokens are cross-domain. Refinement can shed expert 0 inside domain 0
        // but must not increase the cross-domain token count.
        let input = instance(
            vec![0, 0, 1, 1],
            vec![2, 0],
            1,
            vec![1, 1000],
            1,
            vec![vec![9, 0], vec![0, 0], vec![0, 4], vec![0, 0]],
            cost_matrix(4, 1),
        );
        let before = place_and_route(&input);
        let mut after = before.clone();
        refine_plan(&input, &mut after);
        assert!(lmax(&after) < lmax(&before));
        assert!(cross_domain_tokens(&input, &after) <= cross_domain_tokens(&input, &before));
        assert_invariants(&input, &after);
        assert_eq!(solve(&input), after);
    }

    #[test]
    fn deterministic_equal_plans() {
        let input = instance(
            vec![0, 0, 1, 1],
            vec![0, 1, 2, 3, 2],
            1,
            vec![1, 1, 1, 1, 1],
            1,
            {
                let mut omega = vec![vec![0u64; 5]; 4];
                omega[0][4] = 8;
                omega[1][2] = 3;
                omega[3][0] = 5;
                omega
            },
            cost_matrix(4, 2),
        );
        assert_eq!(solve(&input), solve(&input));
    }

    #[test]
    fn narrative_domain_local_execution() {
        // Ranks 0 and 1 are domain 0. Ranks 2 and 3 are domain 1.
        // Expert 4's main is rank 2. Rank 0 has 8 assignments to expert 4.
        // benefit = 2 * 8 * s_tok - w_bytes = 15 > 0.
        let mut omega = vec![vec![0u64; 5]; 4];
        omega[0][4] = 8;
        let input = instance(
            vec![0, 0, 1, 1],
            vec![0, 1, 2, 3, 2],
            1,
            vec![1, 1, 1, 1, 1],
            1,
            omega,
            cost_matrix(4, 1),
        );
        assert_eq!(benefit(8, input.s_tok, input.w_bytes[4]), 15);

        let staged = place_and_route(&input);
        assert_eq!(staged.q[0][4], vec![8, 0, 0, 0]);
        assert!(staged.x[4][0]);
        assert!(staged.x[4][2]);

        let plan = solve(&input);
        assert_invariants(&input, &plan);
        assert_eq!(plan.q[0][4][0] + plan.q[0][4][1], 8);
        assert_eq!(plan.q[0][4][2], 0);
        assert_eq!(plan.q[0][4][3], 0);
        assert_eq!(plan.q[0][4], vec![4, 4, 0, 0]);
        assert!(plan.x[4][2]);
    }

    #[test]
    fn fallback_routing_splits_remainder_to_lower_rank() {
        // Domain 2 demand is too small for a replica (benefit 0). Its tokens
        // fall back to every deployed instance and the odd token goes to the
        // lower rank index.
        let input = instance(
            vec![0, 1, 2, 2],
            vec![1],
            1,
            vec![6],
            1,
            vec![vec![10], vec![0], vec![2], vec![1]],
            cost_matrix(4, 1),
        );
        assert_eq!(benefit(3, 1, 6), 0);
        assert!(benefit(10, 1, 6) > 0);
        let plan = solve(&input);
        assert_invariants(&input, &plan);
        assert_eq!(plan.q[0][0], vec![10, 0, 0, 0]);
        assert_eq!(plan.q[2][0], vec![1, 1, 0, 0]);
        assert_eq!(plan.q[3][0], vec![1, 0, 0, 0]);
    }

    #[test]
    fn cost_tie_break_prefers_lower_cost() {
        let mut cost = cost_matrix(3, 4);
        cost[0][1] = 5;
        cost[0][2] = 1;
        let input = instance(
            vec![0, 0, 0],
            vec![0],
            1,
            vec![1],
            1,
            vec![vec![2], vec![0], vec![0]],
            cost,
        );
        let plan = solve(&input);
        assert_invariants(&input, &plan);
        assert_eq!(plan.q[0][0], vec![1, 0, 1]);

        let equal_cost = instance(
            vec![0, 0, 0],
            vec![0],
            1,
            vec![1],
            1,
            vec![vec![2], vec![0], vec![0]],
            cost_matrix(3, 3),
        );
        let equal_plan = solve(&equal_cost);
        assert_eq!(equal_plan.q[0][0], vec![1, 1, 0]);
    }

    #[test]
    fn lower_target_load_beats_lower_cost() {
        let mut cost = cost_matrix(3, 1);
        cost[0][1] = 1;
        cost[0][2] = 9;
        let input = instance(
            vec![0, 0, 0],
            vec![0, 1],
            1,
            vec![1, 1],
            1,
            vec![vec![3, 0], vec![0, 1], vec![0, 0]],
            cost,
        );
        let plan = solve(&input);
        assert_invariants(&input, &plan);
        assert_eq!(plan.q[0][0], vec![2, 0, 1]);
    }

    #[test]
    fn move_takes_lower_source_index_first() {
        let input = instance(
            vec![0, 0],
            vec![0],
            1,
            vec![1],
            1,
            vec![vec![3], vec![5]],
            cost_matrix(2, 1),
        );
        let plan = solve(&input);
        assert_invariants(&input, &plan);
        assert_eq!(plan.q[0][0], vec![0, 3]);
        assert_eq!(plan.q[1][0], vec![4, 1]);
    }

    #[test]
    fn busiest_tie_break_lower_index() {
        let input = instance(
            vec![0, 0, 0],
            vec![0, 1],
            1,
            vec![1, 1],
            1,
            vec![vec![5, 0], vec![0, 5], vec![0, 0]],
            cost_matrix(3, 1),
        );
        let plan = solve(&input);
        assert_invariants(&input, &plan);
        assert_eq!(plan.q[0][0], vec![2, 0, 3]);
        assert_eq!(plan.q[1][1], vec![1, 4, 0]);
        assert_eq!(plan.rank_load, vec![3, 4, 3]);
    }

    #[test]
    fn best_move_orders_expert_before_target() {
        let input = instance(
            vec![0, 0, 0],
            vec![0, 1],
            1,
            vec![1, 1],
            1,
            vec![vec![0, 0], vec![0, 0], vec![0, 0]],
            cost_matrix(3, 1),
        );
        let x = vec![vec![true, false, false], vec![false, true, false]];
        let assigned = vec![vec![6, 0, 0], vec![6, 0, 0]];
        let slots_used = vec![0usize, 1, 0];
        let rank_load = vec![12u64, 0, 0];
        let chosen = best_move(
            &input,
            &x,
            &assigned,
            &slots_used,
            &rank_load,
            &[0, 1, 2],
            0,
        )
        .expect("a positive move");
        assert_eq!(chosen.expert, 0);
        assert_eq!(chosen.target, 2);
        assert_eq!(chosen.delta, 6);
    }

    #[test]
    fn expert_tie_break_places_lower_index_on_the_only_target() {
        let input = instance(
            vec![0, 0],
            vec![0, 0],
            1,
            vec![1, 1],
            1,
            vec![vec![2, 2], vec![0, 0]],
            cost_matrix(2, 1),
        );
        let plan = solve(&input);
        assert_invariants(&input, &plan);
        assert_eq!(plan.q[0][0], vec![0, 2]);
        assert_eq!(plan.q[0][1], vec![2, 0]);
        assert!(plan.x[0][1]);
        assert!(!plan.x[1][1]);
    }
}
