"""Greedy mixed-precision bit allocation from injected local scores."""


def greedy_allocate(n_g, scores, budget, candidates=(3, 5, 7), budget_is_average=False):
    """Drop bit width one candidate step at a time until the budget holds.

    n_g[g] is the element count of block g. scores[g][b] is the injected
    local score of block g at bit width b. This function does not recompute
    quantization noise.

    candidates are sorted ascending. Every block starts at max(candidates).
    If budget_is_average is true, Bud = budget * sum(n_g). Otherwise Bud = budget.
    Cost = sum_g b_g * n_g.

    While Cost > Bud, among blocks that can drop one candidate step, pick
    the smallest

        Delta = max(s_loc(b_next) - s_loc(b), 0) / ((b - b_next) * n_g)

    Ties take the smallest block index. If no block can drop, stop.
    If Cost is already <= Bud, return the max assignment.
    """
    counts = [int(n) for n in n_g]
    widths = sorted({int(b) for b in candidates})
    if not widths:
        raise ValueError("candidates must be non-empty")
    if any(n < 0 for n in counts):
        raise ValueError("n_g entries must be non-negative")

    assignment = [widths[-1]] * len(counts)
    total_count = sum(counts)
    bud = float(budget) * total_count if budget_is_average else float(budget)

    def cost(bits):
        return sum(b * n for b, n in zip(bits, counts))

    while cost(assignment) > bud:
        best_index = None
        best_delta = None
        for index, current in enumerate(assignment):
            level = widths.index(current)
            if level == 0:
                continue
            nxt = widths[level - 1]
            increase = max(float(scores[index][nxt]) - float(scores[index][current]), 0.0)
            step = (current - nxt) * counts[index]
            if step <= 0:
                raise ValueError("bit drop must reduce a positive element count")
            delta = increase / step
            if best_delta is None or delta < best_delta or (delta == best_delta and index < best_index):
                best_delta = delta
                best_index = index
        if best_index is None:
            break
        level = widths.index(assignment[best_index])
        assignment[best_index] = widths[level - 1]

    return assignment
