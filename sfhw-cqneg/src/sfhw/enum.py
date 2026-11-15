"""Variable-elimination enumerator for full CQ-neg queries.

This is Algorithm 1 (SolveQuery) and Algorithm 2 (BuildEnumDS), including
Extend and Merge, from Section 4 of arXiv:2610.03982.

Bags chi_sigma^S are computed once on the original signed hypergraph. At
step i the reduced query supplies the relations, while annotations on
negative atoms point back to the original negative edges that define the bag.
Join(H[U]) is the natural join of the projections of atoms in H onto U,
including a 0-ary atom as a global true/false constraint. That is the induced
subquery used by the AGM arguments in the appendix.

Runtime simplifications (semantics unchanged):
- joins are nested loops, not worst-case optimal joins;
- Extend scans the skip links at a node instead of hashing by provenance;
- Merge is an iterative pairwise merge, not the linear scan of Lemma E.2;
- a negative atom whose full-list relation is empty is not added to the
  reduced query (it would never match, so later subsets are unchanged).
None of these claim the O(|I|^sfhw) preprocessing bound.
"""

from __future__ import annotations

import itertools

from .width import canonical_bags, fhw_edges, sfhw_edges, witnessing_order


class Atom:
    """A named atom. Negative atoms carry an annotation of original edge ids."""

    def __init__(self, uid, name, variables, tuples, annotation=None):
        self.uid = uid
        self.name = name
        self.variables = tuple(variables)
        self.tuples = {tuple(tup) for tup in tuples}
        self.annotation = frozenset(annotation or ())

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        return isinstance(other, Atom) and self.uid == other.uid

    def contains(self, variable):
        return variable in self.variables

    def as_dicts(self):
        out = []
        for tup in self.tuples:
            out.append(dict(zip(self.variables, tup)))
        return out


class Query:
    """A safe full CQ-neg: positive atoms E+ and negative atoms E-."""

    def __init__(self, variables, positive, negative):
        self.variables = list(variables)
        self.positive = list(positive)
        self.negative = list(negative)

    def pos_edges(self):
        return [frozenset(atom.variables) for atom in self.positive]

    def neg_edges(self):
        return [frozenset(atom.variables) for atom in self.negative]


def make_query(positive, negative=()):
    """Build a query from (name, variables, tuples) triples.

    Variables are the union of atom variables, in order of first appearance.
    Each negative atom is annotated with its own id, as in Section 4.1.
    Raises ValueError if a negative variable is missing from every positive atom.
    """
    positive_atoms = []
    negative_atoms = []
    variables = []
    seen = set()

    def add_var(var):
        if var not in seen:
            seen.add(var)
            variables.append(var)

    for name, vars_, tuples in positive:
        for var in vars_:
            add_var(var)
        positive_atoms.append(Atom(f"P:{name}", name, vars_, tuples))
    for name, vars_, tuples in negative:
        for var in vars_:
            add_var(var)
        negative_atoms.append(
            Atom(f"N:{name}", name, vars_, tuples, annotation=(f"N:{name}",))
        )

    positive_vars = set()
    for atom in positive_atoms:
        positive_vars.update(atom.variables)
    for atom in negative_atoms:
        missing = [var for var in atom.variables if var not in positive_vars]
        if missing:
            raise ValueError(
                f"unsafe query: {missing} occur in negative atom {atom.name} "
                "but in no positive atom"
            )
    return Query(variables, positive_atoms, negative_atoms)


def sfhw(query):
    """Signed fractional hypertree width of a query (Definition 3.7)."""
    return sfhw_edges(query.variables, query.pos_edges(), query.neg_edges())


def fhw(query):
    """Fractional hypertree width of the positive hypergraph."""
    return fhw_edges(query.variables, query.pos_edges())


class _Node:
    def __init__(self, kind, value=None):
        self.kind = kind
        self.value = value
        self.index = 0
        self.skips_out = []
        self.skips_in = []


class _Skip:
    def __init__(self, start, end, label, prov, bucket_key):
        self.start = start
        self.end = end
        self.label = label
        self.prov = prov
        self.bucket_key = bucket_key


class _EnumDS:
    """Hash table of linked lists plus skip links for one eliminated variable."""

    def __init__(self, variable, bucket_vars, buckets):
        self.variable = variable
        self.bucket_vars = tuple(bucket_vars)
        self.buckets = buckets


def _consistent(left, right):
    for var, val in left.items():
        if var in right and right[var] != val:
            return False
    return True


def _label_matches(label, assignment):
    for var, val in label:
        if var not in assignment or assignment[var] != val:
            return False
    return True


def _sort_key(value):
    return (type(value).__name__, repr(value))


def _project_atom(atom, universe):
    """Project an atom onto universe.

    Returns ('false',) if a 0-ary atom is empty or the projection is empty
    in a way that kills the join, ('skip',) if the atom does not meet
    universe, or ('rel', vars, tuples) for a real constraint.
    """
    if not atom.variables:
        if not atom.tuples:
            return ("false",)
        return ("skip",)
    keep = [var for var in atom.variables if var in universe]
    if not keep:
        return ("skip",)
    indexes = [atom.variables.index(var) for var in keep]
    projected = set()
    for tup in atom.tuples:
        projected.add(tuple(tup[i] for i in indexes))
    if not projected:
        return ("false",)
    return ("rel", tuple(keep), projected)


def induced_join(atoms, universe):
    """Join(H[U]): natural join of atom projections onto U.

    A variable in U that no atom mentions is not part of the result schema.
    Zero constraints yield a single empty tuple.
    """
    relations = []
    for atom in atoms:
        projected = _project_atom(atom, universe)
        if projected[0] == "false":
            return []
        if projected[0] == "rel":
            relations.append(projected[1:])
    if not relations:
        return [{}]
    partials = [{}]
    for vars_, tuples in relations:
        nxt = []
        seen = set()
        for partial in partials:
            for tup in tuples:
                candidate = dict(zip(vars_, tup))
                if not _consistent(partial, candidate):
                    continue
                merged = partial.copy()
                merged.update(candidate)
                key = tuple(sorted(merged.items(), key=lambda item: repr(item[0])))
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(merged)
        partials = nxt
        if not partials:
            return []
    return partials


def _precompute_bags(query, order):
    """chi_sigma^S(v) for every subset S of original negative edge ids."""
    pos = [frozenset(atom.variables) for atom in query.positive]
    neg = [(atom.uid, frozenset(atom.variables)) for atom in query.negative]
    bags = {}
    ids = [uid for uid, _ in neg]
    id_edge = dict(neg)
    for r in range(len(ids) + 1):
        for subset in itertools.combinations(ids, r):
            annotation = frozenset(subset)
            edges = list(pos) + [id_edge[uid] for uid in subset]
            bags[annotation] = canonical_bags(order, edges)
    return bags


def _binding_key(binding, variables):
    return tuple(binding[var] for var in variables)


def _degree(tuples, variable, binding):
    count = 0
    for tup in tuples:
        if _consistent(binding, tup) and variable in tup:
            count += 1
    return count


def _join_binding(binding, tuples, variable):
    out = []
    seen = set()
    for tup in tuples:
        if variable not in tup or not _consistent(binding, tup):
            continue
        merged = binding.copy()
        merged.update(tup)
        key = (merged[variable], tuple(sorted(merged.items(), key=lambda item: repr(item[0]))))
        if key in seen:
            continue
        seen.add(key)
        out.append(merged)
    return out


def _noncontaining_overlap(left, right):
    ls, le = left.start.index, left.end.index
    rs, re = right.start.index, right.end.index
    return (ls < rs < le < re) or (rs < ls < re < le)


def merge(skip_links):
    """Merge non-containingly overlapping skip links with the same label.

    Algorithm 2, lines 29-32. Links in different linked lists are never
    compared. The loop repeats until no overlapping pair remains, which is
    the closure of the pairwise step in the pseudocode.
    """
    groups = {}
    for link in skip_links:
        groups.setdefault((link.bucket_key, link.label), []).append(link)
    merged = []
    for group in groups.values():
        current = list(group)
        changed = True
        while changed:
            changed = False
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    if not _noncontaining_overlap(current[i], current[j]):
                        continue
                    left, right = current[i], current[j]
                    start = left.start if left.start.index < right.start.index else right.start
                    end = left.end if left.end.index > right.end.index else right.end
                    combined = _Skip(start, end, left.label, left.prov, left.bucket_key)
                    current = [
                        current[k]
                        for k in range(len(current))
                        if k != i and k != j
                    ]
                    current.append(combined)
                    changed = True
                    break
                if changed:
                    break
        # Identical links (same endpoints) do not overlap strictly. Keep one.
        uniq = []
        seen = set()
        for link in current:
            key = (link.start.index, link.end.index)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(link)
        merged.extend(uniq)
    return merged


def extend(value_node, nodes, binding, provenance, label_vars, bucket_key):
    """Extend one skipped value into a skip link. Algorithm 2, lines 21-28.

    Incoming: longest already-stored skip link whose provenance is a proper
    subset of S, that ends at this value, and whose label agrees with the
    binding. Outgoing is symmetric. Otherwise the link is the singleton
    from prev to next.
    """
    incoming = None
    for link in value_node.skips_in:
        if not (link.prov < provenance and link.end is value_node):
            continue
        if not _label_matches(link.label, binding):
            continue
        if incoming is None or link.start.index < incoming.start.index:
            incoming = link
    if incoming is not None:
        start = incoming.start
    else:
        start = nodes[value_node.index - 1]

    outgoing = None
    for link in value_node.skips_out:
        if not (link.prov < provenance and link.start is value_node):
            continue
        if not _label_matches(link.label, binding):
            continue
        if outgoing is None or link.end.index > outgoing.end.index:
            outgoing = link
    if outgoing is not None:
        end = outgoing.end
    else:
        end = nodes[value_node.index + 1]

    label = tuple((var, binding[var]) for var in label_vars)
    return _Skip(start, end, label, provenance, bucket_key)


def _install(links):
    for link in links:
        link.start.skips_out.append(link)
        link.end.skips_in.append(link)


def _covers_entire(link, nodes):
    return link.start is nodes[0] and link.end is nodes[-1]


def build_enum_ds(query, variable, bags, uid_seed):
    """Algorithm 2. Returns (data structure, reduced query, next uid seed)."""
    empty = frozenset()
    bag_empty = bags[empty][variable]
    bucket_vars = tuple(
        var for var in query.variables if var in bag_empty and var != variable
    )
    label_universe = set(bag_empty)

    positive_atoms = list(query.positive)
    l_empty = induced_join(positive_atoms, label_universe)
    grouped = {}
    for tup in l_empty:
        if variable not in tup:
            continue
        if any(var not in tup for var in bucket_vars):
            continue
        key = _binding_key(tup, bucket_vars)
        grouped.setdefault(key, set()).add(tup[variable])

    buckets = {}
    value_index = {}
    for key, values in grouped.items():
        bucket_node = _Node("bucket", key)
        sentinel = _Node("sentinel")
        middle = [_Node("value", value) for value in sorted(values, key=_sort_key)]
        nodes = [bucket_node, *middle, sentinel]
        for index, node in enumerate(nodes):
            node.index = index
        buckets[key] = nodes
        for node in middle:
            value_index[(key, node.value)] = node

    ds = _EnumDS(variable, bucket_vars, buckets)

    containing = [atom for atom in query.negative if atom.contains(variable)]
    p_relations = {}
    for atom in containing:
        bag = bags[atom.annotation][variable]
        atoms = positive_atoms + [atom]
        joined = induced_join(atoms, set(bag))
        p_relations[atom] = [tup for tup in joined if variable in tup]

    # Increasing cardinality, then a stable name order.
    subsets = []
    containing_ids = list(containing)
    n = len(containing_ids)
    for r in range(1, n + 1):
        level = []
        for mask in range(1 << n):
            if bin(mask).count("1") != r:
                continue
            subset = [containing_ids[i] for i in range(n) if mask & (1 << i)]
            level.append(subset)
        level.sort(key=lambda subset: tuple(atom.uid for atom in subset))
        subsets.extend(level)

    new_negative = []
    for subset in subsets:
        provenance = frozenset(atom.uid for atom in subset)
        annotation = frozenset().union(*(atom.annotation for atom in subset))
        bag_s = bags[annotation][variable]
        label_vars = tuple(
            var
            for var in query.variables
            if var in bag_s and var not in bag_empty
        )
        join_vars = set(bag_s) - {variable}
        l_s = induced_join(positive_atoms + subset, join_vars)
        raw_links = []
        for binding in l_s:
            if any(var not in binding for var in bucket_vars):
                continue
            if any(var not in binding for var in label_vars):
                continue
            best_atom = None
            best_deg = None
            for atom in subset:
                deg = _degree(p_relations[atom], variable, binding)
                if best_deg is None or deg < best_deg:
                    best_deg = deg
                    best_atom = atom
            points = _join_binding(binding, p_relations[best_atom], variable)
            bucket_key = _binding_key(binding, bucket_vars)
            for point in points:
                node = value_index.get((bucket_key, point[variable]))
                if node is None:
                    continue
                raw_links.append(
                    extend(
                        node,
                        buckets[bucket_key],
                        point,
                        provenance,
                        label_vars,
                        bucket_key,
                    )
                )
        merged = merge(raw_links)
        _install(merged)

        # Full-list skip labels become a negative atom of the reduced query.
        # Variables are chi^{A_S}(sigma_i) without sigma_i: label then bucket.
        out_vars = tuple(label_vars) + tuple(bucket_vars)
        tuples = []
        seen = set()
        for link in merged:
            nodes = buckets[link.bucket_key]
            if not _covers_entire(link, nodes):
                continue
            values = dict(link.label)
            for var, val in zip(bucket_vars, link.bucket_key):
                values[var] = val
            tup = tuple(values[var] for var in out_vars)
            if tup not in seen:
                seen.add(tup)
                tuples.append(tup)
        if tuples:
            uid = f"U{uid_seed}"
            uid_seed += 1
            name = "U{" + ",".join(atom.name for atom in subset) + "}"
            new_negative.append(
                Atom(uid, name, out_vars, tuples, annotation=annotation)
            )

    kept_positive = [atom for atom in query.positive if not atom.contains(variable)]
    proj_vars = bucket_vars
    proj_tuples = list(grouped.keys())
    uid = f"U{uid_seed}"
    uid_seed += 1
    kept_positive.append(Atom(uid, f"Uempty[{variable}]", proj_vars, proj_tuples))

    kept_negative = [atom for atom in query.negative if not atom.contains(variable)]
    kept_negative.extend(new_negative)
    reduced_vars = [var for var in query.variables if var != variable]
    reduced = Query(reduced_vars, kept_positive, kept_negative)
    return ds, reduced, uid_seed


def enumerate_extensions(ds, suffix):
    """Walk one linked list, following the longest consistent skip link.

    Section 4.2: from the current node, follow the longest skip whose label
    agrees with the suffix; otherwise step to the successor. Values strictly
    between a skip's endpoints are not emitted. The bucket key and the
    sentinel are not values.
    """
    try:
        key = tuple(suffix[var] for var in ds.bucket_vars)
    except KeyError:
        return
    nodes = ds.buckets.get(key)
    if not nodes:
        return
    current = nodes[0]
    while True:
        best_end = None
        best_index = current.index
        for link in current.skips_out:
            if link.end.index <= best_index:
                continue
            if not _label_matches(link.label, suffix):
                continue
            best_end = link.end
            best_index = link.end.index
        if best_end is not None:
            current = best_end
        else:
            current = nodes[current.index + 1]
        if current.kind == "sentinel":
            return
        yield current.value


def solve_query(query, order=None):
    """Algorithm 1. Emit each full answer tuple once, as a dict.

    If order is omitted, use an order that witnesses sfhw (any correct order
    enumerates the same answers; the width only bounds preprocessing).
    """
    if order is None:
        order = witnessing_order(query.variables, query.pos_edges(), query.neg_edges())
    else:
        order = list(order)
        if set(order) != set(query.variables) or len(order) != len(query.variables):
            raise ValueError("order must be a permutation of the query variables")

    bags = _precompute_bags(query, order)
    structures = []
    current = query
    uid_seed = 0
    for variable in order:
        ds, current, uid_seed = build_enum_ds(current, variable, bags, uid_seed)
        structures.append(ds)

    answers = []

    def rec(index, suffix):
        if index < 0:
            answers.append(dict(suffix))
            return
        for value in enumerate_extensions(structures[index], suffix):
            suffix[order[index]] = value
            rec(index - 1, suffix)
            del suffix[order[index]]

    rec(len(order) - 1, {})
    return answers
