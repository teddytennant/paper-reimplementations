//! Finite weak solver for Appendix A of [arXiv:2610.12306](https://arxiv.org/abs/2610.12306).
//!
//! Objects are feature tuples of length at most [`MAX_DIMENSION`]. Coordinates
//! are 0-based indices into that tuple (the paper writes the same family
//! 1-based). A procedure is an anonymous coordinate test stored with the
//! positive pattern that produced it. The solver does not receive a hidden
//! label: a subset solves a task only when it is acceptable and exactly one
//! candidate matches the pattern.
//!
//! Stability after two successful traces is a reading of the paper's
//! "repeated successful traces".

/// Demonstration bound. Tasks with more coordinates fail; they are not searched.
pub const MAX_DIMENSION: usize = 3;

/// A procedure or concept is stable after this many successful traces that used it.
pub const STABILITY_TRACES: usize = 2;

/// Feature values are opaque strings. The solver does not interpret them.
pub type Feature = String;

/// An object code `x = (x_0, ..., x_{d-1})` with `d <= MAX_DIMENSION`.
pub type Object = Vec<Feature>;

/// `T = (P, N, C)`. `P` must be non-empty for any subset to be acceptable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    pub positives: Vec<Object>,
    pub negatives: Vec<Object>,
    pub candidates: Vec<Object>,
}

impl Task {
    pub fn new(positives: Vec<Object>, negatives: Vec<Object>, candidates: Vec<Object>) -> Self {
        Self {
            positives,
            negatives,
            candidates,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Solved,
    Failed,
    Underdetermined,
}

/// Outcome of one call to [`KnowledgeSpace::solve`].
///
/// `stage` is 1, 2, or 3 when [`Status::Solved`]. It is 0 when the task is
/// failed or underdetermined, and the knowledge space is left unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolveResult {
    pub status: Status,
    pub stage: u8,
    pub candidate: Option<usize>,
    pub procedure_id: Option<usize>,
    pub stable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcedureKind {
    /// Stage 1 primitive singleton.
    Singleton,
    /// Stage 2 union of stored procedures. Not a primitive.
    Composed,
    /// Stage 3 new projection or abstraction.
    Generated,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Procedure {
    pub id: usize,
    pub coords: Vec<usize>,
    pub pattern: Vec<Feature>,
    pub kind: ProcedureKind,
    /// Stored procedure ids combined by a stage-2 composition. Empty otherwise.
    pub components: Vec<usize>,
    pub success_count: usize,
    pub stable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trace {
    pub procedure_id: usize,
    pub stage: u8,
    pub candidate: usize,
}

/// Class-distinction node keyed by a coordinate subset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Concept {
    pub coords: Vec<usize>,
    pub success_count: usize,
    pub stable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Primitive {
    pub name: &'static str,
}

/// K0 primitive names. No classifiers are stored at initialization.
pub const PRIMITIVE_NAMES: &[&str] = &["read", "project", "equal", "not_equal", "and"];

/// Equality primitive. Singleton tests compare coordinate values through this.
pub fn equal<T: PartialEq>(a: &T, b: &T) -> bool {
    a == b
}

/// Inequality primitive.
pub fn not_equal<T: PartialEq>(a: &T, b: &T) -> bool {
    !equal(a, b)
}

/// Conjunction primitive. Coordinate tests combine with this, not a new rule.
pub fn and(a: bool, b: bool) -> bool {
    a && b
}

/// Project `object` onto `coords`. `None` if any index is out of range.
pub fn project(object: &[Feature], coords: &[usize]) -> Option<Vec<Feature>> {
    let mut out = Vec::with_capacity(coords.len());
    for &i in coords {
        out.push(object.get(i)?.clone());
    }
    Some(out)
}

/// Read a segmented task code.
///
/// Returns the common dimension, or `None` when positives are empty, lengths
/// disagree, `d == 0`, or `d > MAX_DIMENSION`. Out-of-bound tasks are not searched.
pub fn read(task: &Task) -> Option<usize> {
    let first = task.positives.first()?;
    let d = first.len();
    if d == 0 || d > MAX_DIMENSION {
        return None;
    }
    let consistent = task
        .positives
        .iter()
        .chain(task.negatives.iter())
        .chain(task.candidates.iter())
        .all(|object| object.len() == d);
    if consistent {
        Some(d)
    } else {
        None
    }
}

/// Non-empty coordinate subsets of `{0..d}`, increasing cardinality then lexicographic.
///
/// For `d == 3` this is `{0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}`.
/// Returns an empty family when `d` is outside the demonstration bound.
pub fn bounded_family(d: usize) -> Vec<Vec<usize>> {
    if d == 0 || d > MAX_DIMENSION {
        return Vec::new();
    }
    let mut family = Vec::new();
    for k in 1..=d {
        let mut prefix = Vec::new();
        combinations(d, k, 0, &mut prefix, &mut family);
    }
    family
}

fn combinations(
    d: usize,
    k: usize,
    start: usize,
    prefix: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if prefix.len() == k {
        out.push(prefix.clone());
        return;
    }
    let remaining = k - prefix.len();
    for i in start..d {
        if d - i < remaining {
            break;
        }
        prefix.push(i);
        combinations(d, k, i + 1, prefix, out);
        prefix.pop();
    }
}

fn projections_equal(left: &[Feature], right: &[Feature]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut acc = true;
    for (a, b) in left.iter().zip(right.iter()) {
        acc = and(acc, equal(a, b));
    }
    acc
}

fn union_coords(parts: &[&[usize]]) -> Vec<usize> {
    let mut coords = Vec::new();
    for part in parts {
        for &c in *part {
            if !coords.contains(&c) {
                coords.push(c);
            }
        }
    }
    coords.sort_unstable();
    coords
}

/// Positive pattern on `coords`, if every positive projects and they all agree.
fn positive_pattern(task: &Task, coords: &[usize]) -> Option<Vec<Feature>> {
    let first = project(task.positives.first()?, coords)?;
    for positive in task.positives.iter().skip(1) {
        let projected = project(positive, coords)?;
        if not_equal(&projected, &first) {
            return None;
        }
    }
    Some(first)
}

#[derive(Clone)]
struct Attempt {
    coords: Vec<usize>,
    pattern: Vec<Feature>,
    candidate: usize,
    existing_id: Option<usize>,
    kind: ProcedureKind,
    components: Vec<usize>,
}

/// Acceptable subset that selects exactly one candidate.
///
/// When `required_pattern` is set, positives must agree on that stored pattern.
/// A mismatch does not solve the task.
fn try_subset(
    task: &Task,
    coords: &[usize],
    required_pattern: Option<&[Feature]>,
    existing_id: Option<usize>,
    kind: ProcedureKind,
    components: Vec<usize>,
) -> Option<Attempt> {
    if coords.is_empty() {
        return None;
    }
    let pattern = positive_pattern(task, coords)?;
    if let Some(required) = required_pattern {
        if not_equal(&pattern, &required.to_vec()) {
            return None;
        }
    }
    for negative in &task.negatives {
        let projected = project(negative, coords)?;
        if !not_equal(&projected, &pattern) {
            return None;
        }
    }
    let mut found = None;
    for (index, candidate) in task.candidates.iter().enumerate() {
        let projected = project(candidate, coords)?;
        if projections_equal(&projected, &pattern) {
            if found.is_some() {
                return None;
            }
            found = Some(index);
        }
    }
    Some(Attempt {
        coords: coords.to_vec(),
        pattern,
        candidate: found?,
        existing_id,
        kind,
        components,
    })
}

enum StageOutcome {
    Miss,
    Conflict,
    Hit(Attempt),
}

/// Among unique selectors in this stage: minimal-cardinality disagreement is
/// underdetermination. Agreement takes the first item in stage order. A
/// conflict only at larger cardinality yields the first minimal selector.
fn decide(solvers: Vec<Attempt>) -> StageOutcome {
    let mut deduped = Vec::new();
    for solver in solvers {
        if !deduped
            .iter()
            .any(|kept: &Attempt| kept.coords == solver.coords)
        {
            deduped.push(solver);
        }
    }
    if deduped.is_empty() {
        return StageOutcome::Miss;
    }
    let min_card = deduped
        .iter()
        .map(|solver| solver.coords.len())
        .min()
        .unwrap();
    let minimal: Vec<Attempt> = deduped
        .iter()
        .filter(|solver| solver.coords.len() == min_card)
        .cloned()
        .collect();
    let chosen = minimal[0].candidate;
    if minimal.iter().any(|solver| solver.candidate != chosen) {
        return StageOutcome::Conflict;
    }
    if deduped.iter().all(|solver| solver.candidate == chosen) {
        return StageOutcome::Hit(deduped.remove(0));
    }
    StageOutcome::Hit(minimal.into_iter().next().unwrap())
}

fn unchanged(status: Status) -> SolveResult {
    SolveResult {
        status,
        stage: 0,
        candidate: None,
        procedure_id: None,
        stable: false,
    }
}

/// Knowledge space `K_t`. `new` is `K_0`: primitive records, no classifiers.
#[derive(Clone, Debug)]
pub struct KnowledgeSpace {
    primitives: Vec<Primitive>,
    procedures: Vec<Procedure>,
    traces: Vec<Trace>,
    concepts: Vec<Concept>,
}

impl KnowledgeSpace {
    pub fn new() -> Self {
        Self {
            primitives: PRIMITIVE_NAMES
                .iter()
                .copied()
                .map(|name| Primitive { name })
                .collect(),
            procedures: Vec::new(),
            traces: Vec::new(),
            concepts: Vec::new(),
        }
    }

    pub fn primitives(&self) -> &[Primitive] {
        &self.primitives
    }

    pub fn procedures(&self) -> &[Procedure] {
        &self.procedures
    }

    pub fn traces(&self) -> &[Trace] {
        &self.traces
    }

    pub fn concepts(&self) -> &[Concept] {
        &self.concepts
    }

    pub fn solve(&mut self, task: &Task) -> SolveResult {
        let Some(d) = read(task) else {
            return unchanged(Status::Failed);
        };

        match self.stage1(task, d) {
            StageOutcome::Conflict => return unchanged(Status::Underdetermined),
            StageOutcome::Hit(attempt) => return self.commit(attempt, 1),
            StageOutcome::Miss => {}
        }
        match self.stage2(task) {
            StageOutcome::Conflict => return unchanged(Status::Underdetermined),
            StageOutcome::Hit(attempt) => return self.commit(attempt, 2),
            StageOutcome::Miss => {}
        }
        match self.stage3(task, d) {
            StageOutcome::Conflict => return unchanged(Status::Underdetermined),
            StageOutcome::Hit(attempt) => return self.commit(attempt, 3),
            StageOutcome::Miss => unchanged(Status::Failed),
        }
    }

    fn subset_stored(&self, coords: &[usize]) -> bool {
        self.procedures
            .iter()
            .any(|procedure| procedure.coords == coords)
    }

    /// Stored procedures in insertion order, then primitive singletons.
    fn stage1(&self, task: &Task, d: usize) -> StageOutcome {
        let mut solvers = Vec::new();
        for procedure in &self.procedures {
            if procedure.coords.iter().any(|&c| c >= d) {
                continue;
            }
            if let Some(attempt) = try_subset(
                task,
                &procedure.coords,
                Some(&procedure.pattern),
                Some(procedure.id),
                procedure.kind,
                procedure.components.clone(),
            ) {
                solvers.push(attempt);
            }
        }
        for coords in bounded_family(d)
            .into_iter()
            .filter(|coords| coords.len() == 1)
        {
            if let Some(attempt) = try_subset(
                task,
                &coords,
                None,
                None,
                ProcedureKind::Singleton,
                Vec::new(),
            ) {
                solvers.push(attempt);
            }
        }
        decide(solvers)
    }

    /// Pairs before triples, each in lexicographic index order.
    /// Conjunction and "after" are the same union of coordinate subsets.
    fn stage2(&self, task: &Task) -> StageOutcome {
        let n = self.procedures.len();
        let mut probes: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                probes.push((vec![i, j], self.union_of(&[i, j])));
            }
        }
        // "after" is enumerated, but it is the same union, so decide() keeps
        // the conjunction that already occupies that coordinate subset.
        for i in 0..n {
            for j in (i + 1)..n {
                probes.push((vec![i, j], self.union_of(&[i, j])));
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    probes.push((vec![i, j, k], self.union_of(&[i, j, k])));
                }
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    probes.push((vec![i, j, k], self.union_of(&[i, j, k])));
                }
            }
        }

        let mut solvers = Vec::new();
        for (components, coords) in probes {
            if coords.is_empty() {
                continue;
            }
            if let Some(attempt) = try_subset(
                task,
                &coords,
                None,
                None,
                ProcedureKind::Composed,
                components,
            ) {
                solvers.push(attempt);
            }
        }
        decide(solvers)
    }

    fn union_of(&self, ids: &[usize]) -> Vec<usize> {
        let parts: Vec<&[usize]> = ids
            .iter()
            .map(|&id| self.procedures[id].coords.as_slice())
            .collect();
        union_coords(&parts)
    }

    /// Remaining family members that are not stored, plus one-coordinate drops
    /// of stored subsets when that smaller subset is not stored. Both stay
    /// inside the bounded family. Order is cardinality, then lexicographic.
    fn stage3(&self, task: &Task, d: usize) -> StageOutcome {
        let mut generated = Vec::new();
        for coords in bounded_family(d) {
            if !self.subset_stored(&coords) {
                generated.push(coords);
            }
        }
        for procedure in &self.procedures {
            if procedure.coords.len() < 2 {
                continue;
            }
            for drop_at in 0..procedure.coords.len() {
                let mut smaller = procedure.coords.clone();
                smaller.remove(drop_at);
                if smaller.is_empty() || smaller.iter().any(|&c| c >= d) {
                    continue;
                }
                if self.subset_stored(&smaller) || generated.iter().any(|have| have == &smaller) {
                    continue;
                }
                if !bounded_family(d).iter().any(|member| member == &smaller) {
                    continue;
                }
                generated.push(smaller);
            }
        }
        generated.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
        generated.dedup();

        let mut solvers = Vec::new();
        for coords in generated {
            if let Some(attempt) = try_subset(
                task,
                &coords,
                None,
                None,
                ProcedureKind::Generated,
                Vec::new(),
            ) {
                solvers.push(attempt);
            }
        }
        decide(solvers)
    }

    fn commit(&mut self, attempt: Attempt, stage: u8) -> SolveResult {
        let procedure_id = if let Some(id) = attempt.existing_id {
            let procedure = &mut self.procedures[id];
            procedure.success_count += 1;
            if procedure.success_count >= STABILITY_TRACES {
                procedure.stable = true;
            }
            id
        } else {
            let id = self.procedures.len();
            debug_assert!(attempt.coords.windows(2).all(|pair| pair[0] < pair[1]));
            self.procedures.push(Procedure {
                id,
                coords: attempt.coords.clone(),
                pattern: attempt.pattern.clone(),
                kind: attempt.kind,
                components: attempt.components.clone(),
                success_count: 1,
                stable: false,
            });
            if !self
                .concepts
                .iter()
                .any(|concept| concept.coords == attempt.coords)
            {
                self.concepts.push(Concept {
                    coords: attempt.coords.clone(),
                    success_count: 0,
                    stable: false,
                });
            }
            id
        };

        if let Some(concept) = self
            .concepts
            .iter_mut()
            .find(|concept| concept.coords == attempt.coords)
        {
            concept.success_count += 1;
            if concept.success_count >= STABILITY_TRACES {
                concept.stable = true;
            }
        }

        self.traces.push(Trace {
            procedure_id,
            stage,
            candidate: attempt.candidate,
        });
        let stable = self.procedures[procedure_id].stable;
        SolveResult {
            status: Status::Solved,
            stage,
            candidate: Some(attempt.candidate),
            procedure_id: Some(procedure_id),
            stable,
        }
    }
}

impl Default for KnowledgeSpace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_primitive_compares_feature_values() {
        assert!(equal(&"red".to_string(), &"red".to_string()));
        assert!(!equal(&"red".to_string(), &"blue".to_string()));
        assert!(not_equal(&"red", &"blue"));
        assert!(and(true, true));
        assert!(!and(true, false));
    }

    #[test]
    fn bounded_family_is_cardinality_then_lex_and_rejects_d_above_3() {
        assert_eq!(
            bounded_family(3),
            vec![
                vec![0],
                vec![1],
                vec![2],
                vec![0, 1],
                vec![0, 2],
                vec![1, 2],
                vec![0, 1, 2],
            ]
        );
        assert!(bounded_family(4).is_empty());
        assert!(bounded_family(0).is_empty());
    }

    #[test]
    fn k0_has_primitives_and_no_classifiers() {
        let space = KnowledgeSpace::new();
        let names: Vec<&str> = space.primitives().iter().map(|p| p.name).collect();
        assert_eq!(names, PRIMITIVE_NAMES);
        assert!(space.procedures().is_empty());
        assert!(space.traces().is_empty());
        assert!(space.concepts().is_empty());
    }
}
