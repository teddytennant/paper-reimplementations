use knowledge_space::{
    equal, KnowledgeSpace, Object, ProcedureKind, Status, Task, STABILITY_TRACES,
};

fn obj(vals: &[&str]) -> Object {
    vals.iter().map(|value| (*value).to_string()).collect()
}

fn task(positives: &[&[&str]], negatives: &[&[&str]], candidates: &[&[&str]]) -> Task {
    Task::new(
        positives.iter().map(|row| obj(row)).collect(),
        negatives.iter().map(|row| obj(row)).collect(),
        candidates.iter().map(|row| obj(row)).collect(),
    )
}

fn counts(space: &KnowledgeSpace) -> (usize, usize, usize) {
    (
        space.procedures().len(),
        space.traces().len(),
        space.concepts().len(),
    )
}

#[test]
fn equal_is_callable_without_a_task() {
    assert!(equal(&"red", &"red"));
    assert!(!equal(&"red", &"blue"));
}

#[test]
fn proposition_1_direct_separation() {
    let mut space = KnowledgeSpace::new();
    assert!(space.procedures().is_empty());
    assert_eq!(space.primitives().len(), 5);

    let task = task(
        &[&["red"], &["red"]],
        &[&["blue"], &["green"]],
        &[&["blue"], &["green"], &["red"]],
    );
    let result = space.solve(&task);

    assert_eq!(result.status, Status::Solved);
    assert_eq!(result.stage, 1);
    assert_eq!(result.candidate, Some(2));
    assert_eq!(result.procedure_id, Some(0));
    assert!(!result.stable);
    assert_eq!(space.procedures().len(), 1);
    assert_eq!(space.traces().len(), 1);
    assert_eq!(space.concepts().len(), 1);
    assert_eq!(space.procedures()[0].coords, vec![0]);
    assert_eq!(space.procedures()[0].pattern, obj(&["red"]));
    assert_eq!(space.procedures()[0].kind, ProcedureKind::Singleton);
    assert_eq!(space.traces()[0].candidate, 2);
}

#[test]
fn proposition_1_multi_coordinate() {
    // Only the full subset separates. A fresh space has nothing to compose,
    // so this is bounded generation; stage 2 would also satisfy the proposition.
    let mut space = KnowledgeSpace::new();
    let task = task(
        &[&["a", "b", "c"], &["a", "b", "c"]],
        &[
            &["a", "x", "y"],
            &["x", "b", "y"],
            &["x", "y", "c"],
            &["a", "b", "y"],
            &["a", "y", "c"],
            &["x", "b", "c"],
        ],
        &[&["a", "b", "y"], &["x", "y", "z"], &["a", "b", "c"]],
    );
    let result = space.solve(&task);

    assert_eq!(result.status, Status::Solved);
    assert!(
        result.stage == 2 || result.stage == 3,
        "stage {}",
        result.stage
    );
    assert_eq!(result.candidate, Some(2));
    assert_eq!(space.procedures().len(), 1);
    assert_eq!(space.traces().len(), 1);
    assert_eq!(space.procedures()[0].coords, vec![0, 1, 2]);
    assert_eq!(space.procedures()[0].pattern, obj(&["a", "b", "c"]));
}

#[test]
fn proposition_2_reuse() {
    let mut space = KnowledgeSpace::new();
    let first = task(&[&["red"], &["red"]], &[&["blue"]], &[&["blue"], &["red"]]);
    let seeded = space.solve(&first);
    assert_eq!(seeded.status, Status::Solved);
    assert_eq!(seeded.candidate, Some(1));
    let procedures_after_seed = space.procedures().len();
    let traces_after_seed = space.traces().len();

    let reuse = task(
        &[&["red"]],
        &[&["green"], &["black"]],
        &[&["black"], &["green"], &["red"]],
    );
    let result = space.solve(&reuse);

    assert_eq!(result.status, Status::Solved);
    assert_eq!(result.stage, 1);
    assert_eq!(result.candidate, Some(2));
    assert_eq!(space.procedures().len(), procedures_after_seed);
    assert_eq!(space.traces().len(), traces_after_seed + 1);
    assert_eq!(result.procedure_id, seeded.procedure_id);
}

#[test]
fn proposition_3_composition() {
    let mut space = KnowledgeSpace::new();

    let seed_left = task(
        &[&["red", "left"], &["red", "right"]],
        &[&["blue", "left"]],
        &[&["red", "up"], &["blue", "up"]],
    );
    let left = space.solve(&seed_left);
    assert_eq!(left.status, Status::Solved);
    assert_eq!(left.stage, 1);
    assert_eq!(left.candidate, Some(0));

    let seed_right = task(
        &[&["circle", "big"], &["square", "big"]],
        &[&["circle", "small"]],
        &[&["triangle", "big"], &["triangle", "small"]],
    );
    let right = space.solve(&seed_right);
    assert_eq!(right.status, Status::Solved);
    assert_eq!(right.stage, 1);
    assert_eq!(right.candidate, Some(0));
    assert_eq!(space.procedures().len(), 2);

    let both = task(
        &[&["red", "big"]],
        &[&["red", "small"], &["blue", "big"]],
        &[&["blue", "big"], &["red", "big"], &["red", "small"]],
    );
    let result = space.solve(&both);

    assert_eq!(result.status, Status::Solved);
    assert_eq!(result.stage, 2);
    assert_eq!(result.candidate, Some(1));
    assert_eq!(space.procedures().len(), 3);
    assert_eq!(space.traces().len(), 3);
    let composed = &space.procedures()[2];
    assert_eq!(composed.kind, ProcedureKind::Composed);
    assert_eq!(composed.coords, vec![0, 1]);
    assert_eq!(composed.pattern, obj(&["red", "big"]));
    assert_eq!(composed.components, vec![0, 1]);
    assert!(composed
        .components
        .iter()
        .all(|id| { space.procedures()[*id].kind == ProcedureKind::Singleton }));
}

#[test]
fn proposition_4_failure() {
    let mut space = KnowledgeSpace::new();
    let before = counts(&space);

    let disagree = task(
        &[&["red", "big"], &["blue", "small"]],
        &[&["red", "small"]],
        &[&["red", "big"], &["blue", "small"]],
    );
    let result = space.solve(&disagree);
    assert_eq!(result.status, Status::Failed);
    assert_eq!(result.stage, 0);
    assert_eq!(result.candidate, None);
    assert_eq!(result.procedure_id, None);
    assert_eq!(counts(&space), before);

    let no_match = task(&[&["red"]], &[&["blue"]], &[&["green"], &["blue"]]);
    let result = space.solve(&no_match);
    assert_eq!(result.status, Status::Failed);
    assert_eq!(result.stage, 0);
    assert_eq!(counts(&space), before);

    let too_wide = task(
        &[&["a", "b", "c", "d"]],
        &[&["x", "b", "c", "d"]],
        &[&["a", "b", "c", "d"]],
    );
    let result = space.solve(&too_wide);
    assert_eq!(result.status, Status::Failed);
    assert_eq!(result.stage, 0);
    assert_eq!(counts(&space), before);

    let mismatched = Task::new(
        vec![obj(&["red", "big"])],
        vec![obj(&["blue"])],
        vec![obj(&["red", "big"])],
    );
    let result = space.solve(&mismatched);
    assert_eq!(result.status, Status::Failed);
    assert_eq!(counts(&space), before);
}

#[test]
fn proposition_5_underdetermination() {
    let mut space = KnowledgeSpace::new();
    let before = counts(&space);
    let task = task(
        &[&["red", "big"], &["red", "big"]],
        &[&["blue", "small"]],
        &[&["red", "other"], &["other", "big"]],
    );
    let result = space.solve(&task);

    assert_eq!(result.status, Status::Underdetermined);
    assert_eq!(result.stage, 0);
    assert_eq!(result.candidate, None);
    assert_eq!(result.procedure_id, None);
    assert!(!result.stable);
    assert_eq!(counts(&space), before);
}

#[test]
fn same_candidate_is_not_underdetermined() {
    let mut space = KnowledgeSpace::new();
    let task = task(&[&["a", "b"]], &[&["c", "d"]], &[&["z", "z"], &["a", "b"]]);
    let result = space.solve(&task);

    assert_eq!(result.status, Status::Solved);
    assert_eq!(result.stage, 1);
    assert_eq!(result.candidate, Some(1));
    assert_eq!(space.procedures().len(), 1);
    assert_eq!(space.procedures()[0].coords, vec![0]);
}

#[test]
fn stored_pattern_mismatch_does_not_reuse() {
    let mut space = KnowledgeSpace::new();
    let red = task(&[&["red"]], &[&["blue"]], &[&["red"]]);
    assert_eq!(space.solve(&red).status, Status::Solved);

    let blue = task(
        &[&["blue"], &["blue"]],
        &[&["green"]],
        &[&["red"], &["blue"]],
    );
    let result = space.solve(&blue);
    assert_eq!(result.status, Status::Solved);
    assert_eq!(result.stage, 1);
    assert_eq!(result.candidate, Some(1));
    assert_eq!(space.procedures().len(), 2);
    assert_eq!(space.procedures()[1].pattern, obj(&["blue"]));
    assert_eq!(space.procedures()[1].coords, vec![0]);
}

#[test]
fn curriculum_smoke() {
    let mut space = KnowledgeSpace::new();

    let one = task(&[&["red"], &["red"]], &[&["blue"]], &[&["blue"], &["red"]]);
    let t1 = space.solve(&one);
    assert_eq!(t1.status, Status::Solved);
    assert_eq!(t1.stage, 1);
    assert_eq!(t1.candidate, Some(1));
    assert_eq!(space.procedures().len(), 1);

    let one_relevant = task(
        &[&["x", "circle"], &["y", "circle"]],
        &[&["x", "square"], &["z", "triangle"]],
        &[&["a", "square"], &["a", "circle"], &["b", "triangle"]],
    );
    let t2 = space.solve(&one_relevant);
    assert_eq!(t2.status, Status::Solved);
    assert_eq!(t2.stage, 1);
    assert_eq!(t2.candidate, Some(1));
    assert_eq!(space.procedures().len(), 2);
    assert_eq!(space.procedures()[1].coords, vec![1]);

    let both = task(
        &[&["red", "circle"]],
        &[&["red", "square"], &["blue", "circle"]],
        &[&["blue", "square"], &["red", "circle"], &["red", "square"]],
    );
    let t3 = space.solve(&both);
    assert_eq!(t3.status, Status::Solved);
    assert_eq!(t3.stage, 2);
    assert_eq!(t3.candidate, Some(1));
    assert_eq!(space.procedures().len(), 3);
    assert_eq!(space.procedures()[2].kind, ProcedureKind::Composed);
    assert!(!space.procedures()[2].stable);
    let procedures_before_reuse = space.procedures().len();
    let traces_before_reuse = space.traces().len();

    let reuse = task(
        &[&["red", "circle"], &["red", "circle"]],
        &[&["red", "triangle"], &["green", "circle"]],
        &[
            &["blue", "square"],
            &["green", "circle"],
            &["red", "circle"],
        ],
    );
    let t4 = space.solve(&reuse);
    assert_eq!(t4.status, Status::Solved);
    assert_eq!(t4.stage, 1);
    assert_eq!(t4.candidate, Some(2));
    assert_eq!(t4.procedure_id, t3.procedure_id);
    assert_eq!(space.procedures().len(), procedures_before_reuse);
    assert_eq!(space.traces().len(), traces_before_reuse + 1);
    assert!(t4.stable);
    assert!(space.procedures()[t4.procedure_id.unwrap()].stable);
    assert_eq!(
        space.procedures()[t4.procedure_id.unwrap()].success_count,
        STABILITY_TRACES
    );
    let concept = space
        .concepts()
        .iter()
        .find(|concept| concept.coords == vec![0, 1])
        .expect("class distinction for the composed subset");
    assert!(concept.stable);
}
