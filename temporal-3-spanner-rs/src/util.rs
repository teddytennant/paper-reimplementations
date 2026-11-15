//! Small set helpers shared by the two constructions.

/// Elements of `all` that are not in `remove`, keeping the order of `all`.
pub(crate) fn difference(all: &[usize], remove: &[usize]) -> Vec<usize> {
    let mut sorted = remove.to_vec();
    sorted.sort_unstable();
    all.iter()
        .copied()
        .filter(|x| sorted.binary_search(x).is_err())
        .collect()
}
