//! Fixed-Range Chunking and Dual-Boundary Matching (Wang, Lin, Shen, Yanaka;
//! arXiv:2609.12674).
//!
//! CPU-checkable core only: Algorithm 2 (FRC) and Algorithm 1 (alignment) with
//! a pluggable `Sim`.

pub mod align;
pub mod frc;

pub use align::{
    align_chunks, relative_position, sim_wrp, token_overlap_cosine, AlignError, AlignParams,
    Alignment, DenseSim, SourceChunk,
};
pub use frc::{block_cost, pack, span_len, Chunk, FrcParams, FrcResult, PackError};

/// Run FRC on source unit lengths, then optionally align to `n_targets` units.
pub fn pack_and_align<F>(
    source_lens: &[usize],
    n_targets: usize,
    sim: F,
    frc: &FrcParams,
    align: &AlignParams,
) -> Result<(FrcResult, Option<Alignment>), PackError>
where
    F: Fn(usize, usize) -> f64,
{
    let packed = pack(source_lens, frc)?;
    if n_targets == 0 {
        return Ok((packed, None));
    }
    let chunks: Vec<SourceChunk> = packed
        .chunks
        .iter()
        .map(|c| SourceChunk::new(c.start, c.end))
        .collect();
    let n_src = source_lens.len();
    let aln = align_chunks(n_src, n_targets, &chunks, sim, align).ok();
    Ok((packed, aln))
}
