use prob_ref_f4::{
    certified_ref, default_block_size, proposition1_lower_bound, success_probability, Fp, Matrix,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let field = Fp::P32003;
    let rows = 12usize;
    let cols = 15usize;
    let s = 2usize;
    let mut rng = StdRng::seed_from_u64(2609_14433);
    let a = Matrix::random(rows, cols, field, &mut rng);
    let ell = default_block_size(rows);
    let l = certified_ref(&a, s, &mut rng);
    let rank = l.rank();
    let q = field.modulus();
    let p_c3 = success_probability(q, rank, s);
    let t = (rows + ell - 1) / ell;
    let p_prop1 = proposition1_lower_bound(q, ell, s, t);
    println!(
        "prime={} {}x{}  rank={}  ell={} s={}",
        q, rows, cols, rank, ell, s
    );
    println!("corollary3_success_prob={:.12}", p_c3);
    println!("proposition1_lower_bound={:.12}", p_prop1);
}
