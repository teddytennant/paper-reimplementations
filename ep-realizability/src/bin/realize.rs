use ep_realizability::{implied_edges, realizable, realize, AhMatrix};

fn main() {
    // 2-cell exact EP of C_6: AH = [[0, 2], [2, 0]], H = [3, 3].
    let ah = AhMatrix::new(vec![vec![0, 2], vec![2, 0]]).expect("square");
    let h = vec![3usize, 3];
    let ok = realizable(&ah, &h);
    println!("realizable: {ok}");
    match realize(&ah, &h) {
        Ok(g) => println!("edges: {}", g.m()),
        Err(e) => {
            let m = implied_edges(&ah, &h).unwrap_or(0);
            println!("not realized ({e}); implied edges: {m}");
        }
    }
}
