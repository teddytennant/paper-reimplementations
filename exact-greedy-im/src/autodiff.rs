//! Tiny reverse-mode tape over additions and multiplications (paper §4.3).

#[derive(Clone, Copy, Debug)]
enum Op {
    Const,
    Param(usize),
    Add(usize, usize),
    Mul(usize, usize),
    OneMinus(usize),
}

#[derive(Clone, Debug)]
struct Node {
    value: f64,
    op: Op,
    adj: f64,
}

/// Arithmetic tape. Every `+` and `*` records its parents so a reverse sweep
/// from `F` yields `∂F/∂θ_v` for all source-edge probabilities at once.
#[derive(Clone, Debug)]
pub struct Tape {
    nodes: Vec<Node>,
    n_params: usize,
    param_nodes: Vec<Option<usize>>,
}

impl Tape {
    pub fn new(n_params: usize) -> Self {
        Self {
            nodes: Vec::new(),
            n_params,
            param_nodes: vec![None; n_params],
        }
    }

    pub fn constant(&mut self, value: f64) -> usize {
        self.nodes.push(Node {
            value,
            op: Op::Const,
            adj: 0.0,
        });
        self.nodes.len() - 1
    }

    /// Parameter `θ_i`. Reused if already created, so all uses share one node.
    pub fn param(&mut self, i: usize, value: f64) -> usize {
        if let Some(id) = self.param_nodes[i] {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            value,
            op: Op::Param(i),
            adj: 0.0,
        });
        self.param_nodes[i] = Some(id);
        id
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let value = self.nodes[a].value + self.nodes[b].value;
        self.nodes.push(Node {
            value,
            op: Op::Add(a, b),
            adj: 0.0,
        });
        self.nodes.len() - 1
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let value = self.nodes[a].value * self.nodes[b].value;
        self.nodes.push(Node {
            value,
            op: Op::Mul(a, b),
            adj: 0.0,
        });
        self.nodes.len() - 1
    }

    /// `1 - x`. Used for absent source-edge branches `p(1 - θ_v)`.
    pub fn one_minus(&mut self, a: usize) -> usize {
        let value = 1.0 - self.nodes[a].value;
        self.nodes.push(Node {
            value,
            op: Op::OneMinus(a),
            adj: 0.0,
        });
        self.nodes.len() - 1
    }

    pub fn value(&self, id: usize) -> f64 {
        self.nodes[id].value
    }

    /// Reverse-mode from scalar `output`. Returns `∂output/∂θ_i`.
    pub fn reverse(&mut self, output: usize) -> Vec<f64> {
        for node in &mut self.nodes {
            node.adj = 0.0;
        }
        self.nodes[output].adj = 1.0;
        for i in (0..self.nodes.len()).rev() {
            let adj = self.nodes[i].adj;
            match self.nodes[i].op {
                Op::Const | Op::Param(_) => {}
                Op::Add(a, b) => {
                    self.nodes[a].adj += adj;
                    self.nodes[b].adj += adj;
                }
                Op::Mul(a, b) => {
                    let va = self.nodes[a].value;
                    let vb = self.nodes[b].value;
                    self.nodes[a].adj += adj * vb;
                    self.nodes[b].adj += adj * va;
                }
                Op::OneMinus(a) => {
                    self.nodes[a].adj -= adj;
                }
            }
        }
        let mut grad = vec![0.0; self.n_params];
        for node in &self.nodes {
            if let Op::Param(i) = node.op {
                grad[i] = node.adj;
            }
        }
        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_gradient() {
        // F = θ0 * (1 - θ1) + 2 * θ1 at (0, 0): ∂F/∂θ0 = 1, ∂F/∂θ1 = 2.
        let mut t = Tape::new(2);
        let th0 = t.param(0, 0.0);
        let th1 = t.param(1, 0.0);
        let one_m = t.one_minus(th1);
        let a = t.mul(th0, one_m);
        let two = t.constant(2.0);
        let b = t.mul(two, th1);
        let f = t.add(a, b);
        let g = t.reverse(f);
        assert!((t.value(f) - 0.0).abs() < 1e-15);
        assert!((g[0] - 1.0).abs() < 1e-15);
        assert!((g[1] - 2.0).abs() < 1e-15);
    }
}
