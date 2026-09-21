//! Algorithm 2: AC4, four-color exploration of GCB.

use crate::sim::{Dest, Unreachable};

/// `C_C4 = {init, fin, front, path}` with `c0 = init`.
/// Color `fin` is never requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorC4 {
    Init,
    Fin,
    Front,
    Path,
}

/// Named rules of Algorithm 2, in paper order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleC4 {
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,
    C8,
    C9,
    C10,
    C11,
    C12,
    C13,
    C14,
}

/// Rule function `ξ_C4(c, M)` for `c ≠ fin`.
///
/// C1 and C2 are checked first for every non-`fin` current color. Remaining
/// cases dispatch on the current color through `η_init`, `η_front`, `η_path`.
/// Case order matches Algorithm 2.
pub fn ac4(c: ColorC4, m: &[ColorC4]) -> Result<(ColorC4, Dest<ColorC4>, RuleC4), Unreachable> {
    use ColorC4::*;
    use Dest::*;
    if c == Fin {
        return Err(Unreachable);
    }
    let has_init = m.iter().any(|&x| x == Init);
    let has_fin = m.iter().any(|&x| x == Fin);
    let has_front = m.iter().any(|&x| x == Front);
    let has_path = m.iter().any(|&x| x == Path);

    // C1, C2 apply for every c ≠ fin, before η_c.
    if !has_init && !has_front && !has_path {
        return Ok((Fin, Stop, RuleC4::C1));
    }
    if !has_fin && !has_front && !has_path {
        return Ok((Front, Neighbor(Init), RuleC4::C2));
    }

    match c {
        Init => eta_init(has_init, has_front, has_path),
        Front => eta_front(has_init, has_front, has_path),
        Path => eta_path(has_init, has_front),
        Fin => Err(Unreachable),
    }
}

fn eta_init(
    has_init: bool,
    has_front: bool,
    has_path: bool,
) -> Result<(ColorC4, Dest<ColorC4>, RuleC4), Unreachable> {
    use ColorC4::*;
    use Dest::*;
    if !has_front && !has_init {
        Ok((Fin, Neighbor(Path), RuleC4::C3))
    } else if !has_init {
        Ok((Fin, Neighbor(Front), RuleC4::C4))
    } else if has_front && has_path {
        Ok((Path, Neighbor(Init), RuleC4::C5))
    } else if has_path {
        Ok((Front, Neighbor(Init), RuleC4::C6))
    } else if has_front {
        Ok((Front, Neighbor(Front), RuleC4::C7))
    } else {
        // Neighbor colors exactly init and fin: paper calls this unreachable
        // at a noninitial arrival at an init vertex.
        Err(Unreachable)
    }
}

fn eta_front(
    has_init: bool,
    has_front: bool,
    _has_path: bool,
) -> Result<(ColorC4, Dest<ColorC4>, RuleC4), Unreachable> {
    use ColorC4::*;
    use Dest::*;
    if has_front {
        Ok((Path, Neighbor(Front), RuleC4::C8))
    } else if has_init {
        Ok((Front, Neighbor(Init), RuleC4::C9))
    } else {
        Ok((Fin, Neighbor(Path), RuleC4::C10))
    }
}

fn eta_path(
    has_init: bool,
    has_front: bool,
) -> Result<(ColorC4, Dest<ColorC4>, RuleC4), Unreachable> {
    use ColorC4::*;
    use Dest::*;
    if !has_front && has_init {
        Ok((Front, Neighbor(Init), RuleC4::C11))
    } else if has_front && has_init {
        Ok((Path, Neighbor(Init), RuleC4::C12))
    } else if has_front {
        Ok((Fin, Neighbor(Front), RuleC4::C13))
    } else {
        Ok((Fin, Neighbor(Path), RuleC4::C14))
    }
}
