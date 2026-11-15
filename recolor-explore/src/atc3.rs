//! Algorithm 1: ATC3, three-color exploration of trees and simple cycles.

use crate::sim::{Dest, Unreachable};

/// `C_TC3 = {init, ℓ0, ℓ1}` with `c0 = init`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorTC3 {
    Init,
    L0,
    L1,
}

/// Named rules of Algorithm 1, in paper order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleTC3 {
    D1,
    D2,
    D3,
    D4,
    D5,
    D6,
    D7,
    D8,
    D9,
    D10,
    D11,
}

/// Rule function `ξ_TC3(c, M)`.
///
/// Case order matches Algorithm 1. Observations with no listed output
/// return `Err(Unreachable)`.
pub fn atc3(
    c: ColorTC3,
    m: &[ColorTC3],
) -> Result<(ColorTC3, Dest<ColorTC3>, RuleTC3), Unreachable> {
    use ColorTC3::*;
    use Dest::*;
    let has_init = m.iter().any(|&x| x == Init);
    let has_l0 = m.iter().any(|&x| x == L0);
    let has_l1 = m.iter().any(|&x| x == L1);
    match c {
        Init => {
            if !has_init && !has_l0 && !has_l1 {
                Ok((L1, Stop, RuleTC3::D1))
            } else if has_init && !has_l0 && !has_l1 {
                Ok((L0, Neighbor(Init), RuleTC3::D2))
            } else if has_init && has_l0 {
                Ok((L1, Neighbor(Init), RuleTC3::D3))
            } else if has_init && has_l1 {
                Ok((Init, Neighbor(L1), RuleTC3::D4))
            } else if has_l1 {
                Ok((L1, Neighbor(L1), RuleTC3::D5))
            } else if has_l0 {
                Ok((L1, Neighbor(L0), RuleTC3::D6))
            } else {
                Err(Unreachable)
            }
        }
        L0 => {
            if has_init && has_l1 {
                Ok((L0, Neighbor(Init), RuleTC3::D7))
            } else if has_l0 && has_l1 {
                Ok((L1, Neighbor(L0), RuleTC3::D8))
            } else if has_l1 && !has_init && !has_l0 {
                Ok((L1, Stop, RuleTC3::D9))
            } else {
                Err(Unreachable)
            }
        }
        L1 => {
            if has_init && has_l0 {
                Ok((L0, Neighbor(Init), RuleTC3::D10))
            } else if has_l0 && has_l1 {
                Ok((L1, Neighbor(L0), RuleTC3::D11))
            } else {
                Err(Unreachable)
            }
        }
    }
}
