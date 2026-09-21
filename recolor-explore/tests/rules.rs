//! Table-driven unit tests: each named rule fires on a constructed observation.

use recolor_explore::{ac4, atc3, ColorC4, ColorTC3, Dest, RuleC4, RuleTC3};

#[test]
fn atc3_named_rules() {
    use ColorTC3::*;
    use Dest::*;
    let cases: &[(ColorTC3, &[ColorTC3], ColorTC3, Dest<ColorTC3>, RuleTC3)] = &[
        (Init, &[], L1, Stop, RuleTC3::D1),
        (Init, &[Init], L0, Neighbor(Init), RuleTC3::D2),
        (Init, &[Init, Init], L0, Neighbor(Init), RuleTC3::D2),
        (Init, &[Init, L0], L1, Neighbor(Init), RuleTC3::D3),
        (Init, &[Init, L0, L1], L1, Neighbor(Init), RuleTC3::D3),
        (Init, &[Init, L1], Init, Neighbor(L1), RuleTC3::D4),
        (Init, &[L1], L1, Neighbor(L1), RuleTC3::D5),
        (Init, &[L1, L1], L1, Neighbor(L1), RuleTC3::D5),
        (Init, &[L0], L1, Neighbor(L0), RuleTC3::D6),
        (Init, &[L0, L1], L1, Neighbor(L1), RuleTC3::D5),
        (L0, &[Init, L1], L0, Neighbor(Init), RuleTC3::D7),
        (L0, &[Init, L1, L0], L0, Neighbor(Init), RuleTC3::D7),
        (L0, &[L0, L1], L1, Neighbor(L0), RuleTC3::D8),
        (L0, &[L1], L1, Stop, RuleTC3::D9),
        (L0, &[L1, L1], L1, Stop, RuleTC3::D9),
        (L1, &[Init, L0], L0, Neighbor(Init), RuleTC3::D10),
        (L1, &[Init, L0, L1], L0, Neighbor(Init), RuleTC3::D10),
        (L1, &[L0, L1], L1, Neighbor(L0), RuleTC3::D11),
    ];
    for &(c, m, want_c, want_d, want_r) in cases {
        let (got_c, got_d, got_r) = atc3(c, m)
            .unwrap_or_else(|_| panic!("ATC3 unreachable on ({c:?}, {m:?}), expected {want_r:?}"));
        assert_eq!(
            (got_c, got_d, got_r),
            (want_c, want_d, want_r),
            "ATC3 ({c:?}, {m:?})"
        );
    }
}

#[test]
fn ac4_named_rules() {
    use ColorC4::*;
    use Dest::*;
    let cases: &[(ColorC4, &[ColorC4], ColorC4, Dest<ColorC4>, RuleC4)] = &[
        // C1: init, front, path not in M (neighbors empty or only fin).
        (Init, &[], Fin, Stop, RuleC4::C1),
        (Init, &[Fin], Fin, Stop, RuleC4::C1),
        (Front, &[Fin, Fin], Fin, Stop, RuleC4::C1),
        (Path, &[], Fin, Stop, RuleC4::C1),
        // C2: fin, front, path not in M (so M is only init), after C1 fails.
        (Init, &[Init], Front, Neighbor(Init), RuleC4::C2),
        (Init, &[Init, Init], Front, Neighbor(Init), RuleC4::C2),
        (Front, &[Init], Front, Neighbor(Init), RuleC4::C2),
        (Path, &[Init, Init], Front, Neighbor(Init), RuleC4::C2),
        // η_init
        (Init, &[Path], Fin, Neighbor(Path), RuleC4::C3),
        (Init, &[Path, Fin], Fin, Neighbor(Path), RuleC4::C3),
        (Init, &[Front], Fin, Neighbor(Front), RuleC4::C4),
        (Init, &[Front, Path], Fin, Neighbor(Front), RuleC4::C4),
        (Init, &[Front, Fin], Fin, Neighbor(Front), RuleC4::C4),
        (Init, &[Front, Path, Init], Path, Neighbor(Init), RuleC4::C5),
        (Init, &[Path, Init], Front, Neighbor(Init), RuleC4::C6),
        (Init, &[Path, Init, Fin], Front, Neighbor(Init), RuleC4::C6),
        (Init, &[Front, Init], Front, Neighbor(Front), RuleC4::C7),
        (
            Init,
            &[Front, Init, Fin],
            Front,
            Neighbor(Front),
            RuleC4::C7,
        ),
        // η_front (C1/C2 already excluded)
        (Front, &[Front], Path, Neighbor(Front), RuleC4::C8),
        (Front, &[Front, Init], Path, Neighbor(Front), RuleC4::C8),
        (Front, &[Front, Path], Path, Neighbor(Front), RuleC4::C8),
        (Front, &[Init, Path], Front, Neighbor(Init), RuleC4::C9),
        (Front, &[Init, Fin], Front, Neighbor(Init), RuleC4::C9),
        (Front, &[Path], Fin, Neighbor(Path), RuleC4::C10),
        (Front, &[Path, Fin], Fin, Neighbor(Path), RuleC4::C10),
        // η_path
        (Path, &[Init, Fin], Front, Neighbor(Init), RuleC4::C11),
        (Path, &[Init, Path], Front, Neighbor(Init), RuleC4::C11),
        (Path, &[Front, Init], Path, Neighbor(Init), RuleC4::C12),
        (
            Path,
            &[Front, Init, Path],
            Path,
            Neighbor(Init),
            RuleC4::C12,
        ),
        (Path, &[Front], Fin, Neighbor(Front), RuleC4::C13),
        (Path, &[Front, Path], Fin, Neighbor(Front), RuleC4::C13),
        (Path, &[Front, Fin], Fin, Neighbor(Front), RuleC4::C13),
        (Path, &[Path], Fin, Neighbor(Path), RuleC4::C14),
        (Path, &[Path, Fin], Fin, Neighbor(Path), RuleC4::C14),
        (Path, &[Fin], Fin, Stop, RuleC4::C1),
    ];
    for &(c, m, want_c, want_d, want_r) in cases {
        let (got_c, got_d, got_r) = ac4(c, m)
            .unwrap_or_else(|_| panic!("AC4 unreachable on ({c:?}, {m:?}), expected {want_r:?}"));
        assert_eq!(
            (got_c, got_d, got_r),
            (want_c, want_d, want_r),
            "AC4 ({c:?}, {m:?})"
        );
    }
}

#[test]
fn ac4_fin_current_is_unreachable() {
    assert!(ac4(ColorC4::Fin, &[]).is_err());
    assert!(ac4(ColorC4::Fin, &[ColorC4::Init]).is_err());
}

#[test]
fn ac4_init_with_only_init_and_fin_is_unreachable() {
    use ColorC4::*;
    assert!(ac4(Init, &[Init, Fin]).is_err());
    assert!(ac4(Init, &[Fin, Init, Init]).is_err());
}
