use cnf_mce::{
    boundary_penalty, build_vocabulary, decode, decode_bytes, encode, encode_bytes, is_valid_span,
    CnfConfig, Vocab,
};

fn cfg(intervals: Vec<(usize, usize)>, thresholds: Vec<u64>, raw: usize, k: usize) -> CnfConfig {
    CnfConfig {
        length_intervals: intervals,
        thresholds,
        raw_vocab_cap: raw,
        vocab_size: k,
        lambda: 0.05,
        eta: 0.0,
    }
}

fn tokens_of(ids: &[u32], vocab: &Vocab) -> Vec<Vec<u8>> {
    ids.iter()
        .map(|&id| vocab.token(id).expect("id").to_vec())
        .collect()
}

#[test]
fn paper_example_validity() {
    let text = "1600 B.C. was a significant year in human history.";
    assert_eq!(&text.as_bytes()[1..2], b"6");
    assert_eq!(&text.as_bytes()[15..19], b" sig");
    assert_eq!(&text.as_bytes()[16..27], b"significant");
    assert_eq!(&text.as_bytes()[23..27], b"cant");
    assert_eq!(&text.as_bytes()[27..32], b" year");
    assert_eq!(&text.as_bytes()[9..15], b" was a");
    assert_eq!(&text.as_bytes()[16..32], b"significant year");
    assert_eq!(&text.as_bytes()[21..32], b"ficant year");

    assert!(is_valid_span(text, 1, 2));
    assert!(is_valid_span(text, 15, 19));
    assert!(is_valid_span(text, 16, 27));
    assert!(is_valid_span(text, 23, 27));
    assert!(is_valid_span(text, 27, 32));
    assert!(is_valid_span(text, 9, 15));
    assert!(is_valid_span(text, 16, 32));
    assert!(!is_valid_span(text, 21, 32));

    // "ficant " has already left the word, so it is invalid too.
    assert_eq!(&text.as_bytes()[21..28], b"ficant ");
    assert!(!is_valid_span(text, 21, 28));

    // A start inside the word that stays inside the word is valid.
    assert!(is_valid_span(text, 21, 27));
}

#[test]
fn co_advisor_penalty() {
    let text = "co-advisor";
    let lambda = 0.05;
    assert_eq!(boundary_penalty(text, 0, text.len(), lambda), Some(0.0));
    assert_eq!(&text.as_bytes()[1..3], b"o-");
    assert_eq!(boundary_penalty(text, 1, 3, lambda), Some(lambda));
    assert_eq!(&text.as_bytes()[1..5], b"o-ad");
    assert_eq!(boundary_penalty(text, 1, 5, lambda), None);
}

#[test]
fn mce_matches_brute_force_and_largest_j() {
    let vocab = Vocab::from_tokens([b"ab".as_slice(), b"bc".as_slice(), b"a", b"c"]);
    let text = "abc";
    let ids = encode(text, &vocab, 0.05, 0.0);
    assert_eq!(
        tokens_of(&ids, &vocab),
        vec![b"ab".to_vec(), b"c".to_vec()],
        "equal cost must keep the larger first cut"
    );
    assert_eq!(brute(text, &vocab, 0.05, 0.0), tokens_of(&ids, &vocab));

    // Three equal-cost cuts from the same start. Largest j wins, then the
    // suffix is solved the same way.
    let vocab = Vocab::from_tokens([b"abc".as_slice(), b"ab", b"cd", b"bcd", b"a", b"d", b"c"]);
    let text = "abcd";
    let ids = encode(text, &vocab, 0.0, 0.0);
    assert_eq!(
        tokens_of(&ids, &vocab),
        vec![b"abc".to_vec(), b"d".to_vec()]
    );
    assert_eq!(brute(text, &vocab, 0.0, 0.0), tokens_of(&ids, &vocab));

    let vocab = Vocab::from_tokens([
        b"co-".as_slice(),
        b"-advisor",
        b"co",
        b"advisor",
        b"o-ad",
        b"o-",
    ]);
    let text = "co-advisor";
    let ids = encode(text, &vocab, 0.05, 0.0);
    let got = tokens_of(&ids, &vocab);
    assert_eq!(got, brute(text, &vocab, 0.05, 0.0));
    assert!(!got.iter().any(|t| t.as_slice() == b"o-ad"));
    assert_eq!(decode(&ids, &vocab), text);
}

#[test]
fn byte_strings_round_trip() {
    let vocab = Vocab::from_tokens([b"ab".as_slice(), b"hello"]);
    let samples: &[&[u8]] = &[
        b"",
        b"a",
        b"hello",
        b"\xff\xfe\x00\x80",
        b"\xc3\x28",
        b"hello\xffworld",
        "héllo 世界".as_bytes(),
        b"\xff",
    ];
    for bytes in samples {
        let ids = encode_bytes(bytes, &vocab, 0.05, 0.0);
        assert_eq!(decode_bytes(&ids, &vocab), *bytes);
    }

    let text = "1600 B.C. was a significant year in human history.";
    let built = build_vocabulary(&[text], &cfg(vec![(1, 8), (9, 16)], vec![0, 0], 1000, 400));
    assert_eq!(decode(&encode(text, &built, 0.05, 0.0), &built), text);

    let weird = [0xff, 0xfe, 0x00, 0x80, 0xc3, 0x28, b'a'];
    assert_eq!(
        decode_bytes(&encode_bytes(&weird, &built, 0.05, 1.0), &built),
        weird
    );
}

#[test]
fn filtering_drops_unused_high_count_span() {
    // "ab" occurs as often as "abc" in the raw text, but MCE emits "abc".
    // "zz" is rarer as a substring and is actually emitted.
    let corpus = ["abc", "abc", "abc", "abc", "abc", "zz"];
    let vocab = build_vocabulary(&corpus, &cfg(vec![(1, 3)], vec![0], 100, 258));
    assert!(vocab.contains(b"abc"), "used token kept");
    assert!(vocab.contains(b"zz"), "rarer used token kept");
    assert!(!vocab.contains(b"ab"), "high raw count but never emitted");
    assert!(!vocab.contains(b"bc"));
    assert!(vocab.get_id(b"abc").unwrap() < vocab.get_id(b"zz").unwrap());
    assert_eq!(vocab.len(), 258);
    assert_eq!(vocab.get_id(&[0x00]), Some(2));
    assert_eq!(vocab.get_id(&[0xff]), Some(257));
}

#[test]
fn early_stop_blocks_longer_extension() {
    let corpus = ["abcd", "abcd", "abcd"];
    let blocked = build_vocabulary(&corpus, &cfg(vec![(1, 2), (3, 4)], vec![4, 1], 100, 300));
    assert!(
        !blocked.contains(b"abcd"),
        "round-1 prefix below threshold must stop the round-2 extension"
    );
    assert!(!blocked.contains(b"abc"));
    assert!(!blocked.contains(b"ab"));

    let kept = build_vocabulary(&corpus, &cfg(vec![(1, 2), (3, 4)], vec![1, 1], 100, 300));
    assert!(
        kept.contains(b"abcd"),
        "the extension is frequent enough when its prefix is retained"
    );
}

#[test]
fn eta_prefers_smaller_id_on_tie() {
    // ids: "a"=0, "bc"=1, "ab"=2, "c"=3, then the other bytes.
    // 1+p ties between "ab"+"c" and "a"+"bc". eta = 0 keeps the larger cut.
    // eta > 0 prefers the smaller id sum.
    let vocab = Vocab::from_tokens([b"a".as_slice(), b"bc", b"ab", b"c"]);
    let without = tokens_of(&encode("abc", &vocab, 0.05, 0.0), &vocab);
    assert_eq!(without, vec![b"ab".to_vec(), b"c".to_vec()]);
    let with = tokens_of(&encode("abc", &vocab, 0.05, 1.0), &vocab);
    assert_eq!(with, vec![b"a".to_vec(), b"bc".to_vec()]);
}

#[test]
fn combining_marks_join_letters_and_cjk_are_singletons() {
    let text = "e\u{0301} x";
    // mark + space starts inside the letter+mark unit and leaves it.
    assert!(!is_valid_span(text, 1, 4));
    assert_eq!(boundary_penalty(text, 0, 3, 0.05), Some(0.0));

    let cjk = "中文";
    // One byte into the second ideograph crosses a unit boundary incompletely.
    assert_eq!(boundary_penalty(cjk, 0, 4, 0.05), Some(0.05));
    assert!(is_valid_span(cjk, 0, 6));
    assert!(is_valid_span(cjk, 3, 6));
}

#[test]
fn mid_codepoint_end_does_not_block_the_full_character() {
    let corpus = ["é", "é"];
    let vocab = build_vocabulary(&corpus, &cfg(vec![(1, 1), (2, 2)], vec![1, 1], 20, 300));
    assert!(vocab.contains("é".as_bytes()));
}

#[test]
fn top_m_prefers_longer_then_lexicographic() {
    // Every substring occurs once, so the cap is settled by length, then bytes.
    let longer = build_vocabulary(&["abc", "xy"], &cfg(vec![(1, 3)], vec![0], 1, 257));
    assert!(longer.contains(b"abc"));
    assert!(!longer.contains(b"xy"));

    let lex = build_vocabulary(&["cd", "ab"], &cfg(vec![(1, 2)], vec![0], 1, 257));
    assert!(lex.contains(b"ab"));
    assert!(!lex.contains(b"cd"));
}

#[test]
fn small_vocab_size_still_keeps_all_bytes() {
    let vocab = build_vocabulary(&["hello"], &cfg(vec![(1, 5)], vec![0], 10, 8));
    assert!(vocab.recorded_vocab_size_below_256);
    assert_eq!(vocab.requested_vocab_size, 8);
    assert_eq!(vocab.len(), 256);
    assert!(!vocab.contains(b"hello"));
    assert_eq!(decode(&encode("hello", &vocab, 0.05, 0.0), &vocab), "hello");
}

fn brute(text: &str, vocab: &Vocab, lambda: f64, eta: f64) -> Vec<Vec<u8>> {
    let n = text.len();
    let bytes = text.as_bytes();
    let vocab_len = vocab.len() as f64;
    let mut best_cost = f64::INFINITY;
    let mut best_cuts: Vec<usize> = Vec::new();
    let mut cuts = vec![0usize];
    fn rec(
        text: &str,
        bytes: &[u8],
        vocab: &Vocab,
        lambda: f64,
        eta: f64,
        vocab_len: f64,
        i: usize,
        n: usize,
        cost: f64,
        cuts: &mut Vec<usize>,
        best_cost: &mut f64,
        best_cuts: &mut Vec<usize>,
    ) {
        if i == n {
            let better = cost < *best_cost || (cost == *best_cost && cuts > best_cuts);
            if better {
                *best_cost = cost;
                *best_cuts = cuts.clone();
            }
            return;
        }
        for j in (i + 1)..=n {
            let Some(id) = vocab.get_id(&bytes[i..j]) else {
                continue;
            };
            let Some(p) = boundary_penalty(text, i, j, lambda) else {
                continue;
            };
            let c = 1.0 + p + eta * (id as f64) / vocab_len;
            cuts.push(j);
            rec(
                text,
                bytes,
                vocab,
                lambda,
                eta,
                vocab_len,
                j,
                n,
                cost + c,
                cuts,
                best_cost,
                best_cuts,
            );
            cuts.pop();
        }
    }
    rec(
        text,
        bytes,
        vocab,
        lambda,
        eta,
        vocab_len,
        0,
        n,
        0.0,
        &mut cuts,
        &mut best_cost,
        &mut best_cuts,
    );
    let mut out = Vec::new();
    for w in best_cuts.windows(2) {
        out.push(bytes[w[0]..w[1]].to_vec());
    }
    out
}
