"""Grammar-task construction matching arXiv:2609.18565 §4."""

import numpy as np

from jax_vqt.data import (
    CONTEXT,
    DET_CYCLE,
    LEX_WORDS,
    N_TEST_WORDS,
    N_TRAIN_WORDS,
    VOCAB,
    decode_tokens,
    deterministic_cycle_validity,
    deterministic_text,
    encode_text,
    grammar_score,
    lexicographic_text,
    lexicographic_word_validity,
    make_dataset,
    separator_position_rate,
    token_accuracy,
    windows,
    word_length_validity,
)


def test_vocabulary_and_lex_language():
    assert VOCAB == ("A", "B", "C", " ")
    assert len(VOCAB) == 4
    assert CONTEXT == 4
    # All and only the nondecreasing length-3 words over {A,B,C}.
    expected = []
    for a in "ABC":
        for b in "ABC":
            for c in "ABC":
                if a <= b <= c:
                    expected.append(a + b + c)
    assert tuple(expected) == LEX_WORDS
    assert len(LEX_WORDS) == 10


def test_deterministic_corpus_word_counts():
    train, test = make_dataset("deterministic")
    train_s = decode_tokens(train)
    test_s = decode_tokens(test)
    assert train_s.split(" ") == [DET_CYCLE[i % 3] for i in range(N_TRAIN_WORDS)]
    assert test_s.split(" ") == [DET_CYCLE[i % 3] for i in range(N_TEST_WORDS)]
    assert len(train_s.split(" ")) == 25
    assert len(test_s.split(" ")) == 40
    # Character-level tokens, spaces between words, no trailing space.
    assert train_s == deterministic_text(25)
    assert " " in train_s
    ids = encode_text("A B")
    assert list(ids) == [0, 3, 1]


def test_lexicographic_sampling_uses_seed_then_next():
    train, test = make_dataset("lexicographic", seed=17)
    assert decode_tokens(train) == lexicographic_text(25, 17)
    assert decode_tokens(test) == lexicographic_text(40, 18)
    words = decode_tokens(train).split(" ")
    assert len(words) == 25
    assert all(w in LEX_WORDS for w in words)
    # Different seeds → different texts (with overwhelming probability).
    other, _ = make_dataset("lexicographic", seed=23)
    assert not np.array_equal(train, other)


def test_windows_are_next_token():
    tokens = encode_text("AAA BBB")
    x, y = windows(tokens, context=4)
    assert x.shape[1] == 4
    assert len(x) == len(tokens) - 4
    for i in range(len(x)):
        assert list(x[i]) == list(tokens[i : i + 4])
        assert y[i] == tokens[i + 4]


def test_perfect_deterministic_sequence_scores_one():
    tokens = encode_text(deterministic_text(10))
    assert separator_position_rate(tokens) == 1.0
    assert word_length_validity(tokens) == 1.0
    assert deterministic_cycle_validity(tokens) == 1.0
    assert grammar_score(tokens, "deterministic") == 1.0
    assert token_accuracy(tokens, tokens) == 1.0


def test_lexicographic_validity_on_language():
    tokens = encode_text(" ".join(LEX_WORDS))
    assert lexicographic_word_validity(tokens) == 1.0
    assert grammar_score(tokens, "lexicographic") == 1.0
    bad = encode_text("CBA AAA")
    assert lexicographic_word_validity(bad) < 1.0
