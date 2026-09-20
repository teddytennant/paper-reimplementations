"""4-token grammar tasks from arXiv:2609.18565 §4.

Vocabulary V = {A, B, C, space}. Deterministic language (AAA BBB CCC)* and
lexicographic language {abc ∈ Σ³ | a ≤ b ≤ c}. Training texts have 25 words,
test texts 40 words.
"""

from __future__ import annotations

import numpy as np

TOKEN_TO_ID = {"A": 0, "B": 1, "C": 2, " ": 3}
ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}
VOCAB = ("A", "B", "C", " ")
DET_CYCLE = ("AAA", "BBB", "CCC")
# {abc ∈ {A,B,C}³ | a ≤ b ≤ c} in lexicographic order of the generating loops.
LEX_WORDS = (
    "AAA",
    "AAB",
    "AAC",
    "ABB",
    "ABC",
    "ACC",
    "BBB",
    "BBC",
    "BCC",
    "CCC",
)
N_TRAIN_WORDS = 25
N_TEST_WORDS = 40
CONTEXT = 4


def encode_text(text: str) -> np.ndarray:
    return np.array([TOKEN_TO_ID[c] for c in text], dtype=np.int32)


def decode_tokens(tokens) -> str:
    return "".join(ID_TO_TOKEN[int(t)] for t in tokens)


def deterministic_text(n_words: int) -> str:
    words = [DET_CYCLE[i % 3] for i in range(n_words)]
    return " ".join(words)


def lexicographic_text(n_words: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(LEX_WORDS), size=n_words)
    words = [LEX_WORDS[int(i)] for i in idx]
    return " ".join(words)


def make_dataset(task: str, seed: int = 17):
    """Return train/test token arrays.

    For the lexicographic task the paper samples the training text from `seed`
    and the held-out test text from the following seed.
    """
    if task == "deterministic":
        train = encode_text(deterministic_text(N_TRAIN_WORDS))
        test = encode_text(deterministic_text(N_TEST_WORDS))
    elif task == "lexicographic":
        train = encode_text(lexicographic_text(N_TRAIN_WORDS, seed))
        test = encode_text(lexicographic_text(N_TEST_WORDS, seed + 1))
    else:
        raise ValueError(f"unknown task {task!r}")
    return train, test


def windows(tokens, context: int = CONTEXT):
    tokens = np.asarray(tokens, dtype=np.int32)
    n = len(tokens) - context
    x = np.stack([tokens[i : i + context] for i in range(n)])
    y = tokens[context:]
    return x, y


def valid_token_rate(tokens) -> float:
    tokens = np.asarray(tokens)
    if tokens.size == 0:
        return 0.0
    return float(np.mean((tokens >= 0) & (tokens < 4)))


def separator_position_rate(tokens) -> float:
    """Spaces belong at indices 3, 7, 11, … of a concatenated 3-letter word stream."""
    tokens = np.asarray(tokens)
    if tokens.size == 0:
        return 0.0
    idx = np.arange(tokens.size)
    should_space = idx % 4 == 3
    correct = np.where(should_space, tokens == 3, tokens != 3)
    return float(np.mean(correct))


def word_length_validity(tokens) -> float:
    words = decode_tokens(tokens).split(" ")
    if not words:
        return 0.0
    return float(np.mean([len(w) == 3 for w in words]))


def deterministic_cycle_validity(tokens) -> float:
    words = [w for w in decode_tokens(tokens).split(" ") if w]
    if not words:
        return 0.0
    best = 0.0
    for phase in range(3):
        ok = sum(w == DET_CYCLE[(phase + i) % 3] for i, w in enumerate(words))
        best = max(best, ok / len(words))
    return float(best)


def lexicographic_word_validity(tokens) -> float:
    words = [w for w in decode_tokens(tokens).split(" ") if len(w) == 3]
    if not words:
        return 0.0

    def ok(w: str) -> bool:
        return all(c in "ABC" for c in w) and w[0] <= w[1] <= w[2]

    return float(np.mean([ok(w) for w in words]))


def grammar_score(tokens, task: str = "deterministic") -> float:
    """Average of four components as described in §4."""
    comps = [
        valid_token_rate(tokens),
        separator_position_rate(tokens),
        word_length_validity(tokens),
    ]
    if task == "deterministic":
        comps.append(deterministic_cycle_validity(tokens))
    elif task == "lexicographic":
        comps.append(lexicographic_word_validity(tokens))
    else:
        raise ValueError(f"unknown task {task!r}")
    return float(np.mean(comps))


def token_accuracy(pred, true) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)
    n = min(pred.size, true.size)
    if n == 0:
        return 0.0
    return float(np.mean(pred[:n] == true[:n]))
