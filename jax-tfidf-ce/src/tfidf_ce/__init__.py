"""TF-IDF weighted cross-entropy (arXiv:2609.11029)."""

from tfidf_ce.buffer import IdfBuffer, update_df_from_batch
from tfidf_ce.loss import mean_cross_entropy, tfidf_cross_entropy
from tfidf_ce.stats import (
    batch_document_frequency,
    sequence_presence,
    smoothed_idf,
    term_frequency,
    tfidf_weights,
)

__all__ = [
    "IdfBuffer",
    "batch_document_frequency",
    "mean_cross_entropy",
    "sequence_presence",
    "smoothed_idf",
    "term_frequency",
    "tfidf_cross_entropy",
    "tfidf_weights",
    "update_df_from_batch",
]
