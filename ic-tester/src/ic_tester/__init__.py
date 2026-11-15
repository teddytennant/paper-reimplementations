"""Query-access IC testers for single-parameter allocation mechanisms.

Faithful CPU kernel of Milionis & Pires, arXiv:2609.17406: Dodis-pair
coordinate-wise monotonicity testing (Algorithm 1) and adaptive DSIC
testing on lines and hypergrids (Algorithms 2 and 3).
"""

from ic_tester.dodis import dodis_pairs, nu2, sample_dodis_pair
from ic_tester.lines import assemble_bids, binary_search_first, eval_line
from ic_tester.prices import (
    is_monotone_1d,
    is_valid_price_1d,
    jumps,
    myerson_price_1d,
)
from ic_tester.testers import (
    ALG1_C,
    TesterResult,
    default_alg1_repeats,
    coordinatewise_monotonicity,
    dsic,
    dsic_1d,
)

__all__ = [
    "ALG1_C",
    "TesterResult",
    "assemble_bids",
    "binary_search_first",
    "coordinatewise_monotonicity",
    "default_alg1_repeats",
    "dodis_pairs",
    "dsic",
    "dsic_1d",
    "eval_line",
    "is_monotone_1d",
    "is_valid_price_1d",
    "jumps",
    "myerson_price_1d",
    "nu2",
    "sample_dodis_pair",
]
