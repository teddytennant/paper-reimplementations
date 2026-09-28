"""Q-DEQ: discrete local QUBO solving and W8A8 fake quant for DEQs."""

from q_deq._host import preload as _preload

_preload()

from q_deq.deq import (
    apply_F,
    count_params,
    head,
    init_head,
    init_shared_params,
    init_stacked_params,
    residual,
    shared_layer,
    stacked_apply,
)
from q_deq.implicit import implicit_adjoint
from q_deq.quantize import (
    quantize_activation,
    quantize_weight,
    quantized_shared_layer,
    ste_round,
)
from q_deq.qubo import (
    BITS,
    binary_encoding_matrix,
    candidate_directions,
    directional_p,
    encode_qubo,
    finite_diff_p,
    gram_h,
    merge_qubo_diagonal,
    qubo_to_ising,
)
from q_deq.sa import sa_solve
from q_deq.solver import (
    anderson_solve,
    qdeq_solve,
    training_forward,
)

__all__ = [
    "BITS",
    "anderson_solve",
    "apply_F",
    "binary_encoding_matrix",
    "candidate_directions",
    "count_params",
    "directional_p",
    "encode_qubo",
    "finite_diff_p",
    "gram_h",
    "head",
    "implicit_adjoint",
    "init_head",
    "init_shared_params",
    "init_stacked_params",
    "merge_qubo_diagonal",
    "qdeq_solve",
    "quantize_activation",
    "quantize_weight",
    "quantized_shared_layer",
    "qubo_to_ising",
    "residual",
    "sa_solve",
    "shared_layer",
    "stacked_apply",
    "ste_round",
    "training_forward",
]
