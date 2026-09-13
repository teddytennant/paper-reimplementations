"""RiVaT-Fuse variational tensor fusion kernel (arXiv:2609.10798)."""

from rivat_fuse.heads import (
    init_quadratic_head,
    quadratic_form,
    quadratic_head,
    task_coupling,
)
from rivat_fuse.interaction import (
    init_interaction_params,
    interaction,
    relational_weights,
    t_add,
    t_mult,
    t_rel,
)
from rivat_fuse.model import fuse, fuse_one, init_params, predict
from rivat_fuse.solve import assemble_H, assemble_b, quadratic_objective, solve_latent
from rivat_fuse.trust import (
    EPS_DEFAULT,
    apply_P,
    apply_P_from_side,
    dense_P,
    dense_P_from_side,
    init_trust_params,
    maps_d_g,
    softplus_pos,
)

__all__ = [
    "EPS_DEFAULT",
    "apply_P",
    "apply_P_from_side",
    "assemble_H",
    "assemble_b",
    "dense_P",
    "dense_P_from_side",
    "fuse",
    "fuse_one",
    "init_interaction_params",
    "init_params",
    "init_quadratic_head",
    "init_trust_params",
    "interaction",
    "maps_d_g",
    "predict",
    "quadratic_form",
    "quadratic_head",
    "quadratic_objective",
    "relational_weights",
    "softplus_pos",
    "solve_latent",
    "t_add",
    "t_mult",
    "t_rel",
    "task_coupling",
]
