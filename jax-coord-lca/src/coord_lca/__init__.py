"""CoordFormer coordinate decoder and localized cross-attention (arXiv:2609.08660)."""

from coord_lca.coords import (
    FOURIER_OUT_MULTIPLIER,
    dense_query_coords,
    fourier_embed,
    fourier_frequencies,
    image_to_encoder_grid,
    normalize_coords,
)
from coord_lca.decoder import (
    binary_cross_entropy,
    build_queries,
    coord_loss,
    cross_entropy,
    decode_dense,
    decode_sparse,
    fuse_query,
    init_decoder,
    soft_dice,
)
from coord_lca.encoder import encode, init_encoder, spatial_size
from coord_lca.lca import (
    bilinear_neighborhood,
    bilinear_sample_tokens,
    cross_attend,
    init_lca,
    localized_cross_attention,
)
from coord_lca.patch import (
    embed_patches,
    extract_patches,
    init_patch_embed,
    patch_and_coord_features,
)

__all__ = [
    "FOURIER_OUT_MULTIPLIER",
    "binary_cross_entropy",
    "bilinear_neighborhood",
    "bilinear_sample_tokens",
    "build_queries",
    "coord_loss",
    "cross_attend",
    "cross_entropy",
    "decode_dense",
    "decode_sparse",
    "dense_query_coords",
    "embed_patches",
    "encode",
    "extract_patches",
    "fourier_embed",
    "fourier_frequencies",
    "fuse_query",
    "image_to_encoder_grid",
    "init_decoder",
    "init_encoder",
    "init_lca",
    "init_patch_embed",
    "localized_cross_attention",
    "normalize_coords",
    "patch_and_coord_features",
    "soft_dice",
    "spatial_size",
]
