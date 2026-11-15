"""Paper defaults from MARS-CLIP Sec. 4.2."""

# Crop size S_crop for the local grid (pixels).
S_CROP = 112

# Encoder input resolution K (pixels).
ENCODER_SIZE = 224

# Fusion weight alpha in Eq. 1 (global vs local).
ALPHA = 0.8

# Intermediate ViT layer used for B_int (ViT-B/16, 1-indexed).
INTERMEDIATE_LAYER = 8
