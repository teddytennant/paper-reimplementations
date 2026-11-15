import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
