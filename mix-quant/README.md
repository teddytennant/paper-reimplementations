Micro-Inverted-Scaling (MiX) software format kernel (Liao & Seo, arXiv:2609.19683).

```
cargo test
```

What does not match the paper: this crate is the software format kernel only — no CMOS accelerator, systolic array, area/power numbers, or VLM end-to-end accuracy. The 4-bit Emax field uses bias 8 (represented range [−8, 7]); the paper never specifies the stored encoding. Δe = 7 encodes zero on both signs (asymmetric −0 reclaim in §IV-C is not implemented). Exponent finetuning is Algorithm 1’s software ±1 search, not the hardware 0.5-mantissa threshold. The MXINT4 baseline is a minimal shared-exponent sign-magnitude integer, not a full OCP MX packer.
