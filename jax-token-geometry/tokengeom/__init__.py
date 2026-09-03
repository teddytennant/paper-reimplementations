"""Numerical instantiation of Wang (2026), arXiv:2608.30072.

The paper is a theory paper: it proves that accurate token prediction under a
softmax head organizes token embeddings by the Hellinger distance between the
conditional context distributions of the token types, that the contextual
representation is a coordinate for the conditional token distribution inside the
model determined by those embeddings, and that both facts carry over into
guarantees for generation, community recovery and linear probing.

This package builds the controlled simulation of Section 5 and Appendix B at small
scale, computes every population quantity the theorems refer to by exact
enumeration, and checks the stated inequalities numerically.
"""
