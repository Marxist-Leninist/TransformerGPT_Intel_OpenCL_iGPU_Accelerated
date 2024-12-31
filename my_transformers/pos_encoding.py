#!/usr/bin/env python3
"""
pos_encoding.py

Provides a naive sinusoidal positional encoding function for sequence length T
and hidden_dim. Often used in Transformers to add position information to tokens.

Usage:
  from .pos_encoding import positional_encoding

  pe = positional_encoding(seq_len=16, hidden_dim=32)
  # shape => (16, 32)
"""

import numpy as np

def positional_encoding(T, hidden_dim):
    """
    T: sequence length
    hidden_dim: model dimension
    returns: shape (T, hidden_dim) of float32
    """
    pe = np.zeros((T, hidden_dim), dtype=np.float32)
    for pos in range(T):
        for i in range(0, hidden_dim, 2):
            denominator = float(10000 ** (i / hidden_dim))
            pe[pos, i] = np.sin(pos / denominator)
            if i + 1 < hidden_dim:
                pe[pos, i+1] = np.cos(pos / denominator)
    return pe

# Optional test
if __name__ == "__main__":
    print("=== Testing pos_encoding.py ===")
    T = 8
    hd = 16
    pe_matrix = positional_encoding(T, hd)
    print("[Test] shape:", pe_matrix.shape)  # (8,16)
    print("[Test] sample:\n", pe_matrix[:3, :6])
    print("=== Done testing pos_encoding.py ===")
