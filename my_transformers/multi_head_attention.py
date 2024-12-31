#!/usr/bin/env python3
"""
multi_head_attention.py

Defines a MultiHeadAttention class referencing `opencl_backend.py` for the rowwise softmax step.
CPU-based Q, K, V transformations, GPU-based attention distribution. 
Returns (x + final_projection) as a residual update.

Usage example:
  from .multi_head_attention import MultiHeadAttention
  mha = MultiHeadAttention(hidden_dim=32, n_heads=2)
  out = mha.forward(x)  # x shape => (batch*seq_len, hidden_dim)
"""

import numpy as np
from .opencl_backend import OpenCLBackend

class MultiHeadAttention:
    """
    A single multi-head self-attention block:
     - Q, K, V transformations: x @ Wq, x @ Wk, x @ Wv
     - For each head, compute rowwise softmax of QK^T / sqrt(dim)
     - Weighted sum => multiply by V
     - Concatenate heads => project => residual add => return
    """
    def __init__(self, hidden_dim, n_heads, backend=None):
        """
        hidden_dim: total model dimension
        n_heads: how many heads
        backend: an OpenCLBackend instance for GPU ops (softmax)
        """
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        if backend is None:
            backend = OpenCLBackend()  # fallback
        self.backend = backend

        rng = np.random.default_rng(42)
        # Wq, Wk, Wv: shape (hidden_dim, hidden_dim)
        self.Wq = (0.02 * rng.normal(size=(hidden_dim, hidden_dim))).astype(np.float32)
        self.Wk = (0.02 * rng.normal(size=(hidden_dim, hidden_dim))).astype(np.float32)
        self.Wv = (0.02 * rng.normal(size=(hidden_dim, hidden_dim))).astype(np.float32)
        # Final projection
        self.Wo = (0.02 * rng.normal(size=(hidden_dim, hidden_dim))).astype(np.float32)

        print(f"[MultiHeadAttention] Created with hidden_dim={hidden_dim}, n_heads={n_heads}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (batch*seq_len, hidden_dim) 
                 or (some_T, hidden_dim)
        Returns: same shape as x
        """
        T = x.shape[0]
        # Q, K, V on CPU for demonstration
        Q = x @ self.Wq  # shape (T, hidden_dim)
        K = x @ self.Wk
        V = x @ self.Wv

        # Reshape => (T, n_heads, head_dim)
        Q_resh = Q.reshape(T, self.n_heads, self.head_dim)
        K_resh = K.reshape(T, self.n_heads, self.head_dim)
        V_resh = V.reshape(T, self.n_heads, self.head_dim)

        heads_output = []
        for h in range(self.n_heads):
            Qh = Q_resh[:, h, :]  # (T, head_dim)
            Kh = K_resh[:, h, :]
            Vh = V_resh[:, h, :]

            # (T, head_dim) x (head_dim, T) => (T, T)
            attn_scores = Qh @ Kh.T
            attn_scores = attn_scores / np.sqrt(self.head_dim)

            # rowwise softmax on GPU
            attn_scores = self.backend.gpu_softmax_2d_inplace(attn_scores)

            # Weighted sum => shape (T, head_dim)
            out_h = attn_scores @ Vh
            heads_output.append(out_h)

        # Concat heads => shape (T, hidden_dim)
        out_cat = np.concatenate(heads_output, axis=1)

        # Final projection => shape (T, hidden_dim)
        proj = out_cat @ self.Wo

        # Residual
        return x + proj

# Optional test
if __name__ == "__main__":
    print("=== Testing multi_head_attention.py ===")
    mha = MultiHeadAttention(hidden_dim=32, n_heads=2)
    # random input (T=5, hidden_dim=32)
    x_dummy = np.random.randn(5,32).astype(np.float32)

    out = mha.forward(x_dummy)
    print("[Test] out shape =", out.shape)  # (5,32)
    print("[Test] out sample:\n", out[:2,:6])
    print("=== Done testing multi_head_attention ===")
