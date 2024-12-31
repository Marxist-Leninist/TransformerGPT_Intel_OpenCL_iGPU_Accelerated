#!/usr/bin/env python3
"""
feed_forward.py

Defines a FeedForward class for the Transformer's MLP sub-layer:
  x -> (x@W_ff1 + b_ff1) -> ReLU -> (that@W_ff2 + b_ff2) -> residual

Uses the opencl_backend.py for GPU matmul, add_bias, relu_inplace. 
"""

import numpy as np
from .opencl_backend import OpenCLBackend

class FeedForward:
    """
    A 2-layer MLP block with an expansion factor (e.g., 4).
    We'll do:
       ff1 = relu( x@W_ff1 + b_ff1 )
       ff2 = ff1@W_ff2 + b_ff2
       return x + ff2  (residual)
    """
    def __init__(self, hidden_dim, expansion=4, backend=None):
        self.hidden_dim = hidden_dim
        self.expanded_dim = hidden_dim * expansion

        if backend is None:
            backend = OpenCLBackend()  # fallback
        self.backend = backend

        rng = np.random.default_rng(1337)
        # (hidden_dim, expanded_dim)
        self.W_ff1 = (0.02 * rng.normal(size=(hidden_dim, self.expanded_dim))).astype(np.float32)
        self.b_ff1 = np.zeros((self.expanded_dim,), dtype=np.float32)

        # (expanded_dim, hidden_dim)
        self.W_ff2 = (0.02 * rng.normal(size=(self.expanded_dim, hidden_dim))).astype(np.float32)
        self.b_ff2 = np.zeros((hidden_dim,), dtype=np.float32)

        print(f"[FeedForward] Created with hidden_dim={hidden_dim}, expansion={expansion}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (T, hidden_dim)
        returns: same shape (T, hidden_dim)
        """
        # 1) x@W_ff1 + b_ff1 => ReLU
        ff1 = self.backend.gpu_matmul(x, self.W_ff1)  # shape (T, expanded_dim)
        ff1 = self.backend.gpu_add_bias(ff1, self.b_ff1)
        ff1 = self.backend.gpu_relu_inplace(ff1)

        # 2) ff1@W_ff2 + b_ff2
        ff2 = self.backend.gpu_matmul(ff1, self.W_ff2)  # shape (T, hidden_dim)
        ff2 = self.backend.gpu_add_bias(ff2, self.b_ff2)

        # 3) residual
        return x + ff2


# Optional test
if __name__ == "__main__":
    print("=== Testing feed_forward.py ===")
    ff = FeedForward(hidden_dim=32, expansion=4)
    # random input
    x_dummy = np.random.randn(5, 32).astype(np.float32)

    out = ff.forward(x_dummy)
    print("[Test] out shape =", out.shape)  # (5,32)
    print("[Test] out sample:\n", out[:2,:6])
    print("=== Done testing feed_forward.py ===")
