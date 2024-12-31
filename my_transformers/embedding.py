#!/usr/bin/env python3
"""
embedding.py

Defines a TokenEmbedding class that transforms integer token indices
into continuous vectors of dimension 'hidden_dim'.

Imports opencl_backend locally, but by default we do CPU-based gather
(e.g., embedding_matrix[idx]), since that's common for embeddings.
If you wish, you can switch to a GPU-based 'one-hot x embedding' approach.

Usage example:
  from .embedding import TokenEmbedding

  embed = TokenEmbedding(vocab_size=1000, hidden_dim=32)
  x_idx = np.array([[0,1,2],[3,4,5]], dtype=np.int32) # shape (batch_size, seq_len)
  x_embed = embed.forward(x_idx) # shape (batch_size, seq_len, hidden_dim)
"""

import numpy as np
# We do a relative import from 'opencl_backend' if you want to do a GPU-based approach:
from .opencl_backend import OpenCLBackend


class TokenEmbedding:
    """
    Minimal token embedding class with shape (vocab_size, hidden_dim).
    By default, forward() does a CPU-based gather approach:
       x_embed[b, s, :] = self.embedding_matrix[x_idx[b, s]]
    If you want GPU-based embedding, you can implement a one-hot matmul approach.
    """
    def __init__(self, vocab_size, hidden_dim, use_gpu_embedding=False, backend=None):
        """
        - vocab_size: int
        - hidden_dim: int
        - use_gpu_embedding: if True, we'll do a one-hot x embedding matmul on GPU
        - backend: an OpenCLBackend instance (if use_gpu_embedding is True)
        """
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.use_gpu_embedding = use_gpu_embedding
        # If we want GPU approach, we need a backend
        if use_gpu_embedding and backend is None:
            backend = OpenCLBackend()
        self.backend = backend

        # Random init
        rng = np.random.default_rng(42)
        self.embedding_matrix = (0.02 * rng.normal(size=(vocab_size, hidden_dim))).astype(np.float32)

        print(f"[TokenEmbedding] Created embedding of shape ({vocab_size}, {hidden_dim})")

    def forward(self, idx_batch: np.ndarray) -> np.ndarray:
        """
        idx_batch: shape (batch_size, seq_len) of int indices in [0, vocab_size)
        Returns: shape (batch_size, seq_len, hidden_dim) of float32
        """
        B, S = idx_batch.shape

        if not self.use_gpu_embedding:
            # CPU gather approach
            # We'll do a direct gather for each token
            x_embed = self.embedding_matrix[idx_batch.reshape(-1)]  # shape (B*S, hidden_dim)
            x_embed = x_embed.reshape(B, S, self.hidden_dim)
            return x_embed
        else:
            # GPU-based approach: create a one-hot for each token, matmul with embedding
            # This is typically slower for large vocab, but we demonstrate how it might work.
            # shape => one_hot: (B*S, vocab_size)
            # embedding_matrix: (vocab_size, hidden_dim)
            # out => (B*S, hidden_dim)

            # 1) create one_hot on CPU
            total_tokens = B * S
            one_hot = np.zeros((total_tokens, self.vocab_size), dtype=np.float32)
            for i in range(total_tokens):
                token_id = idx_batch.ravel()[i]
                one_hot[i, token_id] = 1.0

            # 2) pass one_hot and self.embedding_matrix to GPU matmul
            out_2d = self.backend.gpu_matmul(one_hot, self.embedding_matrix)
            out_3d = out_2d.reshape(B, S, self.hidden_dim)
            return out_3d


# Optional test
if __name__ == "__main__":
    print("=== Testing embedding.py ===")

    # Simple test
    embed = TokenEmbedding(vocab_size=10, hidden_dim=4, use_gpu_embedding=False)
    # random indices
    x_idx = np.array([
        [0,1,2],
        [3,2,9]
    ], dtype=np.int32)  # shape (2,3)

    out = embed.forward(x_idx)
    print("[Test] out shape =", out.shape)  # (2,3,4)
    print("[Test] out sample:\n", out)
    print("=== Done testing embedding.py ===")
