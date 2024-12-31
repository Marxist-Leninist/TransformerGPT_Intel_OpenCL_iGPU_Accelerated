#!/usr/bin/env python3
"""
multi_layer_gpt.py

Defines a GPT-like model that:
 - Has an embedding for tokens
 - Optionally adds sinusoidal positional encoding
 - Stacks multiple layers (each layer = MultiHeadAttention + FeedForward + residual)
 - Has a final linear head for LM tasks (if desired)

Usage example:
  from .multi_layer_gpt import GPTModel
  model = GPTModel(vocab_size=1000, hidden_dim=32, n_heads=2, n_layers=2, block_size=16)
  logits = model.forward(x_idx) 
"""

import numpy as np

from .embedding import TokenEmbedding
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .pos_encoding import positional_encoding


class GPTBlock:
    """
    A single Transformer block: MHA + FF + optional residual connections.
    """
    def __init__(self, hidden_dim, n_heads, expansion=4, backend=None):
        self.mha = MultiHeadAttention(hidden_dim, n_heads, backend=backend)
        self.ffn = FeedForward(hidden_dim, expansion=expansion, backend=backend)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (B*S, hidden_dim) or (T, hidden_dim)
        Returns the updated hidden states after MHA + FF + residual
        """
        # multi-head attn
        x_mha = self.mha.forward(x)  # shape same as x
        # feed-forward
        x_ffn = self.ffn.forward(x_mha)  # shape same as x
        return x_ffn


class GPTModel:
    """
    A multi-layer GPT model:
      - Embedding
      - optional sinusoidal positional enc
      - n_layers of (MHA + FF)
      - final linear head (for language modeling) + bias if do_lm_head=True

    Typical usage:
      model = GPTModel(vocab_size, hidden_dim, n_heads, n_layers, block_size=16, ...)
      logits = model.forward(idx_batch)
    """
    def __init__(self, vocab_size, hidden_dim, n_heads, n_layers,
                 block_size=16, do_pos_encoding=True, expansion=4,
                 do_lm_head=True, backend=None):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.do_pos_encoding = do_pos_encoding
        self.do_lm_head = do_lm_head

        # Create embedding
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            use_gpu_embedding=False,
            backend=backend
        )

        # We'll cache a single positional encoding matrix for block_size
        self.pos_enc_cache = None

        # Build layers
        self.layers = []
        for _ in range(n_layers):
            block = GPTBlock(hidden_dim, n_heads, expansion=expansion, backend=backend)
            self.layers.append(block)

        # Optional final LM head
        self.lm_head = None
        self.lm_bias = None
        if do_lm_head:
            rng = np.random.default_rng(42)
            self.lm_head = (0.02 * rng.normal(size=(hidden_dim, vocab_size))).astype(np.float32)
            self.lm_bias = np.zeros((vocab_size,), dtype=np.float32)

        print(f"[GPTModel] Created GPT with vocab_size={vocab_size}, hidden_dim={hidden_dim}, "
              f"n_heads={n_heads}, n_layers={n_layers}, block_size={block_size}, do_pos_enc={do_pos_encoding}, "
              f"do_lm_head={do_lm_head}")

    def forward(self, idx_batch: np.ndarray) -> np.ndarray:
        """
        idx_batch: shape (batch_size, block_size)
        Returns:
          - shape (batch_size, block_size, vocab_size) if do_lm_head=True
          - shape (batch_size, block_size, hidden_dim) otherwise
        """
        B, S = idx_batch.shape

        # 1) Embedding => (B,S,hidden_dim)
        x_embed = self.embedding.forward(idx_batch)

        # 2) (Optional) add pos encoding
        if self.do_pos_encoding:
            if self.pos_enc_cache is None or self.pos_enc_cache.shape[0] < S:
                # compute fresh or bigger one
                pe = positional_encoding(S, self.hidden_dim)
                self.pos_enc_cache = pe  # shape (S, hidden_dim)

            # Add pos enc to x_embed
            # broadcast across batch
            x_embed += self.pos_enc_cache[np.newaxis, :S, :]

        # Flatten => (B*S, hidden_dim)
        x_2d = x_embed.reshape(B*S, self.hidden_dim)

        # run each layer
        for block in self.layers:
            x_2d = block.forward(x_2d)

        # if no final LM head, return shape (B,S,hidden_dim)
        if not self.do_lm_head or self.lm_head is None:
            return x_2d.reshape(B, S, self.hidden_dim)

        # else do a final linear projection => (B*S,vocab_size)
        logits_2d = x_2d @ self.lm_head + self.lm_bias
        # reshape => (B,S,vocab_size)
        logits_3d = logits_2d.reshape(B, S, self.vocab_size)
        return logits_3d


# Optional test
if __name__ == "__main__":
    print("=== Testing multi_layer_gpt.py ===")

    vocab_size = 50
    hidden_dim = 16
    n_heads = 2
    n_layers = 2
    block_size=8

    model = GPTModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        block_size=block_size,
        do_pos_encoding=True,
        do_lm_head=True
    )

    # random input indices
    B=2
    idx_batch = np.random.randint(0, vocab_size, size=(B, block_size), dtype=np.int32)
    out = model.forward(idx_batch)
    print("[Test] output shape =", out.shape)  # (2,8,50)
    print("[Test] output sample:\n", out[0, :3, :6])
    print("=== Done testing multi_layer_gpt.py ===")
