#!/usr/bin/env python3
"""
training.py

Provides a minimal TransformerTrainer class to train a GPT model
by partial backprop (embedding + final LM head) on CPU. 
Forward passes use GPU ops for MHA, FF, etc.

Usage example:
  from .multi_layer_gpt import GPTModel
  from .training import TransformerTrainer

  model = GPTModel(...)
  trainer = TransformerTrainer(model, lr=1e-4, grad_clip=5.0)

  x_batch, y_batch = dataset.get_batch(batch_size=4)
  loss = trainer.train_step(x_batch, y_batch)
"""

import numpy as np
from .multi_layer_gpt import GPTModel

class TransformerTrainer:
    """
    A minimal trainer that does forward pass on GPU, but partial
    gradient updates on CPU for embedding + final LM head.

    Full backprop of MHA and FF layers is more involved; 
    this at least demonstrates partial param training.
    """
    def __init__(self, model: GPTModel, lr=1e-4, grad_clip=5.0):
        self.model = model
        self.lr = lr
        self.grad_clip = grad_clip

        print(f"[TransformerTrainer] Initialized with lr={lr}, grad_clip={grad_clip}")

    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        x_batch: shape (batch_size, block_size)
        y_batch: shape (batch_size, block_size)
        Returns: scalar loss (cross-entropy)

        We'll do forward pass on GPU, partial CPU-based backprop for:
          - model.embedding.embedding_matrix
          - model.lm_head, model.lm_bias
        ignoring MHA + FF param for brevity
        """
        # 1) Forward pass => shape (B, S, vocab_size)
        logits_3d = self.model.forward(x_batch)
        B, S, _ = logits_3d.shape

        # 2) cross-entropy on CPU
        # flatten => (B*S, vocab_size)
        logits_2d = logits_3d.reshape(B*S, self.model.vocab_size)
        targets_1d = y_batch.reshape(-1)  # (B*S,)

        loss_val = 0.0

        # partial grads
        # we handle only: embedding, final head
        embed = self.model.embedding.embedding_matrix
        d_embed = np.zeros_like(embed)

        W_head = self.model.lm_head
        b_head = self.model.lm_bias
        d_W_head = np.zeros_like(W_head) if W_head is not None else None
        d_b_head = np.zeros_like(b_head) if b_head is not None else None

        for i in range(B*S):
            row = logits_2d[i]  # shape (vocab_size,)
            t = targets_1d[i]
            # stable softmax
            mx = np.max(row)
            exps = np.exp(row - mx)
            sum_exps = np.sum(exps)
            probs = exps / (sum_exps + 1e-9)

            loss_val += -np.log(probs[t] + 1e-9)

            # gradient wrt logits
            dlogits = probs
            dlogits[t] -= 1.0

            # partial param updates for:
            #   embed => gather from x_batch
            #   final head => x ~ embed token
            # We do a naive approach ignoring MHA/FF internal states.

            # find which token index we used
            samp_id = i // S
            pos_id = i % S
            token_idx = x_batch[samp_id, pos_id]

            # the "hidden vector" feeding final head is just embed
            # ignoring the MHA/FF transformations. (incomplete)
            # But at least it trains embed + head.
            feat = embed[token_idx]  # shape (hidden_dim,)

            # gradient wrt W_head, b_head
            if W_head is not None and d_W_head is not None:
                for vc in range(self.model.vocab_size):
                    d_W_head[:, vc] += dlogits[vc] * feat
                d_b_head += dlogits

            # gradient wrt embedding
            if W_head is not None:
                for vc in range(self.model.vocab_size):
                    d_embed[token_idx] += dlogits[vc] * W_head[:, vc]

        loss_val /= (B*S)

        # clip
        grad_sq = (d_embed**2).sum()
        if d_W_head is not None:
            grad_sq += (d_W_head**2).sum() + (d_b_head**2).sum()
        gnorm = np.sqrt(grad_sq)
        if gnorm > self.grad_clip:
            scale = self.grad_clip / gnorm
            d_embed *= scale
            if d_W_head is not None:
                d_W_head *= scale
                d_b_head *= scale

        # update embed
        embed -= self.lr * d_embed
        # update final head
        if W_head is not None and d_W_head is not None:
            W_head -= self.lr * d_W_head
            b_head -= self.lr * d_b_head

        return float(loss_val)


# Optional test
if __name__ == "__main__":
    print("=== Testing training.py ===")
    from .dataset import HardcodedDataset

    # Build a small GPT
    ds = HardcodedDataset(block_size=8)
    from .multi_layer_gpt import GPTModel
    model = GPTModel(
        vocab_size=ds.vocab_size,
        hidden_dim=16,
        n_heads=2,
        n_layers=1,
        block_size=8,
        do_pos_encoding=True,
        do_lm_head=True
    )

    trainer = TransformerTrainer(model, lr=1e-4, grad_clip=5.0)
    x_batch, y_batch = ds.get_batch(batch_size=2)
    loss_before = trainer.train_step(x_batch, y_batch)
    print("[Test] loss before training step:", loss_before)

    # do a second step
    loss_after = trainer.train_step(x_batch, y_batch)
    print("[Test] loss after training step:", loss_after)
    print("=== Done testing training.py ===")
