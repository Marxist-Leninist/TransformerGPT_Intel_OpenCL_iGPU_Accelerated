#!/usr/bin/env python3
"""
main.py

Brings everything together:
 - Loads the dataset (dataset.py)
 - Builds the GPT model (multi_layer_gpt.py)
 - Creates a trainer (training.py) for partial param updates
 - Runs a small training loop
 - Does a short text generation sample

Usage:
  python main.py
"""

import numpy as np
from .dataset import HardcodedDataset
from .multi_layer_gpt import GPTModel
from .training import TransformerTrainer

def main():
    print("=== Minimal MAIN script for GPU-based Transformer GPT training ===")

    # 1) Load data
    ds = HardcodedDataset(block_size=16)
    # specify some hyperparams
    hidden_dim = 32
    n_heads = 2
    n_layers = 1
    block_size = 16

    # 2) Build the model
    model = GPTModel(
        vocab_size=ds.vocab_size,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        block_size=block_size,
        do_pos_encoding=True,
        do_lm_head=True
    )

    # 3) Create trainer
    trainer = TransformerTrainer(
        model=model,
        lr=1e-4,
        grad_clip=5.0
    )

    # 4) Do a small training loop
    steps = 200
    batch_size = 4
    for step_i in range(steps):
        x_batch, y_batch = ds.get_batch(batch_size)
        loss_val = trainer.train_step(x_batch, y_batch)
        if step_i % 20 == 0:
            print(f"Step {step_i}/{steps}, loss={loss_val:.4f}")

    # 5) Simple generation
    print("\n=== Generating text sample ===")
    # We'll start from a random block
    x_init, _ = ds.get_batch(1)  # shape (1, block_size)
    # We'll generate 100 new tokens
    gen_len = 100
    out_str = generate_text(model, x_init[0], ds, gen_len)
    print("Generated text:\n", out_str)

def generate_text(model, init_tokens, ds, gen_len=100):
    """
    model: GPTModel
    init_tokens: shape (block_size,)
    ds: HardcodedDataset for stoi, itos
    gen_len: how many new tokens to generate
    Returns a string
    """
    block_size = model.block_size
    tokens = init_tokens.copy()  # shape (block_size,)

    out_str = ""
    for _ in range(gen_len):
        # forward => shape (1, block_size, vocab_size)
        logits_3d = model.forward(tokens[np.newaxis, :])
        # look at last position
        last_logits = logits_3d[0, -1]  # shape (vocab_size,)
        # pick argmax for a deterministic approach
        idx_next = int(np.argmax(last_logits))
        # append to out_str
        out_str += ds.itos[idx_next]
        # shift tokens
        tokens = np.concatenate([tokens[1:], [idx_next]])

    return out_str

if __name__ == "__main__":
    main()
