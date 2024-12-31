#!/usr/bin/env python3
"""
dataset.py

Hard-coded text dataset for Transformer training.

- Always reads from "C:\\Users\\Scott\\Downloads\\my_transformers\\o1Prochatdata.txt"
- Builds a char-level vocab
- get_batch(...) method provides random (x, y) blocks for training

Usage Example:
  from dataset import HardcodedDataset

  ds = HardcodedDataset(block_size=16)
  x_batch, y_batch = ds.get_batch(batch_size=4)
"""

import os
import numpy as np

class HardcodedDataset:
    def __init__(self, block_size=16):
        # Hard-coded path to the text file
        self.file_path = r"C:\Users\Scott\Downloads\my_transformers\o1Prochatdata.txt"
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Text file not found at {self.file_path}")

        self.block_size = block_size

        # Read entire text
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.data = f.read().strip()

        # Build char-level vocab
        self.chars = sorted(list(set(self.data)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

        print(f"[HardcodedDataset] Loaded text from: {self.file_path}")
        print(f"[HardcodedDataset] Data length: {len(self.data)} chars")
        print(f"[HardcodedDataset] Vocab size: {self.vocab_size}")

    def get_batch(self, batch_size=4):
        """
        Returns (x, y):
         - x: shape (batch_size, block_size)  [int indices]
         - y: shape (batch_size, block_size)  [int indices]
        The target y is the same sequence shifted by 1 char.
        """
        import random

        data_len = len(self.data)
        xs = []
        ys = []
        for _ in range(batch_size):
            start_pos = np.random.randint(0, data_len - self.block_size - 1)
            chunk = self.data[start_pos : start_pos + self.block_size]
            chunk_next = self.data[start_pos+1 : start_pos + self.block_size + 1]

            # Convert to int indices
            x_idx = [self.stoi[ch] for ch in chunk]
            y_idx = [self.stoi[ch] for ch in chunk_next]

            xs.append(x_idx)
            ys.append(y_idx)

        x_array = np.array(xs, dtype=np.int32)
        y_array = np.array(ys, dtype=np.int32)
        return x_array, y_array


# Optional quick test
if __name__ == "__main__":
    ds = HardcodedDataset(block_size=8)
    xb, yb = ds.get_batch(batch_size=2)
    print("[Test] x_batch shape:", xb.shape)
    print("[Test] y_batch shape:", yb.shape)
    print("[Test] x_batch sample:\n", xb)
    print("[Test] y_batch sample:\n", yb)
