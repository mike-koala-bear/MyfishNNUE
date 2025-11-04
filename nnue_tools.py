#!/usr/bin/env python3
"""
Lightweight NNUE I/O helpers for .nnue binaries used by this project.

Format written by training/src/nnue_finetune.py::write_nnue:
  int32[3]: [in_dim, h1, h2]
  float32 arrays in order:
    W1 (h1, in_dim), b1 (h1,)
    W2 (h2, h1),     b2 (h2,)
    W3 (h2,),        b3 (1,)
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np


Weights = Dict[str, np.ndarray]


def load_nnue(path: str) -> Tuple[int, int, int, Weights]:
    p = os.path.abspath(path)
    with open(p, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=3)
        if header.size != 3:
            raise RuntimeError("NNUE header missing or corrupted")
        in_dim, h1, h2 = map(int, header)
        W1 = np.fromfile(f, dtype=np.float32, count=in_dim * h1).reshape(h1, in_dim)
        b1 = np.fromfile(f, dtype=np.float32, count=h1)
        W2 = np.fromfile(f, dtype=np.float32, count=h1 * h2).reshape(h2, h1)
        b2 = np.fromfile(f, dtype=np.float32, count=h2)
        W3 = np.fromfile(f, dtype=np.float32, count=h2)
        b3 = np.fromfile(f, dtype=np.float32, count=1)

    weights: Weights = {
        "fc1.weight": W1.astype(np.float32, copy=False),
        "fc1.bias": b1.astype(np.float32, copy=False),
        "fc2.weight": W2.astype(np.float32, copy=False),
        "fc2.bias": b2.astype(np.float32, copy=False),
        "fc3.weight": W3.astype(np.float32, copy=False),
        "fc3.bias": b3.astype(np.float32, copy=False),
    }
    return in_dim, h1, h2, weights


def save_nnue(path: str, in_dim: int, h1: int, h2: int, weights: Weights) -> None:
    p = os.path.abspath(path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    W1 = np.asarray(weights["fc1.weight"], dtype=np.float32)
    b1 = np.asarray(weights["fc1.bias"], dtype=np.float32)
    W2 = np.asarray(weights["fc2.weight"], dtype=np.float32)
    b2 = np.asarray(weights["fc2.bias"], dtype=np.float32)
    W3 = np.asarray(weights["fc3.weight"], dtype=np.float32).reshape(-1)
    b3 = np.asarray(weights["fc3.bias"], dtype=np.float32).reshape(1)

    # Sanity checks
    assert W1.shape == (h1, in_dim), f"W1 shape mismatch: {W1.shape} != {(h1, in_dim)}"
    assert b1.shape == (h1,), f"b1 shape mismatch: {b1.shape} != {(h1,)}"
    assert W2.shape == (h2, h1), f"W2 shape mismatch: {W2.shape} != {(h2, h1)}"
    assert b2.shape == (h2,), f"b2 shape mismatch: {b2.shape} != {(h2,)}"
    assert W3.shape == (h2,), f"W3 shape mismatch: {W3.shape} != {(h2,)}"
    assert b3.shape == (1,), f"b3 shape mismatch: {b3.shape} != {(1,)}"

    with open(p, "wb") as f:
        np.array([in_dim, h1, h2], dtype=np.int32).tofile(f)
        W1.tofile(f); b1.tofile(f)
        W2.tofile(f); b2.tofile(f)
        W3.tofile(f); b3.tofile(f)


def dims_equal(d1: Tuple[int, int, int], d2: Tuple[int, int, int]) -> bool:
    return tuple(map(int, d1)) == tuple(map(int, d2))


def count_params(in_dim: int, h1: int, h2: int) -> int:
    return (h1 * in_dim + h1) + (h2 * h1 + h2) + (h2 + 1)

