#!/usr/bin/env python3
"""
Blend two NNUE .nnue binaries via linear interpolation and write a new .nnue.

Usage:
  python training/src/nnue_blend.py --a A.nnue --b B.nnue --alpha 0.5 --out blended.nnue

The result is: (1 - alpha) * A + alpha * B for every weight and bias.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np

import sys
# Allow running as a script without package context
sys.path.insert(0, os.path.dirname(__file__))
from nnue_tools import load_nnue, save_nnue, dims_equal


def main() -> int:
    ap = argparse.ArgumentParser(description="Blend two NNUE .nnue files")
    ap.add_argument("--a", required=True, help="Path to first NNUE (.nnue)")
    ap.add_argument("--b", required=True, help="Path to second NNUE (.nnue)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Blend factor: result=(1-a)*A + a*B")
    ap.add_argument("--out", required=True, help="Output .nnue path")
    args = ap.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print("Error: --alpha must be in [0,1]")
        return 2
    if not os.path.exists(args.a) or not os.path.exists(args.b):
        print("Error: input file(s) not found")
        return 2

    in1, h11, h21, w1 = load_nnue(args.a)
    in2, h12, h22, w2 = load_nnue(args.b)
    if not dims_equal((in1, h11, h21), (in2, h12, h22)):
        print(f"Error: dimension mismatch: {(in1,h11,h21)} vs {(in2,h12,h22)}")
        return 3

    a = float(args.alpha)
    b = 1.0 - a
    out: Dict[str, np.ndarray] = {}
    for k in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]:
        A = w1[k].astype(np.float32)
        B = w2[k].astype(np.float32)
        out[k] = (b * A + a * B).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_nnue(args.out, in1, h11, h21, out)
    print(f"Blended NNUE saved to {os.path.abspath(args.out)} (alpha={a:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
