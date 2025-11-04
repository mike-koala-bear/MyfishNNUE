#!/usr/bin/env python3
"""
Compare two NNUE .nnue binaries and report per-layer and global differences.

Usage:
  python training/src/nnue_compare.py --a path/to/A.nnue --b path/to/B.nnue [--epsilon 1e-6]
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np

import sys
# Allow running as a script without package context
sys.path.insert(0, os.path.dirname(__file__))
from nnue_tools import load_nnue, dims_equal, count_params


def layer_stats(a: np.ndarray, b: np.ndarray, eps: float) -> Dict[str, float]:
    d = (b.astype(np.float32) - a.astype(np.float32))
    ad = np.abs(d)
    return {
        "l1": float(np.sum(ad)),
        "l2": float(np.sqrt(np.sum(d * d))),
        "mean_abs": float(np.mean(ad)),
        "max_abs": float(np.max(ad) if ad.size else 0.0),
        "gt_eps": float(np.count_nonzero(ad > eps)),
        "count": float(ad.size),
    }


def fmt_stats(name: str, st: Dict[str, float], eps: float) -> str:
    frac = (st["gt_eps"] / st["count"]) if st["count"] else 0.0
    return (
        f"{name:12s} | l1={st['l1']:.4e}  l2={st['l2']:.4e}  "
        f"mean={st['mean_abs']:.4e}  max={st['max_abs']:.4e}  "
        f">eps({eps:g})={int(st['gt_eps'])}/{int(st['count'])} ({frac*100:.2f}%)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two NNUE .nnue files")
    ap.add_argument("--a", required=True, help="Path to first NNUE (.nnue)")
    ap.add_argument("--b", required=True, help="Path to second NNUE (.nnue)")
    ap.add_argument("--epsilon", type=float, default=1e-6, help="Threshold for significant change")
    args = ap.parse_args()

    if not os.path.exists(args.a) or not os.path.exists(args.b):
        print("Error: input file(s) not found")
        return 2

    in1, h11, h21, w1 = load_nnue(args.a)
    in2, h12, h22, w2 = load_nnue(args.b)
    if not dims_equal((in1, h11, h21), (in2, h12, h22)):
        print(f"Error: dimension mismatch: {(in1,h11,h21)} vs {(in2,h12,h22)}")
        return 3

    print("NNUE Compare")
    print(f"  dims: in={in1}, h1={h11}, h2={h21}; params={count_params(in1,h11,h21):,}")
    print(f"  files:\n    A: {os.path.abspath(args.a)}\n    B: {os.path.abspath(args.b)}")
    print(f"  epsilon: {args.epsilon}")

    keys = [
        ("fc1.weight", "W1"), ("fc1.bias", "b1"),
        ("fc2.weight", "W2"), ("fc2.bias", "b2"),
        ("fc3.weight", "W3"), ("fc3.bias", "b3"),
    ]

    totals = {"l1": 0.0, "l2": 0.0, "mean_abs": 0.0, "max_abs": 0.0, "count": 0.0, "gt_eps": 0.0}
    for k, label in keys:
        st = layer_stats(w1[k], w2[k], args.epsilon)
        for t in ["l1", "l2", "mean_abs", "max_abs", "count", "gt_eps"]:
            totals[t] += st[t]
        print(fmt_stats(label, st, args.epsilon))

    # Aggregate metrics
    mean_abs = totals["mean_abs"] if totals["count"] == 0 else (totals["l1"] / totals["count"])
    frac = (totals["gt_eps"] / totals["count"]) if totals["count"] else 0.0
    print("-" * 80)
    print(
        f"TOTAL        | l1={totals['l1']:.4e}  l2={totals['l2']:.4e}  "
        f"mean={mean_abs:.4e}  max={totals['max_abs']:.4e}  >eps={int(totals['gt_eps'])}/{int(totals['count'])} ({frac*100:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
