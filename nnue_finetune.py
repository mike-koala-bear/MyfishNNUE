#!/usr/bin/env python3
"""
NNUE Fine-Tuner — continues training from an existing checkpoint
and exports a fresh `.nnue` binary for the C++ core.

Features
- Loads from .nnue (custom binary), .npz, or .pt/.pth
- Trains MLP (in=776 → h1 → h2 → 1) with ReLU
- Flexible data: FEN file (+ optional inline cp labels) or quick self‑play
- Optional teacher labels via an external UCI engine (time or depth) or classical eval fallback
- Mixed precision, cosine LR schedule, gradient clipping, early stopping
- Exports both .pt and .nnue outputs

Usage examples
  # Finetune from training/best.nnue using a FEN list labeled by classical eval
  python training/src/nnue_finetune.py \
      --from training/best.nnue \
      --fens training/fens.txt \
      --epochs 3 --batch 512 --lr 7e-4 \
      --out-pt training/checkpoints/finetuned.pt \
      --out-nnue training/best_finetuned.nnue

  # With external UCI teacher (depth based)
  python training/src/nnue_finetune.py \
      --from training/best.nnue \
      --fens training/fens.txt \
      --teacher self --engine-path ./build/myfish --engine-depth 8 \
      --epochs 2 --batch 512 \
      --out-nnue training/best_finetuned.nnue

  # Quick bootstrap without dataset (self-play positions + classical labels)
  python training/src/nnue_finetune.py \
      --from training/best.nnue \
      --selfplay 200000 --epochs 1 --batch 1024 \
      --out-nnue training/best_finetuned.nnue

  # Use your own UCI engine as teacher (time based)
  python training/src/nnue_finetune.py \
      --from training/best.nnue \
      --selfplay 100000 \
      --teacher self --engine-path ./build/myfish --engine-time 0.3 \
      --epochs 2 --batch 512 \
      --out-nnue training/best_selfteacher.nnue
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import logging
from typing import Optional, Iterable, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:
    print("PyTorch is required.", file=sys.stderr)
    raise

try:
    import chess
    import chess.pgn
    import chess.engine
except Exception:
    chess = None  # Only needed for FEN/selfplay


BASE_INPUT_DIM = 776

# Progress bars (tqdm) — optional
try:
    from tqdm.auto import tqdm
except Exception:  # fallback no-op
    def tqdm(iterable=None, total=None, desc=None, unit=None, leave=None):
        return iterable if iterable is not None else range(total or 0)


def _configure_io_and_logging(log_file: Optional[str]) -> Optional[object]:
    """Configure unbuffered I/O and optional live tee logging to a file.

    Returns an open file handle if log_file is provided (caller should close), else None.
    """
    # Make stdout/stderr line-buffered to flush on each newline
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    fh = None
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        fh = open(log_file, 'a', buffering=1)

        class _Tee:
            def __init__(self, *streams):
                self._streams = streams
            def write(self, data):
                for s in self._streams:
                    try:
                        s.write(data)
                        s.flush()
                    except Exception:
                        pass
            def flush(self):
                for s in self._streams:
                    try:
                        s.flush()
                    except Exception:
                        pass

        sys.stdout = _Tee(sys.__stdout__, fh)  # type: ignore
        sys.stderr = _Tee(sys.__stderr__, fh)  # type: ignore

    # Basic logging setup (optional; we primarily rely on prints and tqdm)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    return fh


def features_from_fen(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    feats = np.zeros(BASE_INPUT_DIM, dtype=np.float32)
    piece_order = [
        (chess.PAWN, chess.WHITE),
        (chess.KNIGHT, chess.WHITE),
        (chess.BISHOP, chess.WHITE),
        (chess.ROOK, chess.WHITE),
        (chess.QUEEN, chess.WHITE),
        (chess.KING, chess.WHITE),
        (chess.PAWN, chess.BLACK),
        (chess.KNIGHT, chess.BLACK),
        (chess.BISHOP, chess.BLACK),
        (chess.ROOK, chess.BLACK),
        (chess.QUEEN, chess.BLACK),
        (chess.KING, chess.BLACK),
    ]
    idx = 0
    for pt, color in piece_order:
        for sq in board.pieces(pt, color):
            feats[idx + sq] = 1.0
        idx += 64
    feats[768] = float(board.has_kingside_castling_rights(chess.WHITE))
    feats[769] = float(board.has_queenside_castling_rights(chess.WHITE))
    feats[770] = float(board.has_kingside_castling_rights(chess.BLACK))
    feats[771] = float(board.has_queenside_castling_rights(chess.BLACK))
    feats[772] = float(board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK))
    feats[773] = 1.0 if board.turn == chess.WHITE else 0.0
    feats[774] = min(board.halfmove_clock / 100.0, 1.0)
    feats[775] = min(board.fullmove_number / 200.0, 1.0)
    return feats


# ---------------- Teacher targets ----------------
PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 325, chess.BISHOP: 335, chess.ROOK: 500, chess.QUEEN: 975, chess.KING: 0}
PST_PAWN = [
   0,  0,  0,  0,  0,  0,  0,  0,
   5, 10, 10,-20,-20, 10, 10,  5,
   5,  5, 10, 15, 15, 10,  5,  5,
   0,  0,  0, 20, 20,  0,  0,  0,
   5,  5, 10, 25, 25, 10,  5,  5,
  10, 10, 20, 30, 30, 20, 10, 10,
  50, 50, 50, 50, 50, 50, 50, 50,
   0,  0,  0,  0,  0,  0,  0,  0,
]
PST_KNIGHT = [
  -50,-40,-30,-30,-30,-30,-40,-50,
  -40,-20,  0,  5,  5,  0,-20,-40,
  -30,  5, 10, 15, 15, 10,  5,-30,
  -30,  0, 15, 20, 20, 15,  0,-30,
  -30,  5, 15, 20, 20, 15,  5,-30,
  -30,  0, 10, 15, 15, 10,  0,-30,
  -40,-20,  0,  0,  0,  0,-20,-40,
  -50,-40,-30,-30,-30,-30,-40,-50,
]
PST_BISHOP = [
  -20,-10,-10,-10,-10,-10,-10,-20,
  -10,  5,  0,  0,  0,  0,  5,-10,
  -10, 10, 10, 10, 10, 10, 10,-10,
  -10,  0, 10, 10, 10, 10,  0,-10,
  -10,  5,  5, 10, 10,  5,  5,-10,
  -10,  0,  5, 10, 10,  5,  0,-10,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -20,-10,-10,-10,-10,-10,-10,-20,
]
PST_ROOK = [
   0,  0,  0,  5,  5,  0,  0,  0,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
   5, 10, 10, 10, 10, 10, 10,  5,
   0,  0,  0,  0,  0,  0,  0,  0,
]
PST_QUEEN = [
  -20,-10,-10, -5, -5,-10,-10,-20,
  -10,  0,  5,  0,  0,  0,  0,-10,
  -10,  5,  5,  5,  5,  5,  0,-10,
    0,  0,  5,  5,  5,  5,  0, -5,
   -5,  0,  5,  5,  5,  5,  0, -5,
  -10,  0,  5,  5,  5,  5,  0,-10,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -20,-10,-10, -5, -5,-10,-10,-20,
]
PST_KING = [
   20, 30, 10,  0,  0, 10, 30, 20,
   20, 20,  0,  0,  0,  0, 20, 20,
  -10,-20,-20,-20,-20,-20,-20,-10,
  -20,-30,-30,-40,-40,-30,-30,-20,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
]
PST_MAP = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING,
}


def classical_eval_cp(board: chess.Board) -> int:
    score = 0
    for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
        for pt, val in PIECE_VALUES.items():
            pcs = board.pieces(pt, color)
            for sq in pcs:
                score += sign * val
                pst = PST_MAP[pt]
                idx = sq if color == chess.WHITE else chess.square_mirror(sq)
                score += sign * pst[idx]
    # Mobility proxy
    score += (board.legal_moves.count())
    return score if board.turn == chess.WHITE else -score


# ---------------- Model and loaders ----------------
class TorchNNUE(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def load_from_any(path: str, input_dim: int = BASE_INPUT_DIM) -> Tuple[TorchNNUE, int, int]:
    """Load model weights from .nnue, .npz, or .pt and return (model, h1, h2)."""
    p = os.path.abspath(path)
    if p.endswith('.nnue'):
        with open(p, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=3)
            in_dim, h1, h2 = map(int, header)
            assert in_dim == input_dim, f"Input dim mismatch: {in_dim} != {input_dim}"
            W1 = np.fromfile(f, dtype=np.float32, count=in_dim*h1).reshape(h1, in_dim)
            b1 = np.fromfile(f, dtype=np.float32, count=h1)
            W2 = np.fromfile(f, dtype=np.float32, count=h1*h2).reshape(h2, h1)
            b2 = np.fromfile(f, dtype=np.float32, count=h2)
            W3 = np.fromfile(f, dtype=np.float32, count=h2)
            b3 = np.fromfile(f, dtype=np.float32, count=1)
        model = TorchNNUE(in_dim, h1, h2)
        sd = model.state_dict()
        sd['fc1.weight'] = torch.tensor(W1, dtype=torch.float32)
        sd['fc1.bias'] = torch.tensor(b1, dtype=torch.float32)
        sd['fc2.weight'] = torch.tensor(W2, dtype=torch.float32)
        sd['fc2.bias'] = torch.tensor(b2, dtype=torch.float32)
        sd['fc3.weight'] = torch.tensor(W3.reshape(1, -1), dtype=torch.float32)
        sd['fc3.bias'] = torch.tensor(b3.reshape(1), dtype=torch.float32)
        model.load_state_dict(sd)
        return model, h1, h2
    if p.endswith('.npz'):
        data = np.load(p, allow_pickle=True)
        # Try multiple key layouts
        choices = [
            ('fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc3.weight','fc3.bias'),
            ('W1','b1','W2','b2','Wout','bout'),
            ('layer1.weight','layer1.bias','layer2.weight','layer2.bias','out.weight','out.bias'),
        ]
        tensors = None
        for keys in choices:
            if all(k in data for k in keys):
                tensors = {k: data[k] for k in keys}
                break
        if tensors is None:
            raise RuntimeError('NPZ missing expected keys')
        W1 = np.array(tensors[list(tensors.keys())[0]], dtype=np.float32)
        h1 = int(W1.shape[0])
        W2 = np.array(tensors[list(tensors.keys())[2]], dtype=np.float32)
        h2 = int(W2.shape[0])
        model = TorchNNUE(input_dim, h1, h2)
        sd = model.state_dict()
        sd['fc1.weight'] = torch.tensor(W1, dtype=torch.float32)
        sd['fc1.bias'] = torch.tensor(np.array(tensors[list(tensors.keys())[1]]), dtype=torch.float32)
        sd['fc2.weight'] = torch.tensor(W2, dtype=torch.float32)
        sd['fc2.bias'] = torch.tensor(np.array(tensors[list(tensors.keys())[3]]), dtype=torch.float32)
        W3 = np.array(tensors[list(tensors.keys())[4]], dtype=np.float32)
        b3 = np.array(tensors[list(tensors.keys())[5]], dtype=np.float32)
        sd['fc3.weight'] = torch.tensor(W3.reshape(1, -1), dtype=torch.float32)
        sd['fc3.bias'] = torch.tensor(b3.reshape(1), dtype=torch.float32)
        model.load_state_dict(sd)
        return model, h1, h2
    # .pt/.pth
    state = torch.load(p, map_location='cpu')
    if isinstance(state, nn.Module):
        state = state.state_dict()
    # Infer dims
    W1 = state.get('fc1.weight'); W2 = state.get('fc2.weight')
    if W1 is None or W2 is None:
        raise RuntimeError('Checkpoint missing fc weights')
    h1 = int(W1.shape[0]); h2 = int(W2.shape[0])
    model = TorchNNUE(input_dim, h1, h2)
    model.load_state_dict({k: v.float() for k,v in state.items() if k in model.state_dict()})
    return model, h1, h2


def write_nnue(out_path: str, model: TorchNNUE) -> None:
    sd = model.state_dict()
    W1 = sd['fc1.weight'].detach().cpu().numpy().astype(np.float32)
    b1 = sd['fc1.bias'].detach().cpu().numpy().astype(np.float32)
    W2 = sd['fc2.weight'].detach().cpu().numpy().astype(np.float32)
    b2 = sd['fc2.bias'].detach().cpu().numpy().astype(np.float32)
    W3 = sd['fc3.weight'].detach().cpu().numpy().astype(np.float32).reshape(-1)
    b3 = sd['fc3.bias'].detach().cpu().numpy().astype(np.float32).reshape(1)
    in_dim = int(W1.shape[1]); h1 = int(W1.shape[0]); h2 = int(W2.shape[0])
    with open(out_path, 'wb') as f:
        np.array([in_dim, h1, h2], dtype=np.int32).tofile(f)
        W1.tofile(f); b1.tofile(f); W2.tofile(f); b2.tofile(f); W3.tofile(f); b3.tofile(f)


# ---------------- Datasets ----------------
class FenDataset(Dataset):
    def __init__(self, fen_file: str, teacher: str = 'classical', engine_path: str = '', engine_time: float = 0.03, engine_depth: Optional[int] = None):
        self.fens: list[str] = []
        self.labels: list[Optional[float]] = []  # cp if provided inline
        with open(fen_file) as f:
            for line in f:
                s = line.strip()
                if not s: continue
                # Allow format: "<fen> | cp=<value>"
                cp_val = None
                parts = s.split('|')
                fen = parts[0].strip()
                if len(parts) > 1 and 'cp=' in parts[1]:
                    try:
                        cp_val = float(parts[1].split('cp=')[1].strip())
                    except Exception:
                        cp_val = None
                self.fens.append(fen)
                self.labels.append(cp_val)

        self.teacher = teacher
        self.engine_path = engine_path
        self.engine_time = engine_time
        self.engine_depth = engine_depth
        # Lazy engine init per worker to avoid pickling issues
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        feats = features_from_fen(fen)
        cp = self.labels[idx]
        if cp is None:
            board = chess.Board(fen)
            # Lazily create engine in worker process when needed
            if self.teacher == 'self' and self._engine is None:
                try:
                    self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
                except Exception:
                    print('[WARN] Failed to launch UCI engine in worker; falling back to classical labels')
                    self.teacher = 'classical'
            if self.teacher == 'self' and self._engine is not None:
                limit = chess.engine.Limit(depth=self.engine_depth) if self.engine_depth else chess.engine.Limit(time=self.engine_time)
                try:
                    info = self._engine.analyse(board, limit)
                    score_w = info['score'].white().score(mate_score=30000)
                    score_w = float(score_w if score_w is not None else 0.0)
                    cp = score_w if board.turn == chess.WHITE else -score_w
                except Exception:
                    cp = float(classical_eval_cp(board))
            else:
                cp = float(classical_eval_cp(board))
        # Normalize to [-5,5] caps then scale to [-1,1]ish
        cp = max(min(cp, 5000.0), -5000.0) / 1000.0
        return torch.from_numpy(feats), torch.tensor(cp, dtype=torch.float32)

    def close(self):
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass

    # Ensure dataset is pickleable for DataLoader workers
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_engine'] = None
        return state


def _generate_selfplay_chunk(n_positions: int, sample_every_plies: int,
                             teacher: str, engine_path: str, engine_time: float, engine_depth: Optional[int], seed: int
                             ) -> list[Tuple[np.ndarray, float]]:
    rng = random.Random(seed)
    positions: list[Tuple[np.ndarray, float]] = []
    _engine: Optional[chess.engine.SimpleEngine] = None
    if teacher == 'self':
        try:
            _engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        except Exception:
            teacher = 'classical'
    try:
        count = 0
        while count < n_positions:
            board = chess.Board()
            plies = 0
            while not board.is_game_over() and plies < 240 and count < n_positions:
                legal = list(board.legal_moves)
                # Mild randomness to diversify positions across workers
                mv = legal[plies % len(legal)] if len(legal) == 0 else legal[rng.randrange(len(legal))]
                board.push(mv)
                plies += 1
                if plies % sample_every_plies != 0:
                    continue
                feats = features_from_fen(board.fen())
                if teacher == 'self' and _engine is not None:
                    limit = chess.engine.Limit(depth=engine_depth) if engine_depth else chess.engine.Limit(time=engine_time)
                    try:
                        info = _engine.analyse(board, limit)
                        score_w = info['score'].white().score(mate_score=30000)
                        score_w = float(score_w if score_w is not None else 0.0)
                        cp = score_w if board.turn == chess.WHITE else -score_w
                    except Exception:
                        cp = classical_eval_cp(board)
                else:
                    cp = classical_eval_cp(board)
                positions.append((feats, max(min(cp, 5000.0), -5000.0) / 1000.0))
                count += 1
    finally:
        if _engine is not None:
            try:
                _engine.quit()
            except Exception:
                pass
    return positions


class SelfPlayDataset(Dataset):
    def __init__(self, n_positions: int, search_depth: int = 2, sample_every_plies: int = 2,
                 teacher: str = 'classical', engine_path: str = '', engine_time: float = 0.03, engine_depth: Optional[int] = None,
                 show_progress: bool = True, gen_workers: int = 0):
        self.positions: list[Tuple[np.ndarray, float]] = []
        self.teacher = teacher
        self.engine_time = engine_time
        self.engine_depth = engine_depth

        gen_workers = max(0, min(6, int(gen_workers or 0)))
        if gen_workers > 1:
            # Parallel generation across processes; each worker opens its own engine if needed
            total = n_positions
            # Choose a smaller chunk to give frequent progress updates (~1–3s per chunk)
            est_per_pos = max(0.01, float(engine_time or 0.02))
            target_chunk_time = 0.01  # seconds — faster progress updates
            est_chunk = int(target_chunk_time / est_per_pos)
            chunk = max(10, min(200, est_chunk))
            # Build chunk sizes
            sizes = []
            for start in range(0, total, chunk):
                sizes.append(min(chunk, total - start))

            pbar = tqdm(total=total, desc="Self-play gen", unit="pos", dynamic_ncols=True, mininterval=0.1) if show_progress else None
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=gen_workers) as ex:
                futures = []
                for i, n_chunk in enumerate(sizes):
                    seed = (int(t0 * 1000) ^ (i * 7919)) & 0x7FFFFFFF
                    futures.append(ex.submit(
                        _generate_selfplay_chunk, n_chunk, sample_every_plies,
                        teacher, engine_path, engine_time, engine_depth, seed
                    ))
                for fut in as_completed(futures):
                    res = fut.result()
                    self.positions.extend(res)
                    if pbar is not None:
                        pbar.update(len(res))
            if pbar is not None:
                elapsed = max(1e-6, time.time() - t0)
                rate = len(self.positions) / elapsed
                pbar.set_postfix_str(f"{rate:.0f} pos/s")
                pbar.close()
            # Trim to exactly n_positions if overshot
            if len(self.positions) > n_positions:
                self.positions = self.positions[:n_positions]
        else:
            # Single-process generation (legacy path)
            _engine: Optional[chess.engine.SimpleEngine] = None
            if teacher == 'self':
                try:
                    _engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                except Exception:
                    print('[WARN] Failed to launch UCI engine; falling back to classical labels')
                    self.teacher = 'classical'
            count = 0
            pbar = tqdm(total=n_positions, desc="Self-play gen", unit="pos", dynamic_ncols=True, mininterval=0.1) if show_progress else None
            t0 = time.time()
            while count < n_positions:
                board = chess.Board()
                plies = 0
                while not board.is_game_over() and plies < 240 and count < n_positions:
                    legal = list(board.legal_moves)
                    mv = legal[plies % len(legal)]
                    board.push(mv)
                    plies += 1
                    if plies % sample_every_plies != 0:
                        continue
                    feats = features_from_fen(board.fen())
                    if self.teacher == 'self' and _engine is not None:
                        limit = chess.engine.Limit(depth=self.engine_depth) if self.engine_depth else chess.engine.Limit(time=self.engine_time)
                        try:
                            info = _engine.analyse(board, limit)
                            score_w = info['score'].white().score(mate_score=30000)
                            score_w = float(score_w if score_w is not None else 0.0)
                            cp = score_w if board.turn == chess.WHITE else -score_w
                        except Exception:
                            cp = classical_eval_cp(board)
                    else:
                        cp = classical_eval_cp(board)
                    self.positions.append((feats, max(min(cp, 5000.0), -5000.0) / 1000.0))
                    count += 1
                    if pbar is not None:
                        pbar.update(1)
            if pbar is not None:
                elapsed = max(1e-6, time.time() - t0)
                rate = count / elapsed
                pbar.set_postfix_str(f"{rate:.0f} pos/s")
                pbar.close()
            if _engine is not None:
                try:
                    _engine.quit()
                except Exception:
                    pass

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        feats, cp = self.positions[idx]
        return torch.from_numpy(feats), torch.tensor(cp, dtype=torch.float32)


# ---------------- Training ----------------
@dataclass
class TrainConfig:
    from_path: str
    out_pt: str
    out_nnue: str
    fens: Optional[str] = None
    selfplay: int = 0
    teacher: str = 'classical'  # 'classical' | 'self'
    engine_path: str = ''
    engine_time: float = 0.03
    engine_depth: Optional[int] = None
    epochs: int = 2
    batch: int = 512
    lr: float = 7e-4
    weight_decay: float = 1e-6
    device: str = 'cpu'
    amp: bool = True
    early_stop_patience: int = 3
    workers: int = 0          # DataLoader workers (processes)
    threads: int = 0          # Torch intra-op threads cap (0 = default)


def train(cfg: TrainConfig) -> None:
    # Cap threads if requested (max 6)
    if cfg.threads and cfg.threads > 0:
        max_threads = min(6, int(cfg.threads))
        try:
            torch.set_num_threads(max_threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(max(1, max_threads // 2))
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))

    device = torch.device(cfg.device if (cfg.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    model, h1, h2 = load_from_any(cfg.from_path, input_dim=BASE_INPUT_DIM)
    model.to(device)

    # Data
    if cfg.fens:
        print(f"[data] Loading FENs from {cfg.fens} (teacher={cfg.teacher})")
        dataset = FenDataset(cfg.fens, cfg.teacher, cfg.engine_path, cfg.engine_time, cfg.engine_depth)
    else:
        # Use the requested selfplay count when provided; default to 100k otherwise
        npos = cfg.selfplay if (cfg.selfplay and cfg.selfplay > 0) else 100000
        print(f"[data] Generating self-play positions: {npos} (teacher={cfg.teacher})")
        if cfg.teacher == 'self':
            print(f"[data] UCI engine labels: depth={cfg.engine_depth or 'n/a'} time={cfg.engine_time}s path={cfg.engine_path}")
            if cfg.engine_depth is None and cfg.engine_time is not None:
                est_sec = npos * max(0.0, cfg.engine_time)
                if est_sec > 0:
                    print(f"[data] Rough gen ETA: {est_sec/60:.1f}m (~{est_sec/3600:.2f}h) + overhead")
        dataset = SelfPlayDataset(npos, teacher=cfg.teacher, engine_path=cfg.engine_path,
                                  engine_time=cfg.engine_time, engine_depth=cfg.engine_depth, show_progress=True,
                                  gen_workers=cfg.workers)
    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)
    max_workers = min(6, cfg.workers if cfg.workers is not None else 0)
    if max_workers < 0:
        max_workers = 0
    dl_kwargs = {}
    if max_workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    train_dl = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=max_workers, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=max_workers, **dl_kwargs)


    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    # AMP + GradScaler setup
    # - Use autocast on CUDA or MPS if requested
    # - Use GradScaler only on CUDA; MPS currently does not benefit and may not be supported
    try:
        from contextlib import nullcontext
    except Exception:
        class nullcontext:  # type: ignore
            def __init__(self, *a, **kw):
                pass
            def __enter__(self):
                return None
            def __exit__(self, *a):
                return False

    use_autocast = bool(cfg.amp and device.type in ("cuda", "mps"))
    def autocast_ctx():  # returns a context manager
        try:
            return torch.autocast(device_type=device.type, enabled=use_autocast)
        except Exception:
            # Fallback to CUDA-only autocast if available
            if device.type == 'cuda':
                return torch.cuda.amp.autocast(enabled=use_autocast)
            return nullcontext()

    try:
        scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.amp and device.type == 'cuda'))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == 'cuda'))

    best_val = float('inf'); patience = cfg.early_stop_patience
    total_start = time.time()
    total_steps = cfg.epochs * max(1, math.ceil(n_train / max(1, cfg.batch)))
    print(f"[model] dims: in={BASE_INPUT_DIM}, h1={h1}, h2={h2}; params={sum(p.numel() for p in model.parameters()):,}")
    print(f"[train] device={device.type}  epochs={cfg.epochs}  batches/epoch={math.ceil(n_train / max(1, cfg.batch))}  batch={cfg.batch}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch}/{cfg.epochs} [train]", unit="batch", dynamic_ncols=True, mininterval=0.05)
        postfix_every = max(1, len(train_dl) // 20)  # update ~20x per epoch
        for i, (xb, yb) in enumerate(pbar, 1):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with autocast_ctx():
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                with autocast_ctx():
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            running += loss.item()
            if i % postfix_every == 0:
                cur_lr = sched.get_last_lr()[0]
                denom = float(postfix_every) if postfix_every > 0 else 1.0
                pbar.set_postfix({"loss": f"{running/denom:.5f}", "lr": f"{cur_lr:.2e}"})
                running = 0.0
        sched.step()
        print(f"[epoch {epoch}] train_time={(time.time()-t0)/60:.1f}m  lr={sched.get_last_lr()[0]:.2e}")

        # Val
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            vpbar = tqdm(val_dl, total=len(val_dl), desc=f"Epoch {epoch}/{cfg.epochs} [val]", unit="batch", dynamic_ncols=True, mininterval=0.1)
            for xb, yb in vpbar:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()
        val_loss /= max(1, len(val_dl))
        print(f"  val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss; patience = cfg.early_stop_patience
            # Save .pt and .nnue on improvement
            os.makedirs(os.path.dirname(cfg.out_pt), exist_ok=True)
            os.makedirs(os.path.dirname(cfg.out_nnue), exist_ok=True)
            torch.save(model.state_dict(), cfg.out_pt)
            write_nnue(cfg.out_nnue, model)
            print(f"  ✅ improved — saved {cfg.out_pt} and {cfg.out_nnue}")
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    # Final save if nothing saved yet
    if not os.path.exists(cfg.out_nnue):
        os.makedirs(os.path.dirname(cfg.out_pt), exist_ok=True)
        os.makedirs(os.path.dirname(cfg.out_nnue), exist_ok=True)
        torch.save(model.state_dict(), cfg.out_pt)
        write_nnue(cfg.out_nnue, model)
        print(f"Saved final {cfg.out_pt} and {cfg.out_nnue}")

    total_dt = time.time() - total_start
    print(f"[done] total_time={total_dt/60:.1f}m")

    # Cleanup teacher
    if isinstance(dataset, FenDataset):
        dataset.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='from_path', type=str, required=True, help='Seed checkpoint (.nnue, .npz, or .pt/.pth)')
    ap.add_argument('--out-pt', type=str, default='training/checkpoints/finetuned.pt')
    ap.add_argument('--out-nnue', type=str, default='training/best_finetuned.nnue')
    ap.add_argument('--fens', type=str, default=None, help='Path to FENs file (optional cp labels via "| cp=<val>")')
    ap.add_argument('--selfplay', type=int, default=0, help='If no FENs, generate N self-play positions')
    ap.add_argument('--teacher', type=str, default='classical', choices=['classical','self'])
    ap.add_argument('--engine-path', type=str, default='', help='Path to your UCI engine (required for --teacher self)')
    ap.add_argument('--engine-time', type=float, default=0.03)
    ap.add_argument('--engine-depth', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=7e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-6)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--workers', type=int, default=0, help='DataLoader workers (0..6)')
    ap.add_argument('--threads', type=int, default=0, help='Torch compute threads cap (0..6)')
    ap.add_argument('--log-file', type=str, default='', help='Mirror stdout/stderr to this file (live)')
    args = ap.parse_args()

    # Configure live logging if requested
    _log_fh = _configure_io_and_logging(args.log_file or None)

    cfg = TrainConfig(
        from_path=args.from_path,
        out_pt=args.out_pt,
        out_nnue=args.out_nnue,
        fens=args.fens,
        selfplay=args.selfplay,
        teacher=args.teacher,
        engine_path=args.engine_path,
        engine_time=args.engine_time,
        engine_depth=args.engine_depth,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        amp=not args.no_amp,
        early_stop_patience=args.patience,
        workers=max(0, min(6, int(args.workers or 0))),
        threads=max(0, min(6, int(args.threads or 0))),
    )
    try:
        train(cfg)
    finally:
        if _log_fh is not None:
            try:
                _log_fh.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
