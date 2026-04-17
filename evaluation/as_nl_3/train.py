#!/usr/bin/env python3
"""
train.py — Part 1 CNN-Transformer baseline (as_nl_3)
=====================================================

Saves ALL architecture hyperparameters into norm_stats.pt alongside the
normalization statistics.  get_model() in model.py reads them back so the
reconstructed model always has the exact same shape as the saved checkpoint.

Usage:
    python train.py [--epochs 50] [--batch_size 2] [--lr 1e-4]
                    [--history_len 96] [--grid_size 10] [--embed_dim 64]
                    [--n_layers 2] [--n_heads 4] [--mlp_dim 256]
                    [--dropout 0.1] [--n_train_days 300] [--n_val_days 30]
                    [--save_dir ./checkpoints] [--seed 42]
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Allow importing model.py from the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CNNTransformerModel, make_time_feats

# ============================================================
# Paths
# ============================================================

ROOT        = Path("/cluster/tufts/c26sp1cs0137/data/assignment3_data")
WEATHER_DIR = ROOT / "weather_data"
ENERGY_DIR  = ROOT / "energy_demand_data"

FUTURE_LEN  = 24
TRAIN_YEARS = [2021, 2022]

# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--history_len",  type=int,   default=48,
                   help="Look-back window in hours (≤168).")
    p.add_argument("--grid_size",    type=int,   default=10,
                   help="CNN output grid size; P = grid_size².")
    p.add_argument("--embed_dim",    type=int,   default=64)
    p.add_argument("--n_layers",     type=int,   default=2)
    p.add_argument("--n_heads",      type=int,   default=4)
    p.add_argument("--mlp_dim",      type=int,   default=256)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--n_train_days", type=int,   default=300)
    p.add_argument("--n_val_days",   type=int,   default=30)
    p.add_argument("--save_dir",     type=str,   default="./checkpoints")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()

# ============================================================
# Data helpers
# ============================================================

_wcache: dict = {}

def load_weather(hour_int64: int) -> torch.Tensor:
    """Return (450, 449, 7) float32 tensor for the given hours-since-epoch."""
    if hour_int64 in _wcache:
        return _wcache[hour_int64]
    dt   = pd.Timestamp(int(hour_int64), unit="h")
    path = WEATHER_DIR / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    t    = torch.load(path, weights_only=True).float()
    _wcache[hour_int64] = t
    if len(_wcache) > 2000:   # ~11 GB cap — safe within 128 GB node
        del _wcache[next(iter(_wcache))]
    return t


def preload_weather(all_hours: np.ndarray, indices: np.ndarray,
                    history_len: int) -> None:
    """
    Load every weather frame needed by the given sample indices into _wcache
    before training begins, eliminating per-batch disk I/O.

    Covers history_len + 24 hours per sample index.
    """
    needed = set()
    for t in indices:
        for h in all_hours[t - history_len : t + FUTURE_LEN]:
            needed.add(int(h))
    needed = sorted(needed)
    print(f"  Preloading {len(needed)} weather frames into RAM …", flush=True)
    loaded = 0
    for h in needed:
        try:
            load_weather(h)
            loaded += 1
        except FileNotFoundError:
            pass
    print(f"  Preloaded {loaded}/{len(needed)} frames  "
          f"(~{loaded * 450 * 449 * 7 * 4 / 1e9:.1f} GB)", flush=True)


def load_energy_df() -> pd.DataFrame:
    dfs = []
    for csv in sorted(ENERGY_DIR.glob("target_energy_zonal_*.csv")):
        dfs.append(pd.read_csv(csv, parse_dates=["timestamp_utc"]))
    df = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)
    deltas = df["timestamp_utc"].diff().dropna()
    assert (deltas == pd.Timedelta("1h")).all(), "Gap detected in energy data!"
    return df


def midnight_indices(df: pd.DataFrame, years: list, history_len: int) -> np.ndarray:
    """Row indices at midnight (00:00 UTC) for the given years."""
    mask = (
        df["timestamp_utc"].dt.year.isin(years)
        & (df["timestamp_utc"].dt.hour == 0)
    )
    idxs = np.where(mask)[0]
    return idxs[(idxs >= history_len) & (idxs + FUTURE_LEN <= len(df))]

# ============================================================
# Normalization statistics
# ============================================================

def compute_weather_stats(df: pd.DataFrame, sample_indices: np.ndarray,
                          n: int = 50):
    """Channel-wise mean/std from a random sample of weather frames."""
    print("Computing weather normalization statistics …")
    chosen = np.random.choice(sample_indices, size=min(n, len(sample_indices)),
                              replace=False)
    s1, s2, cnt = torch.zeros(7), torch.zeros(7), 0
    for idx in chosen:
        h = int(df["timestamp_utc"].iloc[idx].timestamp() // 3600)
        try:
            w = load_weather(h).reshape(-1, 7)
        except FileNotFoundError:
            continue
        s1  += w.sum(0)
        s2  += (w ** 2).sum(0)
        cnt += w.shape[0]
    mean = s1 / cnt
    std  = (s2 / cnt - mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
    print(f"  Weather mean: {mean.tolist()}")
    print(f"  Weather std : {std.tolist()}")
    return mean, std


def compute_energy_stats(values: np.ndarray):
    """Per-zone mean/std over (T, Z) array."""
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std  = np.maximum(np.nanstd(values, axis=0).astype(np.float32), 1e-6)
    print(f"  Energy mean per zone: {mean}")
    print(f"  Energy std  per zone: {std}")
    return mean, std

# ============================================================
# Dataset
# ============================================================

class EnergyDataset(Dataset):
    """
    Yields one sample per forecast-midnight window.

    Returns:
        hw     : (S, 450, 449, 7)  raw weather history
        he     : (S, Z)            raw energy history
        fw     : (24, 450, 449, 7) raw future weather
        ft     : (24,)  int64      future hours-since-epoch
        target : (24, Z)           raw energy target (MWh)
    """

    def __init__(self, df, values, all_hours, indices, history_len):
        self.df          = df
        self.values      = values      # (T, Z) float32
        self.all_hours   = all_hours   # (T,)   int64
        self.indices     = indices
        self.history_len = history_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t  = self.indices[i]
        S  = self.history_len
        hs = slice(t - S, t)
        fs = slice(t,     t + FUTURE_LEN)

        hist_hrs = self.all_hours[hs]   # (S,)  int64
        fut_hrs  = self.all_hours[fs]   # (24,) int64

        try:
            hw = torch.stack([load_weather(int(h)) for h in hist_hrs])  # (S, H, W, 7)
            fw = torch.stack([load_weather(int(h)) for h in fut_hrs])   # (24, H, W, 7)
        except FileNotFoundError:
            return None

        he     = torch.from_numpy(self.values[hs].copy())   # (S, Z)  raw
        target = torch.from_numpy(self.values[fs].copy())   # (24, Z) raw
        ft     = torch.tensor(fut_hrs, dtype=torch.int64)   # (24,)

        return hw.float(), he.float(), fw.float(), ft, target.float()


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)

# ============================================================
# Metrics
# ============================================================

def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> float:
    return (((pred - target).abs() / (target.abs() + eps)).mean() * 100).item()

# ============================================================
# Training / validation epoch
# ============================================================

def run_epoch(model, base_model, loader, optimizer, device, train: bool = True):
    """
    base_model: the unwrapped model (handles DataParallel).
    Returns (avg_loss, avg_mape).
    """
    model.train(train)
    total_loss = total_mape = n = 0

    em = base_model.energy_mean.to(device)
    es = base_model.energy_std.to(device)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if batch is None:
                continue

            hw, he, fw, ft, target = [x.to(device) for x in batch]

            # Normalize and compute calendar features (same as evaluate.py)
            outs = base_model.adapt_inputs(hw, he, fw, ft)
            pred = model(*outs)   # (B, 24, Z) raw MWh

            # Loss in normalized space — avoids high-demand zones dominating
            pred_n   = (pred   - em) / (es + 1e-6)
            target_n = (target - em) / (es + 1e-6)
            loss = ((pred_n - target_n) ** 2).mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            B = hw.size(0)
            total_loss += loss.item() * B
            total_mape += mape(pred.detach(), target.detach()) * B
            n += B

    return total_loss / max(n, 1), total_mape / max(n, 1)

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load energy data ─────────────────────────────────────────────────────
    print("Loading energy data …")
    df      = load_energy_df()
    zones   = [c for c in df.columns if c != "timestamp_utc"]
    n_zones = len(zones)
    values  = df[zones].values.astype(np.float32)     # (T, Z)
    hours   = df["timestamp_utc"].values.astype("datetime64[h]").astype(np.int64)
    print(f"  Zones: {n_zones}   Rows: {len(df)}")

    # ── Split indices ─────────────────────────────────────────────────────────
    all_idxs = midnight_indices(df, TRAIN_YEARS, args.history_len)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_idxs)
    val_idxs   = all_idxs[:args.n_val_days]
    train_idxs = all_idxs[args.n_val_days : args.n_val_days + args.n_train_days]
    print(f"  Train days: {len(train_idxs)}   Val days: {len(val_idxs)}")

    # ── Normalization stats ───────────────────────────────────────────────────
    print("Computing normalization statistics …")
    w_mean, w_std = compute_weather_stats(df, train_idxs)

    print("Computing energy statistics …")
    train_year_mask = df["timestamp_utc"].dt.year.isin(TRAIN_YEARS)
    e_mean, e_std   = compute_energy_stats(values[train_year_mask])

    # ── Save norm stats + ALL architecture hyperparameters ───────────────────
    # This is the key to preventing architecture mismatches at evaluation time.
    norm_stats = {
        "weather_mean": w_mean.tolist(),
        "weather_std":  w_std.tolist(),
        "energy_mean":  e_mean.tolist(),
        "energy_std":   e_std.tolist(),
        "history_len":  args.history_len,
        "embed_dim":    args.embed_dim,
        "n_layers":     args.n_layers,
        "n_heads":      args.n_heads,
        "mlp_dim":      args.mlp_dim,
        "grid_size":    args.grid_size,
        "dropout":      args.dropout,
    }
    torch.save(norm_stats, save_dir / "norm_stats.pt")
    print(f"  Norm stats saved to {save_dir / 'norm_stats.pt'}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    ds_train = EnergyDataset(df, values, hours, train_idxs, args.history_len)
    ds_val   = EnergyDataset(df, values, hours, val_idxs,   args.history_len)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=1, collate_fn=collate, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=1, collate_fn=collate, pin_memory=True)

    # ── Build model ───────────────────────────────────────────────────────────
    base_model = CNNTransformerModel(
        n_zones      = n_zones,
        history_len  = args.history_len,
        future_len   = FUTURE_LEN,
        grid_size    = args.grid_size,
        embed_dim    = args.embed_dim,
        n_layers     = args.n_layers,
        n_heads      = args.n_heads,
        mlp_dim      = args.mlp_dim,
        dropout      = args.dropout,
        weather_mean = w_mean,
        weather_std  = w_std,
        energy_mean  = e_mean,
        energy_std   = e_std,
    )

    model = base_model
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(base_model)
    model = model.to(device)

    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # ReduceLROnPlateau: drops LR only when val MAPE stops improving,
    # preventing the aggressive decay of cosine annealing from causing overfitting.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_mape    = float("inf")
    patience     = 0
    max_patience = 10

    print(f"\nTraining for up to {args.epochs} epochs (early stopping patience={max_patience}) …\n")
    print(f" {'Epoch':>6}  {'Train Loss':>12}  {'Val MAPE':>10}  {'LR':>10}  {'Patience':>8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, base_model, dl_train, optimizer, device, train=True)
        _,   val_mape = run_epoch(model, base_model, dl_val,   optimizer, device, train=False)

        lr = optimizer.param_groups[0]["lr"]
        print(f" {epoch:>6}  {train_loss:>12.4f}  {val_mape:>9.2f}%  {lr:>10.2e}  {patience:>8}")

        scheduler.step(val_mape)   # ReduceLROnPlateau uses val metric

        if val_mape < best_mape:
            best_mape = val_mape
            patience  = 0
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            state = base_model.state_dict()   # always save unwrapped state_dict
            torch.save(state, save_dir / f"best_model_{ts}.pt")
            print(f"         ↑ best model saved  (val MAPE {best_mape:.2f} %)")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val MAPE: {best_mape:.2f}%")


if __name__ == "__main__":
    main()
