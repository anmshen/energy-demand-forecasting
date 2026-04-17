#!/usr/bin/env python3
"""
train.py — CNN-Transformer Patch Architecture (improved)
=========================================================

Improvements over baseline:
  1. Spatial crop: computes a tight bounding box from training weather data
     and saves tight_crop.json.  The CNN then processes only the informative
     region (~2.5× fewer FLOPs per frame).
  2. Richer calendar: uses model.make_time_feats() — the same 7-dim function
     as adapt_inputs() — so there is zero train/eval feature skew.
  3. Configurable lookback: --history_len (default 96) is saved in
     norm_stats.pt so get_model() reconstructs the right sequence length.

Usage:
    python train.py [--epochs 50] [--batch_size 4] [--lr 1e-4]
                    [--history_len 96] [--grid_size 10] [--embed_dim 96]
                    [--n_transformer_layers 3] [--n_heads 8] [--mlp_dim 384]
                    [--dropout 0.2] [--n_train_days 300] [--n_val_days 30]
                    [--save_dir ./checkpoints] [--seed 42]
"""

import argparse
import json
import math
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
from model import CNNTransformerModel, make_time_feats   # shared calendar fn

# ============================================================
# Cluster paths  (match evaluate.py)
# ============================================================
ROOT        = Path('/cluster/tufts/c26sp1cs0137/data/assignment3_data')
WEATHER_DIR = ROOT / "weather_data"
ENERGY_DIR  = ROOT / "energy_demand_data"

FUTURE_LEN  = 24
TRAIN_YEARS = [2021, 2022]

# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",               type=int,   default=50)
    p.add_argument("--batch_size",           type=int,   default=4)
    p.add_argument("--lr",                   type=float, default=1e-4)
    p.add_argument("--history_len",          type=int,   default=96,
                   help="Lookback window in hours (≤168). Shorter = faster attention.")
    p.add_argument("--grid_size",            type=int,   default=10)
    p.add_argument("--embed_dim",            type=int,   default=96)
    p.add_argument("--n_transformer_layers", type=int,   default=3)
    p.add_argument("--n_heads",              type=int,   default=8)
    p.add_argument("--mlp_dim",              type=int,   default=384)
    p.add_argument("--dropout",              type=float, default=0.2)
    p.add_argument("--n_train_days",         type=int,   default=300)
    p.add_argument("--n_val_days",           type=int,   default=30)
    p.add_argument("--save_dir",             type=str,   default="./checkpoints")
    p.add_argument("--seed",                 type=int,   default=42)
    return p.parse_args()

# ============================================================
# Data helpers
# ============================================================

_weather_cache: dict = {}

def load_weather(hour_int64: int) -> torch.Tensor:
    """Return (450, 449, 7) float32 tensor for the given hours-since-epoch."""
    if hour_int64 in _weather_cache:
        return _weather_cache[hour_int64]
    dt   = pd.Timestamp(int(hour_int64), unit="h")
    path = WEATHER_DIR / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    t    = torch.load(path, weights_only=True).float()
    _weather_cache[hour_int64] = t
    if len(_weather_cache) > 400:
        del _weather_cache[next(iter(_weather_cache))]
    return t


def load_energy_df() -> pd.DataFrame:
    dfs = []
    for csv_path in sorted(ENERGY_DIR.glob("target_energy_zonal_*.csv")):
        dfs.append(pd.read_csv(csv_path, parse_dates=["timestamp_utc"]))
    df     = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)
    deltas = df["timestamp_utc"].diff().dropna()
    assert (deltas == pd.Timedelta("1h")).all(), "Energy data has gaps!"
    return df


def get_midnight_indices(energy_df: pd.DataFrame, years: list, history_len: int) -> np.ndarray:
    """Row indices at midnight for the given years, with enough history/future."""
    mask = (
        energy_df["timestamp_utc"].dt.year.isin(years) &
        (energy_df["timestamp_utc"].dt.hour == 0)
    )
    idxs = np.where(mask)[0]
    idxs = idxs[
        (idxs >= history_len) &
        (idxs + FUTURE_LEN <= len(energy_df))
    ]
    return idxs

# ============================================================
# Normalization statistics
# ============================================================

def compute_weather_stats(
    energy_df: pd.DataFrame,
    sample_indices: np.ndarray,
    n_samples: int = 50,
) -> tuple:
    """
    Estimate channel-wise mean/std from a random sample of weather files.
    Returns two (7,) float32 tensors: (mean, std).
    """
    print("Computing weather normalization statistics …")
    chosen = np.random.choice(sample_indices,
                              size=min(n_samples, len(sample_indices)),
                              replace=False)
    ch_sum  = torch.zeros(7)
    ch_sum2 = torch.zeros(7)
    count   = 0
    for t_idx in chosen:
        h = int(energy_df["timestamp_utc"].iloc[t_idx].timestamp() // 3600)
        try:
            w = load_weather(h)         # (450, 449, 7)
        except FileNotFoundError:
            continue
        flat      = w.reshape(-1, 7)
        ch_sum   += flat.sum(0)
        ch_sum2  += (flat ** 2).sum(0)
        count    += flat.shape[0]
    mean = ch_sum / count
    std  = (ch_sum2 / count - mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
    print(f"  Weather mean: {mean.tolist()}")
    print(f"  Weather std : {std.tolist()}")
    return mean, std


def compute_energy_stats(energy_values: np.ndarray) -> tuple:
    """
    Per-zone mean and std from the provided energy array (T, Z).
    Returns two (Z,) float32 arrays.
    """
    mean = np.nanmean(energy_values, axis=0).astype(np.float32)
    std  = np.nanstd( energy_values, axis=0).astype(np.float32)
    std  = np.maximum(std, 1e-6)
    print(f"  Energy mean per zone: {mean}")
    print(f"  Energy std  per zone: {std}")
    return mean, std


def compute_tight_crop(
    energy_df: pd.DataFrame,
    sample_indices: np.ndarray,
    n_samples: int = 40,
) -> dict:
    """
    Find a tight bounding box of geographically informative pixels by looking
    at the spatial variance of channel means across sampled weather frames.

    Returns a dict with keys y_min, y_max, x_min, x_max (Python ints).
    """
    print("Computing tight spatial crop …")
    chosen = np.random.choice(sample_indices,
                              size=min(n_samples, len(sample_indices)),
                              replace=False)
    accum = None
    count = 0
    for t_idx in chosen:
        h = int(energy_df["timestamp_utc"].iloc[t_idx].timestamp() // 3600)
        try:
            w = load_weather(h).numpy()   # (450, 449, 7)
        except FileNotFoundError:
            continue
        accum  = w if accum is None else accum + w
        count += 1

    if count == 0:
        print("  WARNING: no weather files loaded — using full extent.")
        return {"y_min": 0, "y_max": 450, "x_min": 0, "x_max": 449}

    mean_map = accum / count                             # (450, 449, 7)
    # Use std across channels as a proxy for geographic signal
    signal   = mean_map.std(axis=-1)                    # (450, 449)
    threshold = np.percentile(signal[signal > 0], 10)
    mask     = signal > threshold

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        print("  WARNING: empty crop mask — using full extent.")
        return {"y_min": 0, "y_max": 450, "x_min": 0, "x_max": 449}

    # Add a small margin so edge features are not cut off
    margin = 5
    H, W   = signal.shape
    crop   = {
        "y_min": max(0,   int(rows[0])  - margin),
        "y_max": min(H,   int(rows[-1]) + margin + 1),
        "x_min": max(0,   int(cols[0])  - margin),
        "x_max": min(W,   int(cols[-1]) + margin + 1),
    }
    crop_h = crop["y_max"] - crop["y_min"]
    crop_w = crop["x_max"] - crop["x_min"]
    print(f"  Tight crop: y={crop['y_min']}:{crop['y_max']}  "
          f"x={crop['x_min']}:{crop['x_max']}  "
          f"({crop_h}×{crop_w} from 450×449)")
    return crop

# ============================================================
# Dataset
# ============================================================

class EnergyDataset(Dataset):
    """
    Returns one sample per midnight window:
        hist_weather  : (S, 450, 449, 7)   raw  (crop + norm done in model)
        hist_energy   : (S, n_zones)        normalized
        fut_weather   : (24, 450, 449, 7)   raw
        hist_time_feats: (S, 7)
        fut_time_feats : (24, 7)
        target         : (24, n_zones)       raw MWh

    NOTE: weather is passed raw to keep the Dataset simple.  The CNN's crop
    and normalization buffers handle preprocessing.  Energy is normalized here
    (as in the original baseline) because the loss is computed in normalized
    space (see mse_loss below).
    """

    def __init__(
        self,
        energy_df:     pd.DataFrame,
        energy_values: np.ndarray,        # (T, Z) float32
        all_hours:     np.ndarray,        # (T,)   int64
        indices:       np.ndarray,
        history_len:   int,
        weather_mean:  torch.Tensor,      # (7,)
        weather_std:   torch.Tensor,      # (7,)
        energy_mean:   np.ndarray,        # (Z,)
        energy_std:    np.ndarray,        # (Z,)
    ):
        self.energy_df     = energy_df
        self.energy_values = energy_values
        self.all_hours     = all_hours
        self.indices       = indices
        self.history_len   = history_len
        self.w_mean        = weather_mean
        self.w_std         = weather_std
        self.e_mean        = energy_mean
        self.e_std         = energy_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t_idx  = self.indices[i]
        S      = self.history_len
        hs     = slice(t_idx - S,       t_idx)
        fs     = slice(t_idx,           t_idx + FUTURE_LEN)

        hist_hours   = self.all_hours[hs]     # (S,)  int64
        future_hours = self.all_hours[fs]     # (24,) int64

        # ── Historical energy (normalized) ────────────────────────────────────
        hist_energy = self.energy_values[hs].copy()        # (S, Z)
        hist_energy = (hist_energy - self.e_mean) / self.e_std

        # ── Weather (raw — CNN will crop + normalize via buffers) ─────────────
        try:
            hist_weather = torch.stack([load_weather(int(h)) for h in hist_hours])
        except FileNotFoundError:
            return None

        try:
            fut_weather = torch.stack([load_weather(int(h)) for h in future_hours])
        except FileNotFoundError:
            return None

        # ── Calendar features (7-dim, same fn as adapt_inputs) ───────────────
        hist_time_feats = make_time_feats(
            torch.tensor(hist_hours, dtype=torch.int64).float()
        )   # (S, 7)
        fut_time_feats = make_time_feats(
            torch.tensor(future_hours, dtype=torch.int64).float()
        )   # (24, 7)

        # ── Target (raw MWh — loss computed on raw scale) ─────────────────────
        target = torch.from_numpy(self.energy_values[fs].copy())   # (24, Z)

        return (
            hist_weather.float(),                                   # (S, 450, 449, 7)
            torch.from_numpy(hist_energy.copy()).float(),           # (S, Z)
            fut_weather.float(),                                    # (24, 450, 449, 7)
            hist_time_feats.float(),                                # (S, 7)
            fut_time_feats.float(),                                 # (24, 7)
            target.float(),                                         # (24, Z)
        )


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)

# ============================================================
# Loss / metric
# ============================================================

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def mape_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    return ((pred - target).abs() / (target.abs() + eps)).mean() * 100

# ============================================================
# Training / validation epoch
# ============================================================

def run_epoch(
    model:     CNNTransformerModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    train:     bool = True,
) -> tuple:
    model.train() if train else model.eval()
    total_loss = total_mape = n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if batch is None:
                continue

            hist_w, hist_e, fut_w, hist_tf, fut_tf, target = [
                x.to(device) for x in batch
            ]

            # ── Normalize weather inside the forward pass ─────────────────────
            # We normalize here rather than in the Dataset so the raw tensors
            # are passed, matching what adapt_inputs() will receive at eval time.
            w_mean = model.weather_mean.to(device)
            w_std  = model.weather_std.to(device)
            hist_w_n = (hist_w - w_mean) / (w_std + 1e-6)
            fut_w_n  = (fut_w  - w_mean) / (w_std + 1e-6)

            pred = model(hist_w_n, hist_e, fut_w_n, hist_tf, fut_tf)  # (B, 24, Z)
            loss = mse_loss(pred, target)
            mape = mape_metric(pred, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            total_mape += mape.item()
            n          += 1

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
    print(f"Device      : {device}")
    print(f"History len : {args.history_len} h")

    # ── Load energy data ──────────────────────────────────────────────────────
    print("\nLoading energy data …")
    energy_df     = load_energy_df()
    zone_cols     = [c for c in energy_df.columns if c != "timestamp_utc"]
    n_zones       = len(zone_cols)
    energy_values = energy_df[zone_cols].values.astype(np.float32)
    all_hours     = (
        energy_df["timestamp_utc"]
        .values
        .astype("datetime64[h]")
        .astype(np.int64)
    )
    print(f"  Zones: {n_zones}   Rows: {len(energy_df)}")

    # ── Train / val split ─────────────────────────────────────────────────────
    all_train_idx = get_midnight_indices(energy_df, TRAIN_YEARS, args.history_len)
    val_idx       = all_train_idx[-args.n_val_days:]
    train_idx     = all_train_idx[:-args.n_val_days]

    if args.n_train_days < len(train_idx):
        chosen    = np.random.choice(len(train_idx), args.n_train_days, replace=False)
        train_idx = train_idx[np.sort(chosen)]

    print(f"  Train days: {len(train_idx)}   Val days: {len(val_idx)}")

    # ── Normalization stats ───────────────────────────────────────────────────
    print("\nComputing normalization statistics …")
    w_mean, w_std = compute_weather_stats(energy_df, train_idx, n_samples=50)

    train_energy_rows = np.unique(np.clip(
        np.concatenate([
            np.arange(t - args.history_len, t + FUTURE_LEN) for t in train_idx
        ]),
        0, len(energy_values) - 1,
    ))
    e_mean, e_std = compute_energy_stats(energy_values[train_energy_rows])

    # ── Tight spatial crop ────────────────────────────────────────────────────
    crop      = compute_tight_crop(energy_df, train_idx, n_samples=40)
    crop_path = save_dir / "tight_crop.json"
    with open(crop_path, "w") as f:
        json.dump(crop, f, indent=2)
    print(f"  Tight crop saved to {crop_path}")

    crop_y = (crop["y_min"], crop["y_max"])
    crop_x = (crop["x_min"], crop["x_max"])

    # ── Save all stats (including history_len) ────────────────────────────────
    stats = {
        "weather_mean": w_mean.tolist(),
        "weather_std":  w_std.tolist(),
        "energy_mean":  e_mean.tolist(),
        "energy_std":   e_std.tolist(),
        "history_len":  args.history_len,    # ← saved so get_model() can use it
    }
    torch.save(stats, save_dir / "norm_stats.pt")
    print(f"  Norm stats saved to {save_dir / 'norm_stats.pt'}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    mk_ds = lambda idxs: EnergyDataset(
        energy_df, energy_values, all_hours, idxs,
        args.history_len, w_mean, w_std, e_mean, e_std,
    )
    train_ds = mk_ds(train_idx)
    val_ds   = mk_ds(val_idx)

    mk_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=2, collate_fn=collate_skip_none, pin_memory=True,
    )
    train_loader = mk_loader(train_ds, shuffle=True)
    val_loader   = mk_loader(val_ds,   shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model = CNNTransformerModel(
        n_zones              = n_zones,
        history_len          = args.history_len,
        future_len           = FUTURE_LEN,
        grid_size            = args.grid_size,
        embed_dim            = args.embed_dim,
        n_transformer_layers = args.n_transformer_layers,
        n_heads              = args.n_heads,
        mlp_dim              = args.mlp_dim,
        dropout              = args.dropout,
        crop_y               = crop_y,
        crop_x               = crop_x,
        weather_mean         = w_mean,
        weather_std          = w_std,
        energy_mean          = e_mean,
        energy_std           = e_std,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7,
    )

    # ── Training loop with early stopping ────────────────────────────────────
    best_val_loss    = float("inf")
    patience         = 10
    patience_counter = 0

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(early-stopping patience={patience}) …\n")
    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Train MAPE':>11}  "
          f"{'Val MSE':>9}  {'Val MAPE':>10}  {'LR':>9}  {'Wait':>4}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mape = run_epoch(model, train_loader, optimizer, device, train=True)
        vl_loss, vl_mape = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_mape:>10.2f}%  "
              f"{vl_loss:>9.4f}  {vl_mape:>9.2f}%  {lr_now:>9.2e}  {patience_counter:>4}")

        if vl_loss < best_val_loss:
            best_val_loss    = vl_loss
            patience_counter = 0
            ts               = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt             = save_dir / f"best_model_{ts}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"         ↑ best checkpoint saved  (val MAPE {vl_mape:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs.")
                break

    # ── Report final artifact locations ──────────────────────────────────────
    print("\nTraining complete.")
    best_models = sorted(save_dir.glob("best_model_*.pt"))
    if best_models:
        print(f"Best weights : {best_models[-1]}")
    print(f"Norm stats   : {save_dir / 'norm_stats.pt'}")
    print(f"Tight crop   : {crop_path}")
    print("\nTo evaluate, copy best_model_*.pt, norm_stats.pt, and tight_crop.json")
    print("into your evaluation/<MODEL_NAME>/ folder alongside model.py, then run:")
    print("  python evaluate.py <MODEL_NAME>")


if __name__ == "__main__":
    main()