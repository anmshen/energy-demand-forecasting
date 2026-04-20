#!/usr/bin/env python3
"""
train.py — Part 1: CNN-Transformer Patch Architecture
=====================================================

Usage:
    python train.py [--epochs 50] [--batch_size 4] [--lr 5e-4] [--grid_size 10]
                    [--n_train_days 300] [--save_dir ./checkpoints]

What this script does:
  1. Loads weather (.pt) and energy (.csv) data from the cluster paths.
  2. Computes normalization statistics (mean/std) from the training split.
  3. Trains CNNTransformerModel: CNN downsamples weather to spatial patches,
     concatenates with tabular tokens, feeds to Transformer, predicts 24-hour ahead.
  4. Saves the best checkpoint (lowest val MAPE) to <save_dir>/best_model.pt.
  5. Saves normalization stats to <save_dir>/norm_stats.pt for evaluation.
"""

import argparse
import math
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.part2_best_model.model import CNNEncoderDecoderModel

# ============================================================
# Paths  (same as evaluate.py)
# ============================================================
ROOT        = Path('/cluster/tufts/c26sp1cs0137/data/assignment3_data')
WEATHER_DIR = ROOT / "weather_data"
ENERGY_DIR  = ROOT / "energy_demand_data"

HISTORY_LEN = 168
FUTURE_LEN  = 24
TRAIN_YEARS = [2019, 2020, 2021, 2022]  # Use recent years closer to test distribution (2023)
VAL_YEAR    = 2022   # last portion of 2022 used for validation

#!/usr/bin/env python3
"""
train.py — Part 1: CNN-Transformer Patch Architecture
=====================================================

Usage:
    python train.py [--epochs 50] [--batch_size 4] [--lr 5e-4] [--grid_size 10]
                    [--n_train_days 300] [--save_dir ./checkpoints]

What this script does:
  1. Loads weather (.pt) and energy (.csv) data from the cluster paths.
  2. Computes normalization statistics (mean/std) from the training split.
  3. Trains CNNTransformerModel: CNN downsamples weather to spatial patches,
     concatenates with tabular tokens, feeds to Transformer, predicts 24-hour ahead.
  4. Saves the best checkpoint (lowest val MAPE) to <save_dir>/best_model.pt.
  5. Saves normalization stats to <save_dir>/norm_stats.pt for evaluation.
"""

# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=4)  # Smaller batch for CNN memory
    p.add_argument("--lr",           type=float, default=1e-4)  # Reduced initial learning rate
    p.add_argument("--grid_size",    type=int,   default=10)  # Spatial patch grid
    p.add_argument("--embed_dim",    type=int,   default=96)  # Reduced from 128
    p.add_argument("--n_transformer_layers", type=int, default=3)  # Reduced from 4
    p.add_argument("--n_heads",      type=int,   default=8)
    p.add_argument("--mlp_dim",      type=int,   default=384)  # Reduced from 512
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--n_train_days", type=int,   default=300,
                   help="Number of training midnight windows to use")
    p.add_argument("--n_val_days",   type=int,   default=30)
    p.add_argument("--save_dir",     type=str,   default="./checkpoints")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()

# ============================================================
# Data helpers
# ============================================================

_weather_cache: dict = {}

def load_weather(hour_int64: int) -> torch.Tensor:
    """Return (450, 449, 7) float32 tensor for the given hours-since-epoch."""
    if hour_int64 in _weather_cache:
        return _weather_cache[hour_int64]
    dt = pd.Timestamp(int(hour_int64), unit="h")
    path = WEATHER_DIR / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    t = torch.load(path, weights_only=True).float()
    _weather_cache[hour_int64] = t
    if len(_weather_cache) > 300:
        oldest = next(iter(_weather_cache))
        del _weather_cache[oldest]
    return t


def load_energy_df():
    dfs = []
    for csv_path in sorted(ENERGY_DIR.glob("target_energy_zonal_*.csv")):
        dfs.append(pd.read_csv(csv_path, parse_dates=["timestamp_utc"]))
    df = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)
    deltas = df["timestamp_utc"].diff().dropna()
    assert (deltas == pd.Timedelta("1h")).all(), "Energy data has gaps!"
    return df


def get_midnight_indices(energy_df, years):
    """Return row indices where timestamp is midnight in the given years."""
    mask = (
        energy_df["timestamp_utc"].dt.year.isin(years) &
        (energy_df["timestamp_utc"].dt.hour == 0)
    )
    idxs = np.where(mask)[0]
    # Only keep rows where full history + future window is in bounds
    idxs = idxs[
        (idxs >= HISTORY_LEN) &
        (idxs + FUTURE_LEN <= len(energy_df))
    ]
    return idxs


# ============================================================
# Normalization statistics
# ============================================================

def compute_weather_stats(energy_df, sample_indices, n_samples=50):
    """
    Estimate channel-wise mean/std from a random sample of weather files.
    Returns two (7,) tensors.
    """
    print("Computing weather normalization statistics …")
    chosen = np.random.choice(sample_indices, size=min(n_samples, len(sample_indices)),
                              replace=False)
    channel_sum  = torch.zeros(7)
    channel_sum2 = torch.zeros(7)
    count = 0
    for t_idx in chosen:
        h_int = int(energy_df["timestamp_utc"].iloc[t_idx].timestamp() // 3600)
        try:
            w = load_weather(h_int)          # (450, 449, 7)
        except FileNotFoundError:
            continue
        w_flat = w.reshape(-1, 7)            # (N, 7)
        channel_sum  += w_flat.sum(0)
        channel_sum2 += (w_flat ** 2).sum(0)
        count        += w_flat.shape[0]
    mean = channel_sum / count
    std  = (channel_sum2 / count - mean ** 2).clamp(min=0).sqrt()
    std  = std.clamp(min=1e-6)
    print(f"  Weather mean: {mean.tolist()}")
    print(f"  Weather std : {std.tolist()}")
    return mean, std


def compute_energy_stats(energy_values):
    """
    Compute mean/std per energy zone.
    energy_values: (T, Z) numpy float32
    Returns: (Z,) and (Z,) arrays for per-zone norm
    """
    mean = np.nanmean(energy_values, axis=0, keepdims=True).astype(np.float32)  # (1, Z)
    std  = np.nanstd(energy_values, axis=0, keepdims=True).astype(np.float32)   # (1, Z)
    std  = np.maximum(std, 1e-6)
    print(f"  Energy mean per zone: {mean.flatten()}")
    print(f"  Energy std per zone : {std.flatten()}")
    return mean.flatten(), std.flatten()  # (Z,) each


# ============================================================
# Dataset
# ============================================================

class EnergyDataset(Dataset):
    def __init__(self, energy_df, energy_values, all_hours,
                 indices, weather_mean, weather_std,
                 energy_mean, energy_std):
        self.energy_df     = energy_df
        self.energy_values = energy_values   # (T, Z) float32
        self.all_hours     = all_hours       # (T,)  int64
        self.indices       = indices
        self.w_mean        = weather_mean    # (7,) tensor
        self.w_std         = weather_std     # (7,) tensor
        self.e_mean        = energy_mean
        self.e_std         = energy_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t_idx = self.indices[i]

        hist_slice   = slice(t_idx - HISTORY_LEN, t_idx)
        future_slice = slice(t_idx, t_idx + FUTURE_LEN)

        hist_hours   = self.all_hours[hist_slice]    # (168,) int64
        future_hours = self.all_hours[future_slice]  # (24,)  int64

        # --- Energy history (normalized) ---
        hist_energy = self.energy_values[hist_slice]  # (168, Z)
        hist_energy = (hist_energy - self.e_mean) / self.e_std

        # --- History weather: KEEP FULL SPATIAL DIMENSIONS for CNN ---
        try:
            hist_weather = torch.stack([load_weather(int(h)) for h in hist_hours])  # (168, 450, 449, 7)
        except FileNotFoundError:
            return None
        # Normalize
        hist_weather = (hist_weather - self.w_mean) / (self.w_std + 1e-6)

        # --- Future weather: KEEP FULL SPATIAL DIMENSIONS for CNN ---
        try:
            fut_weather = torch.stack([load_weather(int(h)) for h in future_hours])  # (24, 450, 449, 7)
        except FileNotFoundError:
            return None
        # Normalize
        fut_weather = (fut_weather - self.w_mean) / (self.w_std + 1e-6)

        # --- Cyclic time features for historical and future ---
        def make_time_feats(hours_tensor):
            h = hours_tensor.float()
            hour_of_day = (h % 24.0) / 24.0 * 2 * math.pi
            day_of_week = ((h // 24.0) % 7.0) / 7.0 * 2 * math.pi
            feats = torch.stack([
                torch.sin(hour_of_day),
                torch.cos(hour_of_day),
                torch.sin(day_of_week),
                torch.cos(day_of_week),
            ], dim=-1)
            return feats
        
        hist_time_feats = make_time_feats(torch.tensor(hist_hours, dtype=torch.int64))  # (168, 4)
        fut_time_feats = make_time_feats(torch.tensor(future_hours, dtype=torch.int64))  # (24, 4)

        # --- Target (raw, un-normalized — loss computed on real units) ---
        target = torch.from_numpy(self.energy_values[future_slice].copy())  # (24, Z)

        return (
            hist_weather.float(),                                  # (168, 450, 449, 7)
            torch.from_numpy(hist_energy.copy()).float(),         # (168, Z)
            fut_weather.float(),                                   # (24, 450, 449, 7)
            hist_time_feats.float(),                               # (168, 4)
            fut_time_feats.float(),                                # (24, 4)
            target.float(),                                        # (24, Z)
        )


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


# Note: CNNTransformerModel is imported from evaluation.our_model.model


# ============================================================
# Loss functions
# ============================================================

def mse_loss(pred, target):
    """Mean Squared Error loss (preferred for training stability over MAPE)."""
    return ((pred - target) ** 2).mean()


def mape_metric(pred, target, epsilon=1.0):
    """MAPE for logging only — not used for gradient updates."""
    return ((pred - target).abs() / (target.abs() + epsilon)).mean() * 100
    return ((pred - target).abs() / (target.abs() + epsilon)).mean() * 100


# ============================================================
# Training loop
# ============================================================

def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_mape = 0.0
    n_batches  = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if batch is None:
                continue
            hist_w, hist_e, fut_w, hist_time_feats, fut_time_feats, target = [x.to(device) for x in batch]

            pred = model(hist_w, hist_e, fut_w, hist_time_feats, fut_time_feats)  # (B, 24, Z)
            loss = mse_loss(pred, target)
            mape = mape_metric(pred, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            total_mape += mape.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1), total_mape / max(n_batches, 1)


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

    # --- Load energy data ---
    print("Loading energy data …")
    energy_df     = load_energy_df()
    zone_cols     = [c for c in energy_df.columns if c != "timestamp_utc"]
    n_zones       = len(zone_cols)
    energy_values = energy_df[zone_cols].values.astype(np.float32)
    all_hours     = energy_df["timestamp_utc"].values.astype("datetime64[h]").astype(np.int64)
    print(f"  Zones: {n_zones}   Rows: {len(energy_df)}")

    # --- Get train / val indices ---
    all_train_idx = get_midnight_indices(energy_df, TRAIN_YEARS)
    # Use last n_val_days of training range as validation
    val_idx   = all_train_idx[-args.n_val_days:]
    train_idx = all_train_idx[:-args.n_val_days]

    # Optionally subsample training days
    if args.n_train_days < len(train_idx):
        chosen = np.random.choice(len(train_idx), args.n_train_days, replace=False)
        train_idx = train_idx[np.sort(chosen)]

    print(f"  Train days: {len(train_idx)}   Val days: {len(val_idx)}")

    # --- Compute normalization statistics from training data ---
    print("\nComputing normalization statistics …")
    w_mean, w_std = compute_weather_stats(energy_df, train_idx, n_samples=50)
    print("  Computing energy statistics …")
    # Use only the rows covered by training windows
    train_energy_rows = np.concatenate([
        np.arange(t - HISTORY_LEN, t + FUTURE_LEN) for t in train_idx
    ])
    train_energy_rows = np.unique(np.clip(train_energy_rows, 0, len(energy_values) - 1))
    e_mean, e_std = compute_energy_stats(energy_values[train_energy_rows])

    # Save norm stats AND architecture hyperparameters so get_model() can
    # reconstruct the model exactly without hardcoded values.
    stats = {
        "weather_mean":      w_mean.tolist(),
        "weather_std":       w_std.tolist(),
        "energy_mean":       e_mean.tolist(),
        "energy_std":        e_std.tolist(),
        # Architecture hyperparameters
        "history_len":       HISTORY_LEN,
        "grid_size":         args.grid_size,
        "embed_dim":         args.embed_dim,
        "n_encoder_layers":  args.n_transformer_layers,
        "n_decoder_layers":  args.n_transformer_layers,
        "n_heads":           args.n_heads,
        "mlp_dim":           args.mlp_dim,
        "dropout":           args.dropout,
    }
    torch.save(stats, save_dir / "norm_stats.pt")
    print(f"  Norm stats saved to {save_dir / 'norm_stats.pt'}")

    # --- Datasets & loaders ---
    train_ds = EnergyDataset(energy_df, energy_values, all_hours,
                             train_idx, w_mean, w_std, e_mean, e_std)
    val_ds   = EnergyDataset(energy_df, energy_values, all_hours,
                             val_idx,   w_mean, w_std, e_mean, e_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=1,
                              collate_fn=collate_skip_none, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=1,
                              collate_fn=collate_skip_none, pin_memory=True)

    # --- Model: Encoder-Decoder CNN-Transformer (Part 2) ---
    model = CNNEncoderDecoderModel(
        n_zones              = n_zones,
        history_len          = HISTORY_LEN,
        future_len           = FUTURE_LEN,
        grid_size            = args.grid_size,
        embed_dim            = args.embed_dim,
        n_encoder_layers     = args.n_transformer_layers,
        n_decoder_layers     = args.n_transformer_layers,
        n_heads              = args.n_heads,
        mlp_dim              = args.mlp_dim,
        dropout              = args.dropout,
        weather_mean         = w_mean,
        weather_std          = w_std,
        energy_mean          = e_mean,
        energy_std           = e_std,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # Adam with L2 regularization (weight decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Cosine annealing — decays smoothly from lr to eta_min over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    # --- Training with Early Stopping ---
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    print(f"\nTraining for up to {args.epochs} epochs (early stopping patience={patience})…\n")
    print(f"{'Epoch':>6}  {'Train MSE':>11}  {'Train MAPE':>11}  {'Val MSE':>9}  {'Val MAPE':>9}  {'LR':>10}  {'Patience':>8}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mape = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss,   val_mape   = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6}  {train_loss:>11.4f}  {train_mape:>10.2f}%  {val_loss:>9.4f}  {val_mape:>8.2f}%  {lr_now:>10.2e}  {patience_counter:>8}")

        # Early stopping on val MSE (training objective), log MAPE for reference
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), save_dir / f"best_model_{timestamp}.pt")
            print(f"         ↑ best model saved  (val MSE {best_val_loss:.4f} | val MAPE {val_mape:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs")
                break

    print(f"\nTraining complete.  Best val MSE: {best_val_loss:.4f}")
    
    # Find the best timestamped model (keeping it unique, not overwritten)
    best_models = list(save_dir.glob("best_model_*.pt"))
    if best_models:
        best_models.sort()  # Get the latest one
        latest_best = best_models[-1]
        print(f"Weights saved to: {latest_best}")
    else:
        print(f"WARNING: No best model found in {save_dir}")
    
    print(f"Norm stats saved to: {save_dir / 'norm_stats.pt'}")
    print("\nTo evaluate, copy the best_model_*.pt and norm_stats.pt into your")
    print("evaluation/<MODEL_NAME>/ folder alongside model.py, then run:")
    print("  python evaluate.py <MODEL_NAME>")


if __name__ == "__main__":
    main()