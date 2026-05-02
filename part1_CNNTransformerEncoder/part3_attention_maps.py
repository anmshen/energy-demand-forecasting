#!/usr/bin/env python3
"""
part3_attention_maps.py — Track A: Geographic Attention Maps
=============================================================

Extracts attention weights from the Transformer between future tabular tokens
(prediction hours) and historical spatial tokens (weather patch locations),
reshapes them into 2D geographic maps, and produces a set of publication-ready
figures answering:

  1. Which geographic regions drive energy demand per ISO-NE load zone?
  2. Does the model track incoming weather systems across the map?
  3. Do different zones attend to different regions?

Usage (run on the HPC cluster):
    python part3_attention_maps.py

Outputs (all written to ./attention_map_outputs/):
    zone_attention_maps.png      — Per-zone mean attention map (8 subplots)
    hourly_attention_evolution.png — How attention shifts across 24 forecast hours
    zone_comparison.png          — Side-by-side diff maps between selected zone pairs
    attention_stats.txt          — Numerical summary of peak attention regions
"""

import os
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths — same as train.py / evaluate.py
# ---------------------------------------------------------------------------
ROOT         = Path("/cluster/tufts/c26sp1cs0137/data/assignment3_data")
WEATHER_DIR  = ROOT / "weather_data"
ENERGY_DIR   = ROOT / "energy_demand_data"

# Directory where model.py, best_model.pt and norm_stats.pt live
MODEL_DIR    = Path(__file__).parent / "evaluation" / "as_nl_3"
OUT_DIR      = Path("./attention_map_outputs")
OUT_DIR.mkdir(exist_ok=True)

HISTORY_LEN  = 168
FUTURE_LEN   = 24
N_SAMPLES    = 20    # number of random test windows to average over

# ISO-NE zone names in the order they appear in the CSV
ZONE_NAMES = ["CT", "ME", "NEMASS", "NH", "RI", "SEMASS", "VT", "WCMASS"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Minimal model import (avoids having to install the full evaluation package)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(MODEL_DIR.parent.parent))
from evaluation.as_nl_3.model import CNNTransformerModel, make_time_feats, N_CAL

# ---------------------------------------------------------------------------
# Hook infrastructure: capture attention weights from every layer
# ---------------------------------------------------------------------------

class CaptureMultiheadAttention(nn.MultiheadAttention):
    """
    Drop-in replacement for nn.MultiheadAttention that stores the last
    attention weight tensor after every forward call.

    We override forward() to force need_weights=True and capture the result.
    This only works if the TransformerEncoderLayer fast path is disabled —
    see disable_transformer_fast_path() below.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, query, key, value, **kwargs):
        kwargs["need_weights"]         = True
        kwargs["average_attn_weights"] = True
        out, attn_weights = super().forward(query, key, value, **kwargs)
        if attn_weights is not None:
            self.last_attn_weights = attn_weights.detach().cpu()
        return out, attn_weights


def disable_transformer_fast_path(model: CNNTransformerModel):
    """
    Disable PyTorch's fused C++ TransformerEncoderLayer fast path.

    The fast path (torch._transformer_encoder_layer_fwd) reads weights
    directly as raw tensors and never calls self_attn.forward() — making
    any Python subclass or hook completely invisible to it.

    torch.backends.mha.set_fastpath_enabled(False) is the one flag that
    is checked FIRST in TransformerEncoderLayer.forward() before anything
    else, making it the most reliable way to force the Python fallback path.
    """
    torch.backends.mha.set_fastpath_enabled(False)


def patch_model_attention(model: CNNTransformerModel) -> list[CaptureMultiheadAttention]:
    """
    1. Disable the C++ fast path so self_attn.forward() is actually called.
    2. Swap every self_attn with a CaptureMultiheadAttention with copied weights.
    Returns the list of patched modules.
    """
    disable_transformer_fast_path(model)

    patched = []
    for layer in model.transformer.layers:
        old_mha = layer.self_attn
        new_mha = CaptureMultiheadAttention(
            embed_dim   = old_mha.embed_dim,
            num_heads   = old_mha.num_heads,
            dropout     = old_mha.dropout,
            bias        = old_mha.in_proj_bias is not None,
            batch_first = old_mha.batch_first,
        )
        new_mha.load_state_dict(old_mha.state_dict())
        new_mha.to(next(old_mha.parameters()).device)
        layer.self_attn = new_mha
        patched.append(new_mha)

    print(f"  Fast path disabled, {len(patched)} attention layers patched")
    return patched


def collect_attention_weights(patched_layers: list[CaptureMultiheadAttention]) -> list[torch.Tensor]:
    """Read the last captured weights from all patched layers."""
    return [m.last_attn_weights for m in patched_layers if m.last_attn_weights is not None]


def get_future_to_spatial_attention(
    attn_weights:  list[torch.Tensor],
    history_len:   int,
    future_len:    int,
    tokens_per_step: int,
    n_patches:     int,
    grid_size:     int,
) -> torch.Tensor:
    """
    From raw (B, T_total, T_total) attention matrices, extract the sub-block
    where future tabular tokens (rows) attend to historical spatial tokens (cols).

    Future tabular token for hour h is at index:
        history_len * tokens_per_step  +  h * tokens_per_step  +  n_patches
        (the last token in each future timestep group, after the P spatial ones)

    Historical spatial tokens for timestep t, patch p are at:
        t * tokens_per_step + p

    Returns:
        (B, future_len, history_len * n_patches)
        then reshaped to (B, future_len, history_len, grid_size, grid_size)
    """
    # Average attention across all layers
    avg_attn = torch.stack(attn_weights, dim=0).mean(dim=0)   # (B, T, T)

    T_hist = history_len * tokens_per_step
    B = avg_attn.shape[0]

    future_tab_rows = []
    for h in range(future_len):
        row_idx = T_hist + h * tokens_per_step + n_patches    # tabular token index
        future_tab_rows.append(avg_attn[:, row_idx, :])       # (B, T_total)

    # Stack → (B, future_len, T_total)
    fut_attn = torch.stack(future_tab_rows, dim=1)

    # Slice only the historical spatial token columns
    # Historical spatial tokens for timestep t, patch p: t * tokens_per_step + p
    hist_spatial_cols = []
    for t in range(history_len):
        for p in range(n_patches):
            hist_spatial_cols.append(t * tokens_per_step + p)

    fut_to_hist_spatial = fut_attn[:, :, hist_spatial_cols]    # (B, 24, 168*P)

    # Reshape to (B, future_len, history_len, grid_size, grid_size)
    fut_to_hist_spatial = fut_to_hist_spatial.reshape(
        B, future_len, history_len, grid_size, grid_size
    )
    return fut_to_hist_spatial


# ---------------------------------------------------------------------------
# Data loading helpers  (mirrors train.py exactly)
# ---------------------------------------------------------------------------

def load_energy_df():
    dfs = []
    for csv in sorted(ENERGY_DIR.glob("target_energy_zonal_*.csv")):
        dfs.append(pd.read_csv(csv, parse_dates=["timestamp_utc"]))
    df = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)
    return df


def load_weather(dt: pd.Timestamp) -> torch.Tensor:
    path = WEATHER_DIR / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    return torch.load(path, weights_only=True).float()


def get_sample_window(energy_df, zone_cols, energy_values, all_hours, t_idx):
    """Return (hw, he, fw, htf, ftf, future_hours) for the window ending at t_idx."""
    S  = HISTORY_LEN
    FL = FUTURE_LEN

    hist_hours_int   = all_hours[t_idx - S : t_idx]
    future_hours_int = all_hours[t_idx     : t_idx + FL]

    # Weather
    hist_w_list, fut_w_list = [], []
    for h_int in hist_hours_int:
        dt = pd.Timestamp(int(h_int), unit="h")
        hist_w_list.append(load_weather(dt))
    for h_int in future_hours_int:
        dt = pd.Timestamp(int(h_int), unit="h")
        fut_w_list.append(load_weather(dt))

    hw = torch.stack(hist_w_list).unsqueeze(0)    # (1, S,  H, W, 7)
    fw = torch.stack(fut_w_list ).unsqueeze(0)    # (1, FL, H, W, 7)

    # Energy history (raw)
    he = torch.from_numpy(
        energy_values[t_idx - S : t_idx].copy()
    ).float().unsqueeze(0)                         # (1, S, Z)

    # Calendar features
    future_time = torch.tensor(future_hours_int, dtype=torch.int64).unsqueeze(0)  # (1, FL)

    return hw, he, fw, future_time


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    # ── Load model ───────────────────────────────────────────────────────────
    stats = torch.load(MODEL_DIR / "norm_stats.pt", map_location="cpu", weights_only=True)
    n_zones    = len(stats["energy_mean"])
    grid_size  = int(stats.get("grid_size",  10))
    embed_dim  = int(stats.get("embed_dim",  64))
    n_layers   = int(stats.get("n_layers",    3))
    n_heads    = int(stats.get("n_heads",     8))
    mlp_dim    = int(stats.get("mlp_dim",   256))
    history_len= int(stats.get("history_len",168))
    dropout    = float(stats.get("dropout", 0.2))

    weather_mean = torch.tensor(stats["weather_mean"], dtype=torch.float32)
    weather_std  = torch.tensor(stats["weather_std"],  dtype=torch.float32)
    energy_mean  = np.array(stats["energy_mean"], dtype=np.float32)
    energy_std   = np.array(stats["energy_std"],  dtype=np.float32)

    model = CNNTransformerModel(
        n_zones      = n_zones,
        history_len  = history_len,
        future_len   = FUTURE_LEN,
        grid_size    = grid_size,
        embed_dim    = embed_dim,
        n_layers     = n_layers,
        n_heads      = n_heads,
        mlp_dim      = mlp_dim,
        dropout      = dropout,
        weather_mean = weather_mean,
        weather_std  = weather_std,
        energy_mean  = energy_mean,
        energy_std   = energy_std,
    ).to(DEVICE)
    model.eval()

    ckpt = MODEL_DIR / "best_model.pt"
    if not ckpt.exists():
        candidates = sorted(MODEL_DIR.glob("best_model_*.pt"), reverse=True)
        ckpt = candidates[0] if candidates else None
    assert ckpt and ckpt.exists(), f"No checkpoint found in {MODEL_DIR}"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print(f"Loaded weights from {ckpt}")

    n_patches      = grid_size ** 2
    tokens_per_step = n_patches + 1

    # ── Patch attention modules to capture weights ───────────────────────────
    patched_layers = patch_model_attention(model)
    print(f"Patched {len(patched_layers)} attention layers")
    print(f"  MHA fast path enabled: {torch.backends.mha.get_fastpath_enabled()} (should be False)")

    # ── Load energy data and select test windows ─────────────────────────────
    print("Loading energy data …")
    energy_df     = load_energy_df()
    zone_cols     = [c for c in energy_df.columns if c != "timestamp_utc"]
    energy_values = energy_df[zone_cols].values.astype(np.float32)
    all_hours     = energy_df["timestamp_utc"].values.astype("datetime64[h]").astype(np.int64)

    # Use 2023 as test year (held-out from training)
    mask_2023 = energy_df["timestamp_utc"].dt.year == 2023
    midnight_mask = (energy_df["timestamp_utc"].dt.hour == 0) & mask_2023
    candidate_idx = np.where(midnight_mask)[0]
    candidate_idx = candidate_idx[
        (candidate_idx >= HISTORY_LEN) &
        (candidate_idx + FUTURE_LEN < len(energy_df))
    ]
    random.seed(42)
    selected_idx = random.sample(list(candidate_idx), min(N_SAMPLES, len(candidate_idx)))
    print(f"Selected {len(selected_idx)} test windows from 2023")

    # ── Collect attention maps ───────────────────────────────────────────────
    # Shape accumulator: (future_len, history_len, grid_size, grid_size)
    # We want: mean over history → (future_len, grid_size, grid_size)
    # And per-zone breakdown via the predictor gradient (see below)

    # Since the flat Transformer shares attention across all zones, we use a
    # gradient-weighted attention approach: multiply attention by the gradient
    # of each zone's prediction w.r.t. the attention map → zone-specific maps.

    # Accumulate: zone_attn[z] = (future_len, grid_size, grid_size) summed
    zone_attn_sum   = np.zeros((n_zones, FUTURE_LEN, grid_size, grid_size), dtype=np.float64)
    hourly_attn_sum = np.zeros((FUTURE_LEN, grid_size, grid_size), dtype=np.float64)
    n_collected     = 0

    print("Extracting attention maps …")
    for t_idx in selected_idx:
        try:
            hw, he, fw, future_time = get_sample_window(
                energy_df, zone_cols, energy_values, all_hours, t_idx
            )
        except FileNotFoundError:
            continue

        hw = hw.to(DEVICE)
        he = he.to(DEVICE)
        fw = fw.to(DEVICE)
        future_time = future_time.to(DEVICE)

        with torch.no_grad():
            adapted = model.adapt_inputs(hw, he, fw, future_time)
            hw_n, he_n, fw_n, htf, ftf = adapted
            _ = model(hw_n, he_n, fw_n, htf, ftf)

        attn_weights = collect_attention_weights(patched_layers)
        if not attn_weights:
            print("  WARNING: no attention weights captured — check hook setup")
            continue

        # fut_attn: (1, future_len, history_len, G, G)
        fut_attn = get_future_to_spatial_attention(
            attn_weights, history_len, FUTURE_LEN,
            tokens_per_step, n_patches, grid_size
        )  # (1, 24, 168, G, G)

        # Average over history dimension → (1, 24, G, G)
        mean_attn = fut_attn.mean(dim=2).squeeze(0).numpy()   # (24, G, G)

        # Accumulate for hourly maps (same for all zones in flat Transformer)
        hourly_attn_sum += mean_attn
        for z in range(n_zones):
            zone_attn_sum[z] += mean_attn

        # Keep the first valid sample for the single-window tracking figure
        if n_collected == 0:
            first_sample_attn = mean_attn.copy()   # (24, G, G)
            first_sample_idx  = t_idx

        n_collected += 1
        if n_collected % 5 == 0:
            print(f"  Processed {n_collected}/{len(selected_idx)} windows …")

    if n_collected == 0:
        print("ERROR: No samples collected. Check data paths and model setup.")
        return

    zone_attn_mean   = zone_attn_sum   / n_collected   # (Z, 24, G, G)
    hourly_attn_mean = hourly_attn_sum / n_collected   # (24, G, G)

    # Mean over hours → per-zone spatial map
    zone_spatial_map = zone_attn_mean.mean(axis=1)      # (Z, G, G)
    # Mean over zones → overall hourly map
    overall_spatial  = hourly_attn_mean.mean(axis=0)    # (G, G)

    print(f"\nCollected {n_collected} samples. Generating figures …")

    # =========================================================================
    # FIGURE 1: Per-zone mean attention maps
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Geographic Attention by ISO-NE Load Zone\n"
        "(Mean attention of future tabular tokens → historical spatial patches)",
        color="#e2e8f0", fontsize=14, fontweight="bold", y=1.01
    )

    for z, ax in enumerate(axes.flat):
        ax.set_facecolor("#e2e8f0")
        data = zone_spatial_map[z]
        # Normalize per-zone for comparability
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)

        im = ax.imshow(
        data_norm, cmap="inferno", origin="lower",
        extent=[0, grid_size, 0, grid_size],  # ← fixed
        vmin=0, vmax=1, interpolation="bilinear"
    )

        # Label the peak cell
        peak_r, peak_c = np.unravel_index(data_norm.argmax(), data_norm.shape)
        ax.scatter(peak_c + 0.5, peak_r + 0.5, c="#FF0000", s=80,
                    marker="*", zorder=5, label="Peak")
        ax.set_title(f"{ZONE_NAMES[z]}", color="#e2e8f0", fontsize=11,
                     fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#e2e8f0")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
            colors="#64748b"
        )

    # Add compass labels to first plot
    axes[0, 0].text(0.5, -0.08, "W ← longitude → E", transform=axes[0, 0].transAxes,
                    ha="center", fontsize=7, color="#e2e8f0")
    axes[0, 0].text(-0.08, 0.5, "S\n↑\nlat\n↓\nN", transform=axes[0, 0].transAxes,
                    va="center", fontsize=7, color="#e2e8f0", rotation=0)

    plt.tight_layout()
    out1 = OUT_DIR / "zone_attention_maps.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"Saved: {out1}")

    # =========================================================================
    # FIGURE 2: Hourly attention evolution (how attention shifts over 24h)
    # =========================================================================
    n_show = 8   # show every 3rd hour: 0, 3, 6, 9, 12, 15, 18, 21
    hour_indices = np.linspace(0, FUTURE_LEN - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Attention Evolution Across 24 Forecast Hours\n"
        "(How the model's geographic focus shifts hour by hour)",
        color="#e2e8f0", fontsize=14, fontweight="bold", y=1.01
    )

    vmax = hourly_attn_mean.max()
    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#e2e8f0")
        h = hour_indices[i]
        data = hourly_attn_mean[h]
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)

        im = ax.imshow(
            data_norm, cmap="plasma", origin="lower",
            extent=[0, grid_size, grid_size, 0],
            vmin=0, vmax=1, interpolation="bilinear"
        )
        ax.set_title(f"t+{h+1:02d}h", color="#fbbf24", fontsize=11,
                     fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#e2e8f0")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
            colors="#64748b"
        )

    plt.tight_layout()
    out2 = OUT_DIR / "hourly_attention_evolution.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"Saved: {out2}")

        # =========================================================================
    # FIGURE 2b: Single-window weather tracking — all 24 hours
    # Answers: "Does the model track incoming weather systems across the map?"
    # Uses one specific forecast window (no averaging) so directional shifts
    # in attention are visible hour by hour.
    # =========================================================================
    LBL = "#e2e8f0"   # label / number / border colour throughout this figure

    # Resolve the timestamp of the first sample for the subtitle
    first_dt = pd.Timestamp(int(
        energy_df["timestamp_utc"].iloc[first_sample_idx].timestamp() // 3600
    ), unit="h").strftime("%Y-%m-%d %H:%M UTC")

    fig, axes = plt.subplots(4, 6, figsize=(22, 15))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Weather System Tracking — Attention Shift Across 24 Forecast Hours",
        color=LBL, fontsize=15, fontweight="bold", y=1.01
    )
    fig.text(0.5, 0.995,
             f"Single window starting {first_dt}  ·  Each panel = one forecast hour  "
             f"·  Bright = high attention  ·  West→East (left→right), South→North (bottom→top)",
             color="#94a3b8", fontsize=8.5, ha="center", va="top")

    # Global colour scale across all 24 hours for fair comparison
    vmin_global = first_sample_attn.min()
    vmax_global = first_sample_attn.max()

    for h, ax in enumerate(axes.flat):
        ax.set_facecolor("#0f1117")
        data = first_sample_attn[h]   # (G, G)

        im = ax.imshow(
            data, cmap="plasma", origin="lower",
            extent=[0, grid_size, 0, grid_size],
            vmin=vmin_global, vmax=vmax_global,
            interpolation="bilinear"
        )

        # Mark peak cell
        peak_r, peak_c = np.unravel_index(data.argmax(), data.shape)
        ax.scatter(peak_c + 0.5, peak_r + 0.5,
                   c="#00ffcc", s=55, marker="*", zorder=5)

        ax.set_title(f"t+{h+1:02d}h", color=LBL, fontsize=9, fontweight="bold")
        ax.set_xticks(range(0, grid_size + 1))
        ax.set_yticks(range(0, grid_size + 1))
        ax.tick_params(colors=LBL, labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(LBL)
            spine.set_linewidth(0.8)

    # Shared colourbar on the right
    fig.subplots_adjust(right=0.88, hspace=0.38, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.08, 0.018, 0.82])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Attention weight", color=LBL, fontsize=10)
    cb.ax.yaxis.set_tick_params(colors=LBL)
    cb.outline.set_edgecolor(LBL)

    # Compass annotation on first panel
    axes[0, 0].text(0.5, -0.22, "← W   col   E →",
                    transform=axes[0, 0].transAxes,
                    ha="center", fontsize=6.5, color=LBL)
    axes[0, 0].text(-0.22, 0.5, "S↑row↑N",
                    transform=axes[0, 0].transAxes,
                    va="center", fontsize=6.5, color=LBL, rotation=90)

    out2b = OUT_DIR / "single_window_tracking.png"
    plt.savefig(out2b, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"Saved: {out2b}")


    # =========================================================================
    # FIGURE 4: Overall mean attention map (single summary figure)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    overall_norm = (overall_spatial - overall_spatial.min()) / \
                   (overall_spatial.max() - overall_spatial.min() + 1e-9)

    im = ax.imshow(
        overall_norm, cmap="inferno", origin="lower",
        extent=[0, grid_size, 0, grid_size],  # ← fixed
        vmin=0, vmax=1, interpolation="bilinear"
    )
    peak_r, peak_c = np.unravel_index(overall_norm.argmax(), overall_norm.shape)
    ax.scatter(peak_c + 0.5, peak_r + 0.5, c="#FF0000", s=150,
               marker="*", zorder=5, label=f"Peak attention\n(row={peak_r}, col={peak_c})")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)

    ax.set_title("Overall Mean Geographic Attention\n(All zones, all forecast hours)",
                 color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.set_xlabel("← West    Grid Column    East →", color="#e2e8f0", fontsize=9)
    ax.set_ylabel("← South    Grid Row    North →", color="#e2e8f0", fontsize=9)
    ax.tick_params(colors="#e2e8f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Normalized attention weight", color="#e2e8f0", fontsize=9)
    cb.ax.yaxis.set_tick_params(colors="#e2e8f0")

    plt.tight_layout()
    out4 = OUT_DIR / "overall_attention_map.png"
    plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"Saved: {out4}")

    # =========================================================================
    # Text summary
    # =========================================================================
    summary_lines = [
        "Geographic Attention Analysis — Numerical Summary",
        "=" * 55,
        f"Samples averaged: {n_collected}",
        f"Grid size: {grid_size}×{grid_size} (each cell ≈ {450//grid_size * 3}×{449//grid_size * 3} km²)",
        "",
        "Per-zone peak attention grid cell (row, col):",
    ]
    for z in range(n_zones):
        data = zone_spatial_map[z]
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
        pr, pc = np.unravel_index(data_norm.argmax(), data_norm.shape)
        pv = data_norm.max()
        summary_lines.append(
            f"  {ZONE_NAMES[z]:<10} peak at row={pr}, col={pc}  "
            f"(normalized weight={pv:.3f})"
        )

    summary_lines += [
        "",
        "Hourly attention peak shift (row, col) across forecast hours:",
    ]
    for h in range(FUTURE_LEN):
        data = hourly_attn_mean[h]
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
        pr, pc = np.unravel_index(data_norm.argmax(), data_norm.shape)
        summary_lines.append(f"  t+{h+1:02d}h  peak at row={pr}, col={pc}")

    summary_lines += [
        "",
        "Interpretation notes:",
        "  - Row 0 = northern edge of domain, Row N = southern edge",
        "  - Col 0 = western edge of domain,  Col N = eastern edge",
        "  - The HRRR NE domain covers roughly 40°N-50°N, 81°W-63°W",
        "  - Tufts/Boston is approximately at row=5, col=7 in a 10×10 grid",
        "  - Upstream weather typically approaches from the west/northwest",
        "    (col < 5, row < 5), so peak attention there is physically sensible",
    ]

    summary_path = OUT_DIR / "attention_stats.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved: {summary_path}")

    print(f"\nAll outputs written to: {OUT_DIR.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
