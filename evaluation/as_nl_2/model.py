"""
Part 1: CNN-Transformer Patch Architecture
==========================================

Improvements over baseline:
  1. Spatial cropping: crop weather maps to tight bounding box before CNN,
     reducing irrelevant geography and CNN FLOPs (~2.5x reduction).
  2. Richer calendar features: 7-dim encoding (hour, day-of-week, day-of-year,
     is_weekend) vs. original 4-dim.  train.py uses the identical function so
     adapt_inputs() is always consistent.
  3. Configurable lookback: history_len saved in norm_stats.pt and loaded by
     get_model(), so the transformer sequence length can be tuned at train time.

Architecture:
  1. CNN: Crops weather maps then downsamples to grid (e.g. 10×10) → P spatial
     tokens per timestep.
  2. Spatial Tokens: For every hour (historical + future), flatten CNN output
     into P tokens.
  3. Historical Tabular Tokens: demand (Y_i) + 7-dim calendar (C_i) → MLP → 1
     token per hour.
  4. Future Tabular Tokens: zero-padded demand + 7-dim calendar → MLP → 1
     token per hour.
  5. Transformer:
       - Add learnable spatial positional embeddings (grid locations)
       - Add learnable per-timestep temporal positional encodings
       - Process full sequence of length (S + 24) × (P + 1)
  6. Prediction: Extract final 24 timesteps → MLP → multi-zone predictions

Normalization: z-score per-zone (saved in norm_stats.pt).

Files expected alongside model.py:
    norm_stats.pt   — weather/energy mean & std, history_len
    best_model.pt   — trained weights (or best_model_<timestamp>.pt)
    tight_crop.json — {y_min, y_max, x_min, x_max} bounding box (optional)
"""

import json
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Calendar features  (7-dim)
# Defined at module level so train.py can import and reuse the exact same fn,
# preventing train/eval skew (the bug Model 2 was designed to avoid).
# ---------------------------------------------------------------------------

N_TIME_FEATURES = 7


def make_time_feats(hours_tensor: torch.Tensor) -> torch.Tensor:
    """
    Build a 7-dimensional cyclic calendar feature vector from integer
    hours-since-Unix-epoch.

    Features:
        sin/cos hour-of-day   (period 24 h)
        sin/cos day-of-week   (period 7 days)
        sin/cos day-of-year   (period 365 days — captures seasonality)
        is_weekend            (binary: Sat=1, Sun=1, else 0)

    Args:
        hours_tensor: (...,) int or float tensor of absolute hours since epoch

    Returns:
        (..., 7) float32 tensor
    """
    h = hours_tensor.float()

    hour_of_day = (h % 24.0) / 24.0 * 2 * math.pi
    day_of_week = ((h // 24.0) % 7.0) / 7.0 * 2 * math.pi
    day_of_year = ((h // 24.0) % 365.0) / 365.0 * 2 * math.pi

    # Thursday = 0 in Unix epoch; Saturday=2, Sunday=3 → weekend when % 7 in {2,3}
    # Actually: epoch day 0 = Thursday.  Sat offset=2, Sun offset=3.
    dow_int = (h // 24.0) % 7.0
    is_weekend = ((dow_int == 2) | (dow_int == 3)).float()

    return torch.stack([
        torch.sin(hour_of_day),
        torch.cos(hour_of_day),
        torch.sin(day_of_week),
        torch.cos(day_of_week),
        torch.sin(day_of_year),
        torch.cos(day_of_year),
        is_weekend,
    ], dim=-1)


# ---------------------------------------------------------------------------
# CNN for spatial token extraction  (with tight crop)
# ---------------------------------------------------------------------------

class SpatialTokenCNN(nn.Module):
    """
    Crop weather maps to a tight bounding box, then downsample to a small grid.

    Input:  (B, T, H, W, 7)           e.g. (B, T, 450, 449, 7)
    Output: (B, T, grid_h*grid_w, embed_dim)

    The crop buffers (y_min, y_max, x_min, x_max) are registered as non-
    parameter buffers so they are saved in the state_dict and move with the
    model to different devices.  get_model() can overwrite them after loading
    weights if a tight_crop.json is present.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_grid_size: int = 10,
        embed_dim: int = 64,
        dropout: float = 0.2,
        crop_y: Tuple[int, int] = (0, 450),
        crop_x: Tuple[int, int] = (0, 449),
    ):
        super().__init__()
        self.out_grid_size = out_grid_size
        self.embed_dim = embed_dim

        # Crop bounding box (stored as buffers → saved/loaded with state_dict)
        self.register_buffer('y_min', torch.tensor(crop_y[0], dtype=torch.long))
        self.register_buffer('y_max', torch.tensor(crop_y[1], dtype=torch.long))
        self.register_buffer('x_min', torch.tensor(crop_x[0], dtype=torch.long))
        self.register_buffer('x_max', torch.tensor(crop_x[1], dtype=torch.long))

        # Multi-scale CNN; exact spatial size after crop varies, so we rely
        # on AdaptiveAvgPool2d to reach the target grid size.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((out_grid_size, out_grid_size))
        self.proj = nn.Linear(96, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, C)
        Returns:
            (B, T, P, embed_dim)  where P = out_grid_size²
        """
        B, T, H, W, C = x.shape

        # --- Spatial crop ---
        y0 = self.y_min.item()
        y1 = self.y_max.item()
        x0 = self.x_min.item()
        x1 = self.x_max.item()
        x = x[:, :, y0:y1, x0:x1, :]          # (B, T, crop_H, crop_W, C)
        _, _, cH, cW, _ = x.shape

        # Reshape for Conv2d: (B*T, C, cH, cW)
        x_cnn = x.reshape(B * T, C, cH, cW)

        x_cnn = self.conv1(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.conv3(x_cnn)
        x_cnn = self.adaptive_pool(x_cnn)      # (B*T, 96, grid_size, grid_size)

        B_T, C_out, gh, gw = x_cnn.shape
        # → (B*T, P, 96)
        x_tokens = x_cnn.permute(0, 2, 3, 1).reshape(B_T, gh * gw, C_out)
        x_tokens = self.proj(x_tokens)          # (B*T, P, embed_dim)
        x_tokens = x_tokens.reshape(B, T, self.out_grid_size ** 2, self.embed_dim)
        return x_tokens


# ---------------------------------------------------------------------------
# Tabular token encoders
# ---------------------------------------------------------------------------

class HistoricalTabularEncoder(nn.Module):
    """Encodes [demand ‖ 7-dim calendar] → 1 tabular token per timestep."""

    def __init__(
        self,
        n_zones: int,
        n_time_features: int = N_TIME_FEATURES,
        embed_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = n_zones + n_time_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, demand: torch.Tensor, time_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            demand:     (B, T, n_zones)
            time_feats: (B, T, N_TIME_FEATURES)
        Returns:
            (B, T, embed_dim)
        """
        B, T, _ = demand.shape
        x = torch.cat([demand, time_feats], dim=-1)
        x = self.mlp(x.reshape(B * T, -1))
        return x.reshape(B, T, -1)


class FutureTabularEncoder(nn.Module):
    """Encodes [zero-padded demand ‖ 7-dim calendar] → 1 token per timestep."""

    def __init__(
        self,
        n_zones: int,
        n_time_features: int = N_TIME_FEATURES,
        embed_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = n_zones + n_time_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, time_feats: torch.Tensor, n_zones: int) -> torch.Tensor:
        """
        Args:
            time_feats: (B, T, N_TIME_FEATURES)
            n_zones:    int
        Returns:
            (B, T, embed_dim)
        """
        B, T, _ = time_feats.shape
        zero_demand = torch.zeros(B, T, n_zones, device=time_feats.device)
        x = torch.cat([zero_demand, time_feats], dim=-1)
        x = self.mlp(x.reshape(B * T, -1))
        return x.reshape(B, T, -1)


# ---------------------------------------------------------------------------
# Main Transformer Model
# ---------------------------------------------------------------------------

class CNNTransformerModel(nn.Module):
    """
    CNN-Transformer Patch Architecture for energy demand forecasting.

    Sequence length: (history_len + 24) × (P + 1)
        P  = grid_size²   spatial patch tokens per timestep
        1  = tabular token per timestep
    """

    def __init__(
        self,
        n_zones: int,
        history_len: int = 96,
        future_len: int = 24,
        grid_size: int = 10,
        embed_dim: int = 96,
        n_transformer_layers: int = 3,
        n_heads: int = 8,
        mlp_dim: int = 384,
        dropout: float = 0.2,
        crop_y: Tuple[int, int] = (0, 450),
        crop_x: Tuple[int, int] = (0, 449),
        weather_mean=None,
        weather_std=None,
        energy_mean=None,
        energy_std=None,
    ):
        super().__init__()
        self.n_zones = n_zones
        self.history_len = history_len
        self.future_len = future_len
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.n_spatial_tokens = grid_size * grid_size
        self.tokens_per_step = self.n_spatial_tokens + 1

        # ── Energy normalization (per-zone) ───────────────────────────────────
        if energy_mean is None:
            energy_mean = np.zeros(n_zones, dtype=np.float32)
        if energy_std is None:
            energy_std = np.ones(n_zones, dtype=np.float32)

        if isinstance(energy_mean, np.ndarray):
            self.register_buffer('energy_mean', torch.from_numpy(energy_mean))
            self.register_buffer('energy_std',  torch.from_numpy(energy_std))
        else:
            self.register_buffer('energy_mean', torch.full((n_zones,), float(energy_mean)))
            self.register_buffer('energy_std',  torch.full((n_zones,), float(energy_std)))

        # ── Weather normalization ─────────────────────────────────────────────
        if weather_mean is None:
            weather_mean = torch.zeros(7)
        if weather_std is None:
            weather_std = torch.ones(7)
        self.register_buffer('weather_mean', weather_mean.float())
        self.register_buffer('weather_std',  weather_std.float())

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.spatial_cnn = SpatialTokenCNN(
            in_channels=7,
            out_grid_size=grid_size,
            embed_dim=embed_dim,
            dropout=dropout,
            crop_y=crop_y,
            crop_x=crop_x,
        )

        self.hist_tabular = HistoricalTabularEncoder(n_zones, N_TIME_FEATURES, embed_dim, dropout)
        self.fut_tabular  = FutureTabularEncoder(n_zones, N_TIME_FEATURES, embed_dim, dropout)

        # Learnable spatial positional embeddings (one per grid cell, shared across time)
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.n_spatial_tokens, embed_dim) * 0.02
        )

        # Learnable temporal positional encodings (one per timestep, shared across tokens)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, history_len + future_len, 1, embed_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # Prediction MLP: flatten (P+1) tokens per future step → n_zones
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * self.tokens_per_step, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, n_zones),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── adapt_inputs ──────────────────────────────────────────────────────────

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,   # (B, 168, 450, 449, 7)  raw
        history_energy:  torch.Tensor,   # (B, 168, n_zones)       raw
        future_weather:  torch.Tensor,   # (B, 24,  450, 449, 7)  raw
        future_time:     torch.Tensor,   # (B, 24)  int64, hours since epoch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize inputs and build calendar features.

        Uses the module-level make_time_feats() — the same function used in
        train.py — so there is no train/eval feature skew.

        Slices history to the model's lookback window (history_len ≤ 168) so
        the evaluator's fixed 168-h window is handled transparently.

        Returns:
            history_weather : (B, S, 450, 449, 7)  normalized
            history_energy  : (B, S, n_zones)       normalized
            future_weather  : (B, 24, 450, 449, 7) normalized
            hist_time_feats : (B, S, N_TIME_FEATURES)
            fut_time_feats  : (B, 24, N_TIME_FEATURES)
        """
        device = history_weather.device
        S = self.history_len

        # ── Normalize weather ─────────────────────────────────────────────────
        w_mean = self.weather_mean.to(device)
        w_std  = self.weather_std.to(device)
        history_weather = (history_weather - w_mean) / (w_std + 1e-6)
        future_weather  = (future_weather  - w_mean) / (w_std + 1e-6)

        # ── Slice to lookback window ──────────────────────────────────────────
        history_weather = history_weather[:, -S:, ...]
        history_energy  = history_energy[:,  -S:, :]

        # ── Normalize energy ──────────────────────────────────────────────────
        e_mean = self.energy_mean.to(device)
        e_std  = self.energy_std.to(device)
        history_energy = (history_energy - e_mean) / (e_std + 1e-6)

        # ── Calendar features ─────────────────────────────────────────────────
        # Back-derive historical hours from future_time[:,0]
        first_fut = future_time[:, 0:1].float()                        # (B, 1)
        hist_offset = torch.arange(S, 0, -1, device=device, dtype=torch.float32)  # (S,)
        hist_hours = first_fut - hist_offset.unsqueeze(0)              # (B, S)

        hist_time_feats = make_time_feats(hist_hours)                  # (B, S, 7)
        fut_time_feats  = make_time_feats(future_time.float())         # (B, 24, 7)

        return history_weather, history_energy, future_weather, hist_time_feats, fut_time_feats

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        history_weather: torch.Tensor,   # (B, S, 450, 449, 7)  normalized
        history_energy:  torch.Tensor,   # (B, S, n_zones)       normalized
        future_weather:  torch.Tensor,   # (B, 24, 450, 449, 7) normalized
        hist_time_feats: torch.Tensor,   # (B, S, N_TIME_FEATURES)
        fut_time_feats:  torch.Tensor,   # (B, 24, N_TIME_FEATURES)
    ) -> torch.Tensor:                   # (B, 24, n_zones)  raw MWh
        B = history_weather.shape[0]
        S = self.history_len

        # 1. Spatial tokens via CNN (crop + downsample)
        hist_spatial = self.spatial_cnn(history_weather)   # (B, S,  P, D)
        fut_spatial  = self.spatial_cnn(future_weather)    # (B, 24, P, D)

        # 2. Tabular tokens
        hist_tab = self.hist_tabular(history_energy, hist_time_feats)  # (B, S,  D)
        fut_tab  = self.fut_tabular(fut_time_feats, self.n_zones)      # (B, 24, D)

        # 3. Add spatial positional embeddings
        hist_spatial = hist_spatial + self.spatial_pos_embed   # broadcast over B, T
        fut_spatial  = fut_spatial  + self.spatial_pos_embed

        # 4. Build per-timestep token sequences  [spatial_patches ‖ tabular]
        #    Result: (B, (S+24)*(P+1), D)
        hist_steps = [
            torch.cat([hist_spatial[:, t, :, :], hist_tab[:, t:t+1, :]], dim=1)
            for t in range(S)
        ]
        fut_steps = [
            torch.cat([fut_spatial[:, t, :, :], fut_tab[:, t:t+1, :]], dim=1)
            for t in range(self.future_len)
        ]
        sequence = torch.cat(hist_steps + fut_steps, dim=1)  # (B, (S+24)*(P+1), D)

        # 5. Add temporal positional encoding
        T_total = S + self.future_len
        temp_emb = self.temporal_pos_embed.expand(B, T_total, self.tokens_per_step, -1)
        temp_emb = temp_emb.reshape(B, T_total * self.tokens_per_step, self.embed_dim)
        sequence = sequence + temp_emb

        # 6. Transformer
        out = self.transformer(sequence)                            # (B, seq_len, D)

        # 7. Slice future timesteps and flatten per-step tokens
        fut_start = S * self.tokens_per_step
        fut_out   = out[:, fut_start:, :]                          # (B, 24*(P+1), D)
        fut_out   = fut_out.reshape(B, self.future_len, self.tokens_per_step, self.embed_dim)
        fut_flat  = fut_out.reshape(B, self.future_len, -1)        # (B, 24, (P+1)*D)

        # 8. Predict and denormalize
        pred = self.predictor(fut_flat)                            # (B, 24, n_zones)

        e_mean = self.energy_mean.to(pred.device)
        e_std  = self.energy_std.to(pred.device)
        pred   = pred * e_std + e_mean

        return pred


# ---------------------------------------------------------------------------
# Factory function required by evaluate.py
# ---------------------------------------------------------------------------

def get_model(metadata: dict) -> CNNTransformerModel:
    """
    Instantiate CNNTransformerModel, then load weights and normalization stats.

    Looks for (all in the same directory as this model.py):
        norm_stats.pt    — weather/energy mean & std, history_len
        tight_crop.json  — {y_min, y_max, x_min, x_max}  (optional)
        best_model.pt    — trained weights
                           (falls back to latest best_model_<timestamp>.pt)
    """
    model_dir = Path(__file__).parent

    # ── Normalization stats ───────────────────────────────────────────────────
    stats_path = model_dir / "norm_stats.pt"
    if stats_path.exists():
        stats = torch.load(stats_path, weights_only=True)
        weather_mean = torch.tensor(stats["weather_mean"], dtype=torch.float32)
        weather_std  = torch.tensor(stats["weather_std"],  dtype=torch.float32)
        energy_mean  = np.array(stats["energy_mean"], dtype=np.float32)
        energy_std   = np.array(stats["energy_std"],  dtype=np.float32)
        history_len  = int(stats.get("history_len", 96))
        print(f"  Loaded norm stats from {stats_path}  (history_len={history_len})")
    else:
        print(f"  WARNING: {stats_path} not found — using default norm constants.")
        weather_mean = None
        weather_std  = None
        energy_mean  = None
        energy_std   = None
        history_len  = 96

    # ── Spatial crop ──────────────────────────────────────────────────────────
    crop_path = model_dir / "tight_crop.json"
    if crop_path.exists():
        with open(crop_path) as f:
            crop = json.load(f)
        crop_y = (int(crop["y_min"]), int(crop["y_max"]))
        crop_x = (int(crop["x_min"]), int(crop["x_max"]))
        print(f"  Loaded tight crop: y={crop_y}, x={crop_x}")
    else:
        print(f"  tight_crop.json not found — using full spatial extent.")
        crop_y = (0, 450)
        crop_x = (0, 449)

    # ── Build model ───────────────────────────────────────────────────────────
    model = CNNTransformerModel(
        n_zones              = metadata["n_zones"],
        history_len          = history_len,
        future_len           = metadata.get("future_len", 24),
        grid_size            = 10,
        embed_dim            = 96,
        n_transformer_layers = 3,
        n_heads              = 8,
        mlp_dim              = 384,
        dropout              = 0.2,
        crop_y               = crop_y,
        crop_x               = crop_x,
        weather_mean         = weather_mean,
        weather_std          = weather_std,
        energy_mean          = energy_mean,
        energy_std           = energy_std,
    )

    # ── Load weights ──────────────────────────────────────────────────────────
    weights_path = model_dir / "best_model.pt"
    if not weights_path.exists():
        candidates = sorted(model_dir.glob("best_model_*.pt"))
        if candidates:
            weights_path = candidates[-1]
            print(f"  best_model.pt not found; using {weights_path.name}")

    if weights_path.exists():
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"  Loaded weights from {weights_path}")
    else:
        print(f"  WARNING: No weights found in {model_dir} — model has random weights.")

    return model