"""
Part 1 (as_nl_3): Baseline CNN-Transformer Patch Architecture
=============================================================

Clean reference implementation satisfying the Part 1 spec.

Architecture:
  1. CNN: Downsample weather maps (450×449×7) → (grid_size²) spatial tokens
  2. Historical Tabular Tokens: [demand_norm ‖ calendar] → MLP → 1 token/hour
  3. Future Tabular Tokens:     [zeros        ‖ calendar] → MLP → 1 token/hour
  4. Sequence assembly:
       - Add learnable spatial positional embeddings to spatial tokens
       - Group [P spatial ‖ 1 tabular] per timestep
       - Add learnable per-timestep temporal positional encodings
       - Concatenate all (S+24) groups → sequence of length (S+24)×(P+1)
  5. Transformer Encoder
  6. Slice 24 future timesteps → flatten → MLP → predictions → denormalize

Key design:
  All architecture hyperparameters are stored in norm_stats.pt at training time.
  get_model() reads them back so the reconstructed architecture exactly matches
  the saved weights — preventing the size-mismatch errors seen in earlier models.

Files expected alongside model.py:
    norm_stats.pt         — normalization stats + architecture hyperparameters
    best_model.pt         — trained weights
      (or latest best_model_<timestamp>.pt if best_model.pt is absent)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Calendar features  (4-dim)
# ---------------------------------------------------------------------------

N_CAL = 4  # sin/cos hour-of-day + sin/cos day-of-year


def make_time_feats(hours: torch.Tensor) -> torch.Tensor:
    """
    4-dim cyclic calendar features from absolute hours-since-Unix-epoch.
    Returns (..., 4) float32.
    """
    h = hours.float()
    hod = (h % 24.0) / 24.0 * 2.0 * math.pi
    doy = ((h // 24.0) % 365.0) / 365.0 * 2.0 * math.pi
    return torch.stack(
        [torch.sin(hod), torch.cos(hod), torch.sin(doy), torch.cos(doy)],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Spatial token CNN
# ---------------------------------------------------------------------------

class SpatialCNN(nn.Module):
    """
    Multi-scale CNN: (B, T, H, W, 7) → (B, T, grid_size², embed_dim).

    Three strided Conv2d layers reduce (450, 449) to roughly (12, 12),
    then AdaptiveAvgPool2d snaps the output to exactly grid_size × grid_size.
    """

    def __init__(self, grid_size: int = 10, embed_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=7, stride=4, padding=3),   # → H/4
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=4, padding=2),  # → H/16
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),  # → H/32
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.proj = nn.Linear(96, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, 7)  →  (B, T, P, D)  where P = grid_size²"""
        B, T, H, W, C = x.shape
        # (B, T, H, W, 7) → (B*T, 7, H, W)
        x = x.permute(0, 1, 4, 2, 3).contiguous().reshape(B * T, C, H, W)
        x = self.pool(self.net(x))                                   # (B*T, 96, G, G)
        G = self.grid_size
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B * T, G * G, 96)  # (B*T, P, 96)
        x = self.drop(self.proj(x))                                  # (B*T, P, D)
        return x.reshape(B, T, G * G, self.embed_dim)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CNNTransformerModel(nn.Module):
    """
    Baseline CNN-Transformer for energy demand forecasting (Part 1).

    Sequence: (S + 24) × (P + 1) tokens
        P = grid_size²   spatial patch tokens per timestep
        1                tabular token per timestep
    """

    def __init__(
        self,
        n_zones:      int,
        history_len:  int   = 96,
        future_len:   int   = 24,
        grid_size:    int   = 10,
        embed_dim:    int   = 64,
        n_layers:     int   = 2,
        n_heads:      int   = 4,
        mlp_dim:      int   = 256,
        dropout:      float = 0.1,
        weather_mean        = None,
        weather_std         = None,
        energy_mean         = None,
        energy_std          = None,
    ):
        super().__init__()
        self.n_zones      = n_zones
        self.history_len  = history_len
        self.future_len   = future_len
        self.grid_size    = grid_size
        self.embed_dim    = embed_dim
        P = grid_size ** 2
        self.n_patches       = P
        self.tokens_per_step = P + 1   # spatial + tabular

        # ── Normalization buffers ────────────────────────────────────────────
        em = np.zeros(n_zones, dtype=np.float32) if energy_mean is None \
             else np.asarray(energy_mean, dtype=np.float32)
        es = np.ones(n_zones, dtype=np.float32) if energy_std is None \
             else np.asarray(energy_std, dtype=np.float32)
        self.register_buffer("energy_mean", torch.from_numpy(em))
        self.register_buffer("energy_std",  torch.from_numpy(es))

        wm = torch.zeros(7) if weather_mean is None \
             else weather_mean.clone().detach().float()
        ws = torch.ones(7)  if weather_std  is None \
             else weather_std.clone().detach().float()
        self.register_buffer("weather_mean", wm)
        self.register_buffer("weather_std",  ws)

        # ── Encoders ─────────────────────────────────────────────────────────
        self.spatial_cnn = SpatialCNN(grid_size, embed_dim, dropout)

        self.hist_enc = nn.Sequential(
            nn.Linear(n_zones + N_CAL, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.fut_enc = nn.Sequential(
            nn.Linear(n_zones + N_CAL, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # ── Positional embeddings ────────────────────────────────────────────
        self.spatial_pos  = nn.Parameter(
            torch.randn(1, 1, P, embed_dim) * 0.02
        )
        self.temporal_pos = nn.Parameter(
            torch.randn(1, history_len + future_len, 1, embed_dim) * 0.02
        )

        # ── Transformer Encoder ──────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)

        # ── Prediction MLP ───────────────────────────────────────────────────
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * self.tokens_per_step, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, n_zones),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── adapt_inputs (required by evaluate.py) ───────────────────────────────

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,   # (B, 168, 450, 449, 7) raw
        history_energy:  torch.Tensor,   # (B, 168, n_zones)      raw
        future_weather:  torch.Tensor,   # (B, 24,  450, 449, 7)  raw
        future_time:     torch.Tensor,   # (B, 24)  int64, hours since epoch
    ) -> Tuple:
        """
        Normalize inputs and compute calendar features.

        Slices history to self.history_len so the evaluator's fixed 168-hour
        window is handled transparently.

        Returns:
            hw  : (B, S, 450, 449, 7)  normalized weather history
            he  : (B, S, n_zones)      normalized energy history
            fw  : (B, 24, 450, 449, 7) normalized future weather
            htf : (B, S, N_CAL)        historical calendar features
            ftf : (B, 24, N_CAL)       future calendar features
        """
        device = history_weather.device
        S = self.history_len

        # Slice to lookback window (contiguous to avoid CUDA alignment issues)
        hw = history_weather[:, -S:, ...].contiguous()
        he = history_energy[:,  -S:, :].contiguous()

        # Normalize weather
        wm = self.weather_mean.to(device)
        ws = self.weather_std.to(device)
        hw = (hw - wm) / (ws + 1e-6)
        fw = (future_weather - wm) / (ws + 1e-6)

        # Normalize energy
        em = self.energy_mean.to(device)
        es = self.energy_std.to(device)
        he = (he - em) / (es + 1e-6)

        # Calendar features — back-derive historical hours from future_time[:,0]
        first       = future_time[:, 0:1].float()                         # (B, 1)
        offsets     = torch.arange(S, 0, -1, device=device, dtype=torch.float32)
        hist_hours  = first - offsets.unsqueeze(0)                        # (B, S)
        htf = make_time_feats(hist_hours)             # (B, S, 4)
        ftf = make_time_feats(future_time.float())    # (B, 24, 4)

        return hw, he, fw, htf, ftf

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        hw:  torch.Tensor,   # (B, S, H, W, 7)   normalized weather history
        he:  torch.Tensor,   # (B, S, n_zones)    normalized energy history
        fw:  torch.Tensor,   # (B, 24, H, W, 7)  normalized future weather
        htf: torch.Tensor,   # (B, S, N_CAL)      historical calendar feats
        ftf: torch.Tensor,   # (B, 24, N_CAL)     future calendar feats
    ) -> torch.Tensor:       # (B, 24, n_zones)   raw MWh predictions
        B   = hw.shape[0]
        S   = self.history_len
        D   = self.embed_dim
        P   = self.n_patches
        TPS = self.tokens_per_step   # P + 1

        # 1. Spatial tokens via CNN
        hs = self.spatial_cnn(hw)   # (B, S,  P, D)
        fs = self.spatial_cnn(fw)   # (B, 24, P, D)

        # 2. Tabular tokens
        ht_tok = self.hist_enc(
            torch.cat([he, htf], dim=-1)
        ).unsqueeze(2)              # (B, S, 1, D)

        zero_y = torch.zeros(B, self.future_len, self.n_zones, device=hw.device)
        ft_tok = self.fut_enc(
            torch.cat([zero_y, ftf], dim=-1)
        ).unsqueeze(2)              # (B, 24, 1, D)

        # 3. Spatial positional embeddings  (shared across time and batch)
        hs = hs + self.spatial_pos   # (B, S,  P, D)
        fs = fs + self.spatial_pos   # (B, 24, P, D)

        # 4. Group tokens per timestep: [P spatial ‖ 1 tabular]
        #    and add temporal positional encoding
        T_total = S + self.future_len
        tp = self.temporal_pos.expand(B, T_total, TPS, D)   # (B, T, TPS, D)

        hist_tokens = torch.cat([hs, ht_tok], dim=2) + tp[:, :S]    # (B, S,  TPS, D)
        fut_tokens  = torch.cat([fs, ft_tok], dim=2) + tp[:, S:]    # (B, 24, TPS, D)

        # 5. Flatten to sequence and run Transformer
        seq = torch.cat([
            hist_tokens.reshape(B, S * TPS, D),
            fut_tokens.reshape(B, self.future_len * TPS, D),
        ], dim=1)                                   # (B, (S+24)*TPS, D)

        out = self.transformer(seq)                 # (B, (S+24)*TPS, D)

        # 6. Slice future timesteps, flatten tokens per step, predict
        fut_out = out[:, S * TPS:].reshape(B, self.future_len, TPS, D)
        fut_flat = fut_out.reshape(B, self.future_len, TPS * D)     # (B, 24, TPS*D)
        pred = self.predictor(fut_flat)             # (B, 24, n_zones)

        # 7. Denormalize
        em = self.energy_mean.to(pred.device)
        es = self.energy_std.to(pred.device)
        return pred * es + em


# ---------------------------------------------------------------------------
# Factory function (required by evaluate.py)
# ---------------------------------------------------------------------------

def get_model(metadata: dict) -> CNNTransformerModel:
    """
    Instantiate CNNTransformerModel with architecture hyperparameters and
    normalization stats loaded from norm_stats.pt, then load trained weights.

    All hyperparameters are saved into norm_stats.pt by train.py, so the
    reconstructed model always matches the saved checkpoint exactly.
    """
    model_dir = Path(__file__).parent

    # ── Load norm stats + architecture hyperparameters ───────────────────────
    stats_path = model_dir / "norm_stats.pt"
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        weather_mean = torch.tensor(stats["weather_mean"], dtype=torch.float32)
        weather_std  = torch.tensor(stats["weather_std"],  dtype=torch.float32)
        energy_mean  = np.array(stats["energy_mean"], dtype=np.float32)
        energy_std   = np.array(stats["energy_std"],  dtype=np.float32)
        history_len  = int(stats.get("history_len", 96))
        embed_dim    = int(stats.get("embed_dim",    64))
        n_layers     = int(stats.get("n_layers",      2))
        n_heads      = int(stats.get("n_heads",       4))
        mlp_dim      = int(stats.get("mlp_dim",     256))
        grid_size    = int(stats.get("grid_size",    10))
        dropout      = float(stats.get("dropout",   0.1))
        print(
            f"  Loaded norm_stats.pt: history_len={history_len}, embed_dim={embed_dim}, "
            f"n_layers={n_layers}, n_heads={n_heads}, grid_size={grid_size}"
        )
    else:
        print("  WARNING: norm_stats.pt not found — using default hyperparameters")
        weather_mean = weather_std = energy_mean = energy_std = None
        history_len = 96
        embed_dim = 64
        n_layers  = 2
        n_heads   = 4
        mlp_dim   = 256
        grid_size = 10
        dropout   = 0.1

    # ── Build model ──────────────────────────────────────────────────────────
    model = CNNTransformerModel(
        n_zones      = metadata["n_zones"],
        history_len  = history_len,
        future_len   = metadata.get("future_len", 24),
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
    )

    # ── Load trained weights ─────────────────────────────────────────────────
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        candidates = sorted(model_dir.glob("best_model_*.pt"), reverse=True)
        ckpt_path = candidates[0] if candidates else None

    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        # Support both plain state_dict and wrapped {"model_state_dict": ...}
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"  Loaded weights from {ckpt_path}")
    else:
        print("  WARNING: No checkpoint found — model is randomly initialized!")

    return model
