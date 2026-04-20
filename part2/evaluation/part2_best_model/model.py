"""
Part 1: Baseline CNN-Transformer Patch Architecture (40 points)
=============================================================

Architecture:
  1. CNN: Downsamples weather maps (450, 449, 7) → smaller grid (e.g., 10×10) 
     to create P spatial tokens per timestep.
  
  2. Spatial Tokens: For every hour (historical + future), flatten CNN output 
     into P tokens.
  
  3. Historical Tabular Tokens: Concatenate demand (Y_i) + calendar features (C_i)
     → linear layer → 1 token per hour.
  
  4. Future Tabular Tokens: Calendar features only (C_i) + zero-padding for demand
     → linear layer → 1 token per hour.
  
  5. Transformer: 
     - Project all tokens to embedding dim D
     - Add learnable spatial positional embeddings (grid locations)
     - Add per-timestep positional encodings (temporal)
     - Process full sequence of length (S + 24) × (P + 1)
  
  6. Prediction: Extract final 24 timesteps from Transformer output
     → MLP → multi-zone predictions

Normalization: z-score per-zone (saved in norm_stats.pt).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# CNN for spatial token extraction
# ---------------------------------------------------------------------------

class SpatialTokenCNN(nn.Module):
    """
    CNN to downsample (B, T, H, W, 7) → (B, T, P) where P = grid_h × grid_w.
    
    Input:  (B, T, 450, 449, 7)
    Output: (B, T, grid_h × grid_w, embed_dim)
    """
    def __init__(self, in_channels: int = 7, out_grid_size: int = 10, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.out_grid_size = out_grid_size
        self.embed_dim = embed_dim
        
        # Multi-scale CNN to downsample (450, 449) → (10, 10)
        # Downsample by ~45x in each spatial dimension
        # Reduced channel sizes: 32→24, 64→48, 128→96
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )  # (H, W) / 4
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )  # (H, W) / 16
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )  # (H, W) / 48 ≈ (10, 10) for 450×449
        
        # Adaptive pooling to exact target grid size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((out_grid_size, out_grid_size))
        
        # Project to embedding dimension
        self.proj = nn.Linear(96, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, C) = (B, T, 450, 449, 7)
        Returns:
            (B, T, P, embed_dim) where P = grid_h × grid_w
        """
        B, T, H, W, C = x.shape
        
        # Reshape to (B*T, C, H, W) for CNN processing
        x_cnn = x.reshape(B * T, C, H, W)
        
        # Apply CNN
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.conv3(x_cnn)
        x_cnn = self.adaptive_pool(x_cnn)  # (B*T, 128, 10, 10)
        
        # Reshape to tokens: (B*T, 128, grid_h, grid_w) → (B*T, P, 128)
        B_T, C_out, grid_h, grid_w = x_cnn.shape
        x_tokens = x_cnn.permute(0, 2, 3, 1).reshape(B_T, grid_h * grid_w, C_out)
        
        # Project to embed_dim
        x_tokens = self.proj(x_tokens)  # (B*T, P, embed_dim)
        
        # Reshape back to (B, T, P, embed_dim)
        x_tokens = x_tokens.reshape(B, T, self.out_grid_size * self.out_grid_size, self.embed_dim)
        
        return x_tokens


# ---------------------------------------------------------------------------
# Tabular token encoders
# ---------------------------------------------------------------------------

class HistoricalTabularEncoder(nn.Module):
    """Encodes historical [demand + calendar] → tabular token."""
    
    def __init__(self, n_zones: int, n_time_features: int = 4, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        # Demand (n_zones) + time features (4)
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
            demand: (B, T, n_zones)
            time_feats: (B, T, 4)
        Returns:
            (B, T, embed_dim)
        """
        B, T, Z = demand.shape
        x = torch.cat([demand, time_feats], dim=-1)  # (B, T, Z + 4)
        x = x.reshape(B * T, -1)
        x = self.mlp(x)
        x = x.reshape(B, T, -1)
        return x


class FutureTabularEncoder(nn.Module):
    """Encodes future [zero-padded demand + calendar] → tabular token."""
    
    def __init__(self, n_zones: int, n_time_features: int = 4, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        # Zero-padded demand (n_zones) + time features (4)
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
            time_feats: (B, T, 4)
            n_zones: int
        Returns:
            (B, T, embed_dim)
        """
        B, T, _ = time_feats.shape
        # Zero-pad missing demand
        zero_demand = torch.zeros(B, T, n_zones, device=time_feats.device)
        x = torch.cat([zero_demand, time_feats], dim=-1)  # (B, T, Z + 4)
        x = x.reshape(B * T, -1)
        x = self.mlp(x)
        x = x.reshape(B, T, -1)
        return x


# ---------------------------------------------------------------------------
# Main Transformer Model (Part 1 Baseline)
# ---------------------------------------------------------------------------

class CNNTransformerModel(nn.Module):
    """
    CNN-Transformer Patch Architecture for energy forecasting.
    
    Sequence: (S + 24) × (P + 1) tokens where P = spatial patches, 1 = tabular.
    """
    
    def __init__(
        self,
        n_zones: int,
        history_len: int = 168,
        future_len: int = 24,
        grid_size: int = 10,
        embed_dim: int = 96,
        n_transformer_layers: int = 3,
        n_heads: int = 8,
        mlp_dim: int = 384,
        dropout: float = 0.2,
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
        self.dropout = dropout
        
        # Energy normalization (per-zone)
        if energy_mean is None:
            energy_mean = np.zeros(n_zones, dtype=np.float32)
        if energy_std is None:
            energy_std = np.ones(n_zones, dtype=np.float32)
        
        if isinstance(energy_mean, np.ndarray):
            self.register_buffer('energy_mean', torch.from_numpy(energy_mean))
            self.register_buffer('energy_std', torch.from_numpy(energy_std))
        else:
            self.register_buffer('energy_mean', torch.full((n_zones,), float(energy_mean)))
            self.register_buffer('energy_std', torch.full((n_zones,), float(energy_std)))
        
        # Weather normalization
        if weather_mean is None:
            weather_mean = torch.zeros(7)
        if weather_std is None:
            weather_std = torch.ones(7)
        self.register_buffer('weather_mean', weather_mean.float())
        self.register_buffer('weather_std', weather_std.float())
        
        # 1. Spatial Token CNN
        self.spatial_cnn = SpatialTokenCNN(
            in_channels=7,
            out_grid_size=grid_size,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        
        # 2. Tabular encoders
        self.hist_tabular = HistoricalTabularEncoder(n_zones, 4, embed_dim, dropout)
        self.fut_tabular = FutureTabularEncoder(n_zones, 4, embed_dim, dropout)
        
        # 3. Learnable spatial positional embeddings (for grid locations)
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.n_spatial_tokens, embed_dim) * 0.02
        )
        
        # 4. Per-timestep positional encodings (learnable)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, history_len + future_len, 1, embed_dim) * 0.02
        )
        
        # 5. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
        # 6. Prediction MLP (on 24 future timesteps, with dropout)
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
    
    def adapt_inputs(
        self,
        history_weather: torch.Tensor,
        history_energy: torch.Tensor,
        future_weather: torch.Tensor,
        future_time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = history_weather.device
        
        w_mean = self.weather_mean.to(device)
        w_std = self.weather_std.to(device)
        history_weather = (history_weather - w_mean) / (w_std + 1e-6)
        future_weather = (future_weather - w_mean) / (w_std + 1e-6)
        
        e_mean = self.energy_mean.to(device)
        e_std = self.energy_std.to(device)
        history_energy = (history_energy - e_mean.unsqueeze(0).unsqueeze(0)) / (e_std.unsqueeze(0).unsqueeze(0) + 1e-6)
        
        def make_time_feats(time_vals):
            h = time_vals.float()
            hour_of_day = (h % 24.0) / 24.0 * 2 * math.pi
            day_of_week = ((h // 24.0) % 7.0) / 7.0 * 2 * math.pi
            feats = torch.stack([
                torch.sin(hour_of_day), torch.cos(hour_of_day),
                torch.sin(day_of_week), torch.cos(day_of_week),
            ], dim=-1)
            return feats
        
        first_future_hour = future_time[:, 0:1].float()
        hist_offset = torch.arange(self.history_len, 0, -1, device=device, dtype=torch.float32)
        hist_hours = first_future_hour - hist_offset.unsqueeze(0)
        hist_time_feats = make_time_feats(hist_hours)
        fut_time_feats = make_time_feats(future_time.float())
        
        return history_weather, history_energy, future_weather, hist_time_feats, fut_time_feats
    
    def forward(
        self,
        history_weather: torch.Tensor,
        history_energy: torch.Tensor,
        future_weather: torch.Tensor,
        hist_time_feats: torch.Tensor,
        fut_time_feats: torch.Tensor,
    ) -> torch.Tensor:
        B = history_weather.shape[0]
        
        hist_spatial = self.spatial_cnn(history_weather)
        fut_spatial = self.spatial_cnn(future_weather)
        
        hist_tabular = self.hist_tabular(history_energy, hist_time_feats)
        fut_tabular = self.fut_tabular(fut_time_feats, self.n_zones)
        
        hist_spatial = hist_spatial + self.spatial_pos_embed
        fut_spatial = fut_spatial + self.spatial_pos_embed
        
        hist_steps = []
        for t in range(self.history_len):
            step_tokens = torch.cat([
                hist_spatial[:, t, :, :],
                hist_tabular[:, t:t+1, :],
            ], dim=1)
            hist_steps.append(step_tokens)
        
        fut_steps = []
        for t in range(self.future_len):
            step_tokens = torch.cat([
                fut_spatial[:, t, :, :],
                fut_tabular[:, t:t+1, :],
            ], dim=1)
            fut_steps.append(step_tokens)
        
        sequence = torch.cat(hist_steps + fut_steps, dim=1)
        
        temporal_embeds = self.temporal_pos_embed.expand(B, -1, self.tokens_per_step, -1)
        temporal_embeds = temporal_embeds.reshape(B, (self.history_len + self.future_len) * self.tokens_per_step, self.embed_dim)
        sequence = sequence + temporal_embeds
        
        transformer_out = self.transformer(sequence)
        
        future_start_idx = self.history_len * self.tokens_per_step
        future_tokens = transformer_out[:, future_start_idx:, :]
        
        future_tokens = future_tokens.reshape(B, self.future_len, self.tokens_per_step, self.embed_dim)
        future_tokens_flat = future_tokens.reshape(B, self.future_len, -1)
        
        predictions = self.predictor(future_tokens_flat)
        
        energy_mean = self.energy_mean.to(predictions.device)
        energy_std = self.energy_std.to(predictions.device)
        predictions = predictions * energy_std.unsqueeze(0).unsqueeze(0) + energy_mean.unsqueeze(0).unsqueeze(0)
        
        return predictions


# ---------------------------------------------------------------------------
# Part 2: Encoder-Decoder CNN-Transformer
# ---------------------------------------------------------------------------

class CNNEncoderDecoderModel(nn.Module):
    """
    Encoder-Decoder CNN-Transformer for energy forecasting (Part 2).

    Key differences from the Part 1 flat Transformer:
    - Encoder: processes only historical tokens (S × (P+1)) with self-attention.
    - Decoder: processes only future tokens (24 × (P+1)) with self-attention
      AND cross-attention into the encoder memory.
    - This respects the natural historical/future boundary in the data,
      avoids the future tokens attending to each other causally,
      and cuts the quadratic attention cost significantly.

    Sequence lengths:
      Encoder input : history_len × (P + 1)  =  168 × 101  ≈ 16,968 tokens
      Decoder input : future_len  × (P + 1)  =   24 × 101  ≈  2,424 tokens
      (vs. 19,392 tokens in the flat baseline)
    """

    def __init__(
        self,
        n_zones: int,
        history_len: int = 168,
        future_len: int = 24,
        grid_size: int = 10,
        embed_dim: int = 96,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        n_heads: int = 8,
        mlp_dim: int = 384,
        dropout: float = 0.2,
        weather_mean=None,
        weather_std=None,
        energy_mean=None,
        energy_std=None,
    ):
        super().__init__()
        self.n_zones          = n_zones
        self.history_len      = history_len
        self.future_len       = future_len
        self.grid_size        = grid_size
        self.embed_dim        = embed_dim
        self.n_spatial_tokens = grid_size * grid_size
        self.tokens_per_step  = self.n_spatial_tokens + 1

        # ── Normalization buffers ────────────────────────────────────────────
        if energy_mean is None:
            energy_mean = np.zeros(n_zones, dtype=np.float32)
        if energy_std is None:
            energy_std  = np.ones(n_zones,  dtype=np.float32)
        if isinstance(energy_mean, np.ndarray):
            self.register_buffer('energy_mean', torch.from_numpy(energy_mean))
            self.register_buffer('energy_std',  torch.from_numpy(energy_std))
        else:
            self.register_buffer('energy_mean', torch.full((n_zones,), float(energy_mean)))
            self.register_buffer('energy_std',  torch.full((n_zones,), float(energy_std)))

        if weather_mean is None:
            weather_mean = torch.zeros(7)
        if weather_std is None:
            weather_std  = torch.ones(7)
        self.register_buffer('weather_mean', weather_mean.float())
        self.register_buffer('weather_std',  weather_std.float())

        # ── Shared CNN (same architecture as baseline) ───────────────────────
        self.spatial_cnn = SpatialTokenCNN(
            in_channels=7,
            out_grid_size=grid_size,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # ── Tabular encoders ─────────────────────────────────────────────────
        self.hist_tabular = HistoricalTabularEncoder(n_zones, 4, embed_dim, dropout)
        self.fut_tabular  = FutureTabularEncoder(n_zones, 4, embed_dim, dropout)

        # ── Positional embeddings ────────────────────────────────────────────
        # Spatial: shared across encoder and decoder (same grid locations)
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.n_spatial_tokens, embed_dim) * 0.02
        )
        # Temporal: separate for encoder (history) and decoder (future)
        self.enc_temporal_pos = nn.Parameter(
            torch.randn(1, history_len, 1, embed_dim) * 0.02
        )
        self.dec_temporal_pos = nn.Parameter(
            torch.randn(1, future_len, 1, embed_dim) * 0.02
        )

        # ── Transformer Encoder (historical self-attention only) ─────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # ── Transformer Decoder (future self-attention + cross-attention) ────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # ── Prediction MLP ───────────────────────────────────────────────────
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

    def _build_sequence(self, spatial, tabular, temporal_pos, T):
        """
        Assemble per-timestep tokens into a flat sequence and add positional embeddings.

        Args:
            spatial     : (B, T, P, D)
            tabular     : (B, T, D)
            temporal_pos: (1, T, 1, D)  learnable temporal embedding
            T           : int

        Returns:
            (B, T*(P+1), D)
        """
        B = spatial.shape[0]
        # Add spatial positional embedding
        spatial = spatial + self.spatial_pos_embed          # (B, T, P, D)

        # Stack spatial + tabular token for each step → (B, T, P+1, D)
        tabular_exp = tabular.unsqueeze(2)                  # (B, T, 1, D)
        tokens = torch.cat([spatial, tabular_exp], dim=2)   # (B, T, P+1, D)

        # Add temporal positional embedding (broadcast over P+1 tokens per step)
        tokens = tokens + temporal_pos.expand(B, T, self.tokens_per_step, self.embed_dim)

        # Flatten to (B, T*(P+1), D)
        return tokens.reshape(B, T * self.tokens_per_step, self.embed_dim)

    def forward(
        self,
        history_weather:  torch.Tensor,   # (B, 168, 450, 449, 7)
        history_energy:   torch.Tensor,   # (B, 168, Z)
        future_weather:   torch.Tensor,   # (B,  24, 450, 449, 7)
        hist_time_feats:  torch.Tensor,   # (B, 168, 4)
        fut_time_feats:   torch.Tensor,   # (B,  24, 4)
    ) -> torch.Tensor:                    # (B,  24, Z)
        B = history_weather.shape[0]

        # 1. Extract spatial tokens via shared CNN
        hist_spatial = self.spatial_cnn(history_weather)   # (B, 168, P, D)
        fut_spatial  = self.spatial_cnn(future_weather)    # (B,  24, P, D)

        # 2. Encode tabular tokens
        hist_tab = self.hist_tabular(history_energy, hist_time_feats)  # (B, 168, D)
        fut_tab  = self.fut_tabular(fut_time_feats, self.n_zones)      # (B,  24, D)

        # 3. Build encoder sequence: history tokens (B, 168*(P+1), D)
        enc_seq = self._build_sequence(
            hist_spatial, hist_tab, self.enc_temporal_pos, self.history_len
        )

        # 4. Build decoder sequence: future tokens (B, 24*(P+1), D)
        dec_seq = self._build_sequence(
            fut_spatial, fut_tab, self.dec_temporal_pos, self.future_len
        )

        # 5. Encoder: self-attention over historical sequence only
        memory = self.encoder(enc_seq)                     # (B, 168*(P+1), D)

        # 6. Decoder: self-attention over future tokens + cross-attention to memory
        #    This is the key change vs Part 1 — future tokens can attend to the
        #    full historical context via cross-attention, but the encoder never
        #    sees the future, respecting the causal boundary.
        dec_out = self.decoder(dec_seq, memory)            # (B, 24*(P+1), D)

        # 7. Reshape decoder output → (B, 24, P+1, D) → flatten last two dims
        dec_out = dec_out.reshape(B, self.future_len, self.tokens_per_step, self.embed_dim)
        dec_out_flat = dec_out.reshape(B, self.future_len, -1)

        # 8. MLP prediction head
        predictions = self.predictor(dec_out_flat)         # (B, 24, Z)

        # 9. Un-normalize
        e_mean = self.energy_mean.to(predictions.device)
        e_std  = self.energy_std.to(predictions.device)
        predictions = predictions * e_std.unsqueeze(0).unsqueeze(0) \
                                  + e_mean.unsqueeze(0).unsqueeze(0)
        return predictions

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,
        history_energy:  torch.Tensor,
        future_weather:  torch.Tensor,
        future_time:     torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize inputs and compute time features (mirrors train.py)."""
        device = history_weather.device

        w_mean = self.weather_mean.to(device)
        w_std  = self.weather_std.to(device)
        history_weather = (history_weather - w_mean) / (w_std + 1e-6)
        future_weather  = (future_weather  - w_mean) / (w_std + 1e-6)

        e_mean = self.energy_mean.to(device)
        e_std  = self.energy_std.to(device)
        history_energy  = (history_energy - e_mean) / (e_std + 1e-6)

        def make_time_feats(t):
            h = t.float()
            hod = (h % 24.0) / 24.0 * 2 * math.pi
            dow = ((h // 24.0) % 7.0) / 7.0 * 2 * math.pi
            return torch.stack([torch.sin(hod), torch.cos(hod),
                                 torch.sin(dow), torch.cos(dow)], dim=-1)

        first_future = future_time[:, 0:1].float()
        offsets      = torch.arange(self.history_len, 0, -1, device=device, dtype=torch.float32)
        hist_hours   = first_future - offsets.unsqueeze(0)
        hist_time_feats = make_time_feats(hist_hours)
        fut_time_feats  = make_time_feats(future_time.float())

        return history_weather, history_energy, future_weather, hist_time_feats, fut_time_feats


# ---------------------------------------------------------------------------
# Factory function required by evaluate.py
# ---------------------------------------------------------------------------

def get_model(metadata: dict) -> CNNEncoderDecoderModel:
    """
    Instantiate CNNEncoderDecoderModel with architecture hyperparameters and
    normalization stats loaded from norm_stats.pt, then load trained weights.
    All hyperparameters are saved into norm_stats.pt by train.py, so the
    reconstructed model always matches the saved checkpoint exactly.
    """
    model_dir  = Path(__file__).parent
    stats_path = model_dir / "norm_stats.pt"

    # ── Load norm stats + architecture hyperparameters ────────────────────────
    if stats_path.exists():
        stats        = torch.load(stats_path, map_location="cpu", weights_only=True)
        weather_mean = torch.tensor(stats["weather_mean"], dtype=torch.float32)
        weather_std  = torch.tensor(stats["weather_std"],  dtype=torch.float32)
        energy_mean  = np.array(stats["energy_mean"], dtype=np.float32)
        energy_std   = np.array(stats["energy_std"],  dtype=np.float32)
        history_len      = int(stats.get("history_len",      168))
        grid_size        = int(stats.get("grid_size",         10))
        embed_dim        = int(stats.get("embed_dim",         96))
        n_encoder_layers = int(stats.get("n_encoder_layers",   3))
        n_decoder_layers = int(stats.get("n_decoder_layers",   3))
        n_heads          = int(stats.get("n_heads",            8))
        mlp_dim          = int(stats.get("mlp_dim",          384))
        dropout          = float(stats.get("dropout",         0.2))
        print(
            f"  Loaded norm_stats.pt: history_len={history_len}, "
            f"embed_dim={embed_dim}, n_encoder_layers={n_encoder_layers}, "
            f"n_decoder_layers={n_decoder_layers}, n_heads={n_heads}, "
            f"grid_size={grid_size}"
        )
    else:
        print("  WARNING: norm_stats.pt not found — using default hyperparameters")
        weather_mean = weather_std = energy_mean = energy_std = None
        history_len      = 168
        grid_size        = 10
        embed_dim        = 96
        n_encoder_layers = 3
        n_decoder_layers = 3
        n_heads          = 8
        mlp_dim          = 384
        dropout          = 0.2

    # ── Build model ───────────────────────────────────────────────────────────
    model = CNNEncoderDecoderModel(
        n_zones          = metadata["n_zones"],
        history_len      = history_len,
        future_len       = metadata.get("future_len", 24),
        grid_size        = grid_size,
        embed_dim        = embed_dim,
        n_encoder_layers = n_encoder_layers,
        n_decoder_layers = n_decoder_layers,
        n_heads          = n_heads,
        mlp_dim          = mlp_dim,
        dropout          = dropout,
        weather_mean     = weather_mean,
        weather_std      = weather_std,
        energy_mean      = energy_mean,
        energy_std       = energy_std,
    )

    # ── Load trained weights ──────────────────────────────────────────────────
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        candidates = sorted(model_dir.glob("best_model_*.pt"), reverse=True)
        ckpt_path  = candidates[0] if candidates else None

    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"  Loaded weights from {ckpt_path}")
    else:
        print("  WARNING: No checkpoint found — model is randomly initialized!")

    return model