# Assignment 3 — Part 2: CNN Encoder–Decoder Transformer

## Overview

Part 2 forecasts **24-hour zonal energy demand** for eight New England zones using **historical weather maps, historical demand, and future weather**. The model is a **CNN encoder–decoder Transformer** (`CNNEncoderDecoderModel`): a shared CNN turns each hourly weather frame into spatial tokens; historical and future **tabular tokens** (demand + calendar, or calendar-only for the future) are concatenated per timestep; a **Transformer encoder** reads only the **past** sequence, and a **Transformer decoder** attends to the **future** token sequence with **cross-attention** to encoder memory. This matches the train/test causal boundary and reduces attention cost versus flattening all timesteps into one long sequence (as in the Part 1 flat Transformer baseline).

Training and evaluation follow the same data layout and `evaluate.py` contract as Part 1.

## Directory structure

```
assignment3/part2/
├── train.py                    # Trains CNNEncoderDecoderModel; saves checkpoints + norm_stats.pt
├── train_eval.sh               # SLURM job: train, copy best checkpoint, run evaluate.py
├── plot_loss_curves.py         # Plots train/val MSE, MAPE, LR from a captured training log → loss_curves_part2.png
├── train_1.py                  # Copy/variant of train.py (same Part 2 model import path)
├── train_eval_1.sh             # Alternate SLURM wrapper (if used)
├── stub_performance.txt        # Reference stub / older baseline numbers (not Part 2 primary metric)
└── evaluation/
    ├── evaluate.py             # Loads model folders; MAPE on held-out dates in 2023
    ├── part2_best_model/       # Part 2 submission model
    │   ├── model.py            # CNNEncoderDecoderModel + get_model()
    │   ├── best_model.pt       # Best weights (copied from checkpoints by train_eval.sh)
    │   ├── norm_stats.pt       # Weather/energy mean–std + architecture hyperparameters
    │   └── checkpoints/        # Timestamped best_model_*.pt during training
    └── part1_best_model/       # Optional: Part 1 checkpoint folder for side-by-side evaluation
```

## File roles

| File | Role |
|------|------|
| **`evaluation/part2_best_model/model.py`** | Defines **`SpatialTokenCNN`** (weather → \(10\times10\) grid tokens), **`HistoricalTabularEncoder`** / **`FutureTabularEncoder`** (demand + cyclic time features), and **`CNNEncoderDecoderModel`** (PyTorch **`TransformerEncoder`** on history, **`TransformerDecoder`** on future with cross-attention, then an MLP head for \((B,24,\text{zones})\)). Implements **`get_model(metadata)`** and **`adapt_inputs`** for evaluation (normalization + time features). |
| **`train.py`** | Loads cluster weather/energy data, computes normalization stats, builds **`EnergyDataset`** windows (168 h history, 24 h ahead), trains with **MSE** on raw-scale targets, **cosine LR schedule**, **early stopping on validation MSE** (patience 15). Saves **`norm_stats.pt`** (includes `embed_dim`, layer counts, etc.) and timestamped **`best_model_*.pt`**. |
| **`evaluation/evaluate.py`** | Imports `evaluation/<MODEL_NAME>/model.py`, runs **`get_model`**, evaluates **MAPE** per zone and overall on configurable days in **`TEST_YEAR`** (default 2023). |
| **`train_eval.sh`** | **`sbatch`** script: runs **`train.py`** with fixed hyperparameters, copies latest **`best_model_*.pt`** to **`part2_best_model/best_model.pt`**, copies **`norm_stats.pt`**, runs **`evaluate.py part2_best_model`**, then optionally **`evaluate.py part1_best_model`** for comparison. |
| **`plot_loss_curves.py`** | Reproduces **training/validation curves** from logged epoch metrics (e.g. job `.out`); writes **`loss_curves_part2.png`**. |

## Model structure (summary)

1. **Shared CNN (`SpatialTokenCNN`)**  
   Input weather \((B,T,450,449,7)\) → conv blocks + adaptive pool → \((B,T,P,D)\) with \(P=\texttt{grid\_size}^2\) (default **10×10 → 100** spatial tokens per hour).

2. **Tokens per hour**  
   For each hour: **100 spatial tokens + 1 tabular token** (\(P{+}1=101\)). Tabular side uses **per-zone demand + 4 cyclic time features** for history; **zeros + time features** for future hours.

3. **Encoder**  
   Sequence length **168 × 101**: self-attention over **historical** tokens only → encoder **memory**.

4. **Decoder**  
   Sequence length **24 × 101**: self-attention over **future** tokens + **cross-attention** to encoder memory → outputs reshaped to \((B,24,101,D)\), flattened per timestep, **MLP predictor** → \((B,24,\text{n\_zones})\), then **denormalization** to real energy units.

5. **Positional information**  
   Learnable **spatial** embeddings (shared), separate learnable **temporal** embeddings for encoder vs decoder steps.

**Checkpoint metadata** (saved in `norm_stats.pt` and printed at eval): e.g. **`history_len=168`**, **`embed_dim`**, **`n_encoder_layers`**, **`n_decoder_layers`**, **`n_heads`**, **`grid_size`** — must match the trained weights.

## Best reported performance (`train_eval_475554.out`)

Results below come from the logged run that trains **`part2_best_model`** and evaluates on **2023**, **25 consecutive days** (**2023-12-07** → **2023-12-31**), **8 zones**.

### Training (validation)

- **Parameters:** **8,444,200**
- **Split (this run):** 600 train days, 30 validation days; training **up to 30 epochs**, early stopping **patience 15** on **validation MSE**.
- **Best validation checkpoint:** **lowest val MSE = 22,340.0476** at **epoch 29** (at that epoch: **val MAPE 6.84%**, train MSE **51,260.61**, train MAPE **8.96%**).  
  *(Epoch 30 had slightly higher val MSE **23,361.62**.)*

### Test (evaluation script)

**Model: `part2_best_model` (`CNNEncoderDecoderModel`)**

| Zone | MAPE |
|------|------|
| ME | 9.38% |
| NH | 6.22% |
| VT | 11.34% |
| CT | 6.62% |
| RI | 6.47% |
| SEMA | 6.94% |
| WCMA | 6.05% |
| NEMA_BOST | 5.25% |
| **Overall** | **7.28%** |

The same log file also evaluates **`part1_best_model`** on the same **25 days** for comparison (**overall MAPE 5.37%** in that run). Part 2’s goal is the encoder–decoder architecture and behavior; **direct numeric comparison** depends on matching training budget and hyperparameters.

## Workflow

### Train locally / interactively

```bash
cd assignment3/part2
python train.py --epochs 30 --batch_size 4 --lr 1e-3 \
  --grid_size 10 --embed_dim 128 --n_transformer_layers 3 \
  --n_heads 8 --mlp_dim 512 --dropout 0.3 --n_train_days 600 \
  --save_dir evaluation/part2_best_model/checkpoints
```

Then copy the latest **`best_model_*.pt`** and **`norm_stats.pt`** into **`evaluation/part2_best_model/`** as **`best_model.pt`** and **`norm_stats.pt`** (or use **`train_eval.sh`**, which does this).

### Evaluate

From **`assignment3/part2/evaluation/`**:

```bash
python evaluate.py part2_best_model 25
```

### Cluster (SLURM)

```bash
sbatch train_eval.sh 25
```

Optional argument: number of **evaluation days** (default **25** in `train_eval.sh`).

## Model interface (evaluation contract)

Same as Part 1: `get_model(metadata)` returns a module implementing **`adapt_inputs(...)`** and **`forward(...)`** such that the network outputs **`(B, 24, n_zones)`** demand in **original units**.

**Inputs:** `history_weather` \((B,168,450,449,7)\), `history_energy` \((B,168,n\_zones)\), `future_weather` \((B,24,450,449,7)\), `future_time` \((B,24)\) int64.

## Data paths

- **Weather:** `/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data/`
- **Energy:** `/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data/`

## Notes

- **Normalization:** Weather is normalized channel-wise; energy is **per-zone** z-score using statistics from the **training** windows. **`norm_stats.pt`** must accompany **`best_model.pt`** so inference matches training.
- **Loss:** Training optimizes **MSE** on **denormalized** (physical scale) targets; **MAPE** is logged for monitoring and used in **`evaluate.py`** for reporting.
