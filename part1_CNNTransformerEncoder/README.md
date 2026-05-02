# Assignment 3: Energy Demand Forecasting

## Overview
This assignment implements a neural network model to forecast energy demand for 8 zones in New England using weather data and historical energy patterns. The workflow is organized into two main stages: **training** and **evaluation**.

## Directory Structure

```
assignment3/part1_CNNTransformerEncoder
├── train.py              # Main training script
├── train_eval.sh         # SLURM wrapper for training + evaluation
├── evaluation/           # Evaluation environment
│   ├── evaluate.py       # Evaluation script (loads models from subdirs)
│   ├── example_model/    # Baseline stub model (persistence)
│   │   ├── model.py      # StubModel implementation
│   │   └── submodule.py  # Stub submodule (example for imports)
│   └── our_model/        # Your trained energy forecast model
│       ├── model.py      # EnergyForecastModel implementation
│       ├── best_model.pt # Best checkpoint (loaded by train_eval.sh)
│       ├── norm_stats.pt # Normalization statistics
│       └── checkpoints/  # Training intermediate checkpoints
│── stub_performance.txt  # Reference baseline performance
└── part3_aattention_maps.py # plots the attention maps using part 1 model

```

## Workflow

### 1. Training

Run the main training script to train the EnergyForecastModel:

```bash
python train.py [OPTIONS]
```

**Options:**
- `--epochs 20` — Number of training epochs (default: 20)
- `--batch_size 8` — Batch size (default: 8)
- `--lr 1e-3` — Learning rate (default: 1e-3)
- `--hidden 256` — Hidden dimension size (default: 256)
- `--n_train_days 300` — Number of training windows (default: 300)
- `--save_dir ./checkpoints` — Where to save checkpoints (default: ./checkpoints)

**What it does:**
1. Loads weather (.pt) and energy demand (.csv) data from the cluster
2. Computes normalization statistics (mean/std) from the training split
3. Trains using Adam optimizer with ReduceLROnPlateau scheduler
4. Saves best checkpoint (lowest validation MAPE) to `<save_dir>/best_model.pt`
5. Saves normalization stats to `<save_dir>/norm_stats.pt`

### 2. Evaluation

Evaluate a trained model on the test set:

```bash
python evaluation/evaluate.py <MODEL_NAME> [N_DAYS]
```

**Arguments:**
- `MODEL_NAME` — Name of the model folder (e.g., `our_model`, `example_model`)
- `N_DAYS` — Number of test days to evaluate (optional, default varies by model)

**Output:**
- Per-zone Mean Absolute Percentage Error (MAPE)
- Overall MAPE across all zones and timesteps

### 3. Train + Evaluate (SLURM)

For cluster submission with GPU support, use the SLURM wrapper:

```bash
sbatch train_eval.sh [N_DAYS]
```

This script:
1. Trains the model using `train.py`
2. Copies the best checkpoint and norm stats to `evaluation/our_model/`
3. Evaluates on the test set using `evaluate.py`
4. Generates SLURM output logs with job ID

## Model Interface

Both example_model and our_model must implement:

```python
def get_model(metadata: dict) -> torch.nn.Module:
    """Return a PyTorch model instance."""
    ...

class YourModel(nn.Module):
    def adapt_inputs(self, history_weather, history_energy, 
                     future_weather, future_time) -> tuple:
        """Preprocess raw evaluation inputs before forward pass."""
        ...
    
    def forward(self, *args) -> torch.Tensor:
        """Return (B, 24, n_zones) predictions."""
        ...
```

**Input Shapes:**
- `history_weather` : (B, 168, 450, 449, 7) — 7-day weather history on grid
- `history_energy` : (B, 168, n_zones) — 7-day energy demand history
- `future_weather` : (B, 24, 450, 449, 7) — 24-hour weather forecast on grid
- `future_time` : (B, 24) int64 — hours since Unix epoch

**Output:**
- (B, 24, n_zones) — 24-hour energy demand forecast (un-normalized)

## Data Paths

All data is loaded from:
- **Weather:** `/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data/`
- **Energy:** `/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data/`

## Baselines

**example_model (StubModel):** Persistence baseline that predicts the next 24 hours of energy by repeating the last 24 hours of history.
- Overall MAPE: ~4.43% (see stub_performance.txt)

**our_model (EnergyForecastModel):** Neural network with:
- Separate encoders for history weather and energy
- Per-timestep encoder for future weather + cyclic time features
- Fusion decoder with batch normalization throughout
- Achieves better than baseline performance through learned patterns

## Notes

- **Normalization:** Weather and energy data are z-score normalized using statistics computed from the training split (2021-2022). These stats are saved and loaded automatically to ensure evaluation consistency.
- **Checkpoints:** All intermediate checkpoints are saved in `evaluation/our_model/checkpoints/` with timestamps for debugging.
- **Submodules:** The example_model includes a submodule import test to ensure the evaluation framework can load complex model structures.
# energy-demand-forecasting
