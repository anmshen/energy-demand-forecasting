#!/bin/bash
#SBATCH --job-name=as_nl_3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashen05@tufts.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=train_eval_%j.out
#SBATCH --error=train_eval_%j.err

module load class/default
module load cs137/2026spring

N_DAYS=${1:-25}
RUN_TAG=${2:-default}

EVAL_DIR="/cluster/tufts/c26sp1cs0137/ashen05/assignment3/evaluation"
MODEL_NAME="as_nl_3"
MODEL_DIR="$EVAL_DIR/$MODEL_NAME"
CKPT_DIR="$MODEL_DIR/checkpoints_$RUN_TAG"

mkdir -p "$MODEL_DIR" "$CKPT_DIR"

# Remove stale promoted artifacts so evaluation cannot load a checkpoint
# from a different run that has a mismatched architecture.
rm -f "$MODEL_DIR/best_model.pt" "$MODEL_DIR/norm_stats.pt"

echo "Starting training + evaluation for model: $MODEL_NAME"
echo "  MODEL_DIR   : $MODEL_DIR"
echo "  CHECKPOINTS : $CKPT_DIR"
echo "  RUN_TAG     : $RUN_TAG"
echo "  GPUs        : $CUDA_VISIBLE_DEVICES"

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS=50
BATCH_SIZE=4
LR=1e-4
HISTORY_LEN=168
GRID_SIZE=10
EMBED_DIM=64
N_LAYERS=3
N_HEADS=8
MLP_DIM=256
DROPOUT=0.2
N_TRAIN_DAYS=1400
N_VAL_DAYS=30

echo ""
echo "=== Hyperparameters ==="
echo "  epochs       : $EPOCHS"
echo "  batch_size   : $BATCH_SIZE"
echo "  lr           : $LR"
echo "  history_len  : $HISTORY_LEN"
echo "  grid_size    : $GRID_SIZE"
echo "  embed_dim    : $EMBED_DIM"
echo "  n_layers     : $N_LAYERS"
echo "  n_heads      : $N_HEADS"
echo "  mlp_dim      : $MLP_DIM"
echo "  dropout      : $DROPOUT"
echo "  n_train_days : $N_TRAIN_DAYS"
echo "  n_val_days   : $N_VAL_DAYS"

# ── Training ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Training ==="
python "$MODEL_DIR/train.py" \
    --epochs       $EPOCHS       \
    --batch_size   $BATCH_SIZE   \
    --lr           $LR           \
    --history_len  $HISTORY_LEN  \
    --grid_size    $GRID_SIZE    \
    --embed_dim    $EMBED_DIM    \
    --n_layers     $N_LAYERS     \
    --n_heads      $N_HEADS      \
    --mlp_dim      $MLP_DIM      \
    --dropout      $DROPOUT      \
    --n_train_days $N_TRAIN_DAYS \
    --n_val_days   $N_VAL_DAYS   \
    --save_dir    "$CKPT_DIR"

# ── Copy artifacts next to model.py ──────────────────────────────────────────
echo ""
echo "=== Copying artifacts ==="
BEST=$(ls -t "$CKPT_DIR"/best_model_*.pt 2>/dev/null | head -1)
if [ -z "$BEST" ]; then
    echo "ERROR: No checkpoint found in $CKPT_DIR — training may have failed."
    exit 1
fi
cp "$BEST"                    "$MODEL_DIR/best_model.pt"
cp "$CKPT_DIR/norm_stats.pt"  "$MODEL_DIR/norm_stats.pt"
echo "  Checkpoint : $BEST"
echo "  Copied to  : $MODEL_DIR/best_model.pt"

# ── Evaluation ───────────────────────────────────────────────────────────────
cd "$EVAL_DIR"

echo ""
echo "=== Evaluating $MODEL_NAME ($N_DAYS days) ==="
python evaluate.py "$MODEL_NAME" "$N_DAYS"

echo ""
echo "=== Evaluating example_model ($N_DAYS days) ==="
python evaluate.py example_model "$N_DAYS"
