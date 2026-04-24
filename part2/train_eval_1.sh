#!/bin/bash
#SBATCH --job-name=train_eval
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashen05@tufts.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=train_eval_%j.out
#SBATCH --error=train_eval_%j.err

# Load modules
module load class/default
module load cs137/2026spring

N_DAYS=${1:-25}

# All paths anchored to the evaluation/ directory (where evaluate.py lives)
EVAL_DIR="/cluster/tufts/c26sp1cs0137/nliu05/assignment3/part2/evaluation"
MODEL_NAME="part2_best_model"
MODEL_DIR="$EVAL_DIR/$MODEL_NAME"
CHECKPOINTS_DIR="$MODEL_DIR/checkpoints"

mkdir -p "$MODEL_DIR"
mkdir -p "$CHECKPOINTS_DIR"

echo "Starting training + evaluation for model: $MODEL_NAME"
echo "  EVAL_DIR    : $EVAL_DIR"
echo "  MODEL_DIR   : $MODEL_DIR"
echo "  CHECKPOINTS : $CHECKPOINTS_DIR"
echo "  GPUs        : $CUDA_VISIBLE_DEVICES"

# Run training  train.py is one level above evaluation/
echo "=== Training CNN-EncodeDecoder Model ==="
python "$EVAL_DIR/../train_1.py" \
    --epochs 60 \
    --batch_size 4 \
    --lr 1e-3 \
    --grid_size 10 \
    --embed_dim 128 \
    --n_transformer_layers 3 \
    --n_heads 8 \
    --mlp_dim 512 \
    --dropout 0.3 \
    --n_train_days 600 \
    --save_dir "$CHECKPOINTS_DIR"

# Copy the most recently saved checkpoint and norm stats next to model.py
# so that get_model() in model.py can find them
echo "=== Copying best checkpoint ==="
BEST_CKPT=$(ls -t "$CHECKPOINTS_DIR"/best_model_*.pt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: No best_model_*.pt found in $CHECKPOINTS_DIR  training may have failed."
    exit 1
fi
echo "Using checkpoint: $BEST_CKPT"
cp "$BEST_CKPT"                      "$MODEL_DIR/best_model.pt"
cp "$CHECKPOINTS_DIR/norm_stats.pt"  "$MODEL_DIR/norm_stats.pt"

# # cd into evaluation/ so evaluate.py resolves $MODEL_NAME as a subfolder correctly
cd "$EVAL_DIR"
echo "=== Evaluating: $MODEL_NAME for $N_DAYS days ==="
python evaluate.py $MODEL_NAME $N_DAYS

# Also evaluate the stub/baseline model
echo ""
echo "=== Evaluating: example_model (stub baseline) for $N_DAYS days ==="
python evaluate.py part1_best_model $N_DAYS