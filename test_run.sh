#!/bin/bash
#SBATCH --job-name=test_run
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
EVAL_DIR="/cluster/tufts/c26sp1cs0137/ashen05/assignment3/evaluation"
MODEL_NAME="as_nl"
MODEL_DIR="$EVAL_DIR/$MODEL_NAME"
CHECKPOINTS_DIR="$MODEL_DIR/checkpoints"

# Copy the most recently saved checkpoint and norm stats next to model.py
# so that get_model() in model.py can find them
echo "=== Copying best checkpoint ==="
BEST_CKPT=$(ls -t "$CHECKPOINTS_DIR"/best_model_*.pt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: No best_model_*.pt found in $CHECKPOINTS_DIR — training may have failed."
    exit 1
fi
echo "Using checkpoint: $BEST_CKPT"
cp "$BEST_CKPT"                      "$MODEL_DIR/best_model.pt"
cp "$CHECKPOINTS_DIR/norm_stats.pt"  "$MODEL_DIR/norm_stats.pt"

# cd into evaluation/ so evaluate.py resolves $MODEL_NAME as a subfolder correctly
cd "$EVAL_DIR"
echo "=== Evaluating: $MODEL_NAME for $N_DAYS days ==="
python evaluate.py $MODEL_NAME $N_DAYS

# Also evaluate the stub/baseline model
echo ""
echo "=== Evaluating: example_model (stub baseline) for $N_DAYS days ==="
python evaluate.py example_model $N_DAYS