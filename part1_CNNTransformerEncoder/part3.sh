#!/bin/bash
#SBATCH --job-name=part3_graph
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nliu05@tufts.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=train_eval_%j.out
#SBATCH --error=train_eval_%j.err

# Load modules
module load class/default
module load cs137/2026spring

python "part3_attention_maps.py"

