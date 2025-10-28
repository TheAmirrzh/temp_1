#!/bin/bash

# Full Pipeline Script for Horn Clause GNN Project
# This script orchestrates data generation, spectral feature
# preprocessing, and model training.

# --- Configuration ---

# Directories (adjust paths as needed)
DATA_DIR="./generated_data"        # Where raw JSON instances will be saved
SPECTRAL_DIR="./spectral_cache"    # Where precomputed spectral features will be saved
EXP_DIR="./experiment_results"     # Where training results (model, logs) will be saved
VENV_PYTHON="/Users/amirmac/WorkSpace/Codes/LogNet/.venv/bin/python"   # Path to your virtual environment's Python

# Data Generation Config (Number of instances per difficulty)
N_EASY=50
N_MEDIUM=30
N_HARD=10

# Spectral Preprocessing Config
SPECTRAL_K=16
NUM_WORKERS_SPECTRAL=$(($(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) - 1)) # Use N-1 cores, default 4

# Training Config (Example - can be overridden by train.py defaults/args)
EPOCHS=50
BATCH_SIZE=32
HIDDEN_DIM=128
LEARNING_RATE=0.0005
VALUE_LOSS_WEIGHT=0.1
TACTIC_LOSS_WEIGHT=0.1

# --- Script Logic ---

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting Full Horn Clause GNN Pipeline..."
echo "================================================="
date

# 1. Create Directories
echo "📂 [Step 1/4] Creating directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$SPECTRAL_DIR"
mkdir -p "$EXP_DIR"
echo "   - Data directory: $DATA_DIR"
echo "   - Spectral cache: $SPECTRAL_DIR"
echo "   - Experiment output: $EXP_DIR"
echo "✅ Directories created/ensured."
echo "-------------------------------------------------"

# 2. Data Generation
echo "🧬 [Step 2/4] Generating dataset..."
"$VENV_PYTHON" data_generator.py \
    --output-dir "$DATA_DIR" \
    --easy $N_EASY \
    --medium $N_MEDIUM \
    --hard $N_HARD \
    # Add other difficulties (--very-hard, --extreme-hard) if needed
echo "✅ Dataset generated in $DATA_DIR."
echo "-------------------------------------------------"

# 3. Spectral Feature Preprocessing
echo "📊 [Step 3/4] Preprocessing spectral features..."
# --- MODIFIED: Corrected argument hyphens ---
"$VENV_PYTHON" batch_process_spectral.py \
    --data_dir "$DATA_DIR" \       # Corrected: two hyphens and underscore
    --output_dir "$SPECTRAL_DIR" \   # Corrected: two hyphens and underscore
    --k $SPECTRAL_K \
    --num_workers $NUM_WORKERS_SPECTRAL \
    --adaptive-k # This flag should be okay if the script runs
# --- END MODIFIED ---
echo "✅ Spectral features saved in $SPECTRAL_DIR."
echo "-------------------------------------------------"

# 4. Model Training
echo "🧠 [Step 4/4] Starting model training..."
"$VENV_PYTHON" train.py \
    --data-dir "$DATA_DIR" \
    --spectral-dir "$SPECTRAL_DIR" \
    --exp-dir "$EXP_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --hidden-dim $HIDDEN_DIM \
    --lr $LEARNING_RATE \
    --value-loss-weight $VALUE_LOSS_WEIGHT \
    --tactic-loss-weight $TACTIC_LOSS_WEIGHT \
    --k-dim $SPECTRAL_K # Ensure model uses same k as preprocessing
    # Add other train.py arguments as needed (e.g., --num-layers, --dropout)
echo "✅ Training complete. Results in $EXP_DIR."
echo "================================================="

# (Optional) Placeholder for Inference/Search Agent
# echo "🔍 [Optional] Running inference (Beam Search)..."
# "$VENV_PYTHON" run_inference.py \
#     --model-path "$EXP_DIR/best.pt" \
#     --data-dir "$DATA_DIR/test" \ # Assuming a test split directory
#     --output-file "$EXP_DIR/inference_results.json" \
#     --beam-width 5
# echo "✅ Inference complete."
# echo "-------------------------------------------------"

echo "🎉 Pipeline Finished Successfully!"
date
echo "================================================="

exit 0