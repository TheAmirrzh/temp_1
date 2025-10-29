#!/bin/bash

# ==============================================================================
# SOTA Horn Clause GNN - Full Research Pipeline (v5 - Final Fix)
#
# This script is now robust to stale caches and k-dimension mismatches.
#
# 1. Clears the stale spectral cache to force regeneration.
# 2. Disables 'adaptive-k' during preprocessing to ensure a consistent k=16.
# 3. Passes '--k-dim 16' to train.py, which now forces the model AND
#    the dataset loaders to use this exact k-value, overriding auto-detection.
#
# ==============================================================================

# --- Configuration ---

# Directories
DATA_DIR="./generated_data"
SPECTRAL_DIR="./spectral_cache"
EXP_DIR="./experiment_results"
VALIDATION_DIR="./validation_reports"

# Path to your virtual environment's Python
VENV_PYTHON="/Users/amirmac/WorkSpace/Codes/LogNet/.venv/bin/python"

# Data Generation Config
N_EASY=400
N_MEDIUM=400
N_HARD=300
N_VERY_HARD=300

# Spectral Preprocessing Config
SPECTRAL_K=16 # This is the single source of truth for k
NUM_WORKERS_SPECTRAL=$(($(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) - 1))

# Training Config
EPOCHS=10
BATCH_SIZE=32
HIDDEN_DIM=128
NUM_LAYERS=4
LEARNING_RATE=0.000۱
VALUE_LOSS_WEIGHT=0.1
TACTIC_LOSS_WEIGHT=0.2

# --- Script Logic ---

set -e

echo "🚀 Starting Full SOTA Horn Clause GNN Pipeline..."
echo "================================================="
date

# 0. Create Directories
echo "📂 [Step 0/6] Creating directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$SPECTRAL_DIR"
mkdir -p "$EXP_DIR"
mkdir -p "$VALIDATION_DIR"
echo "✅ Directories created."
echo "-------------------------------------------------"

# 1. Data Generation
echo "🧬 [Step 1/6] Generating dataset..."
"$VENV_PYTHON" data_generator.py \
    --output-dir "$DATA_DIR" \
    --easy $N_EASY \
    --medium $N_MEDIUM \
    --hard $N_HARD \
    --very-hard $N_VERY_HARD
echo "✅ Dataset generated in $DATA_DIR."
echo "-------------------------------------------------"

# 2. Clear Stale Cache (NEW FIX)
echo "🔥 [Step 2/6] Clearing stale spectral cache..."
rm -rf "$SPECTRAL_DIR"
mkdir -p "$SPECTRAL_DIR"
echo "✅ Stale cache cleared. Ready for regeneration."
echo "-------------------------------------------------"

# 3. Spectral Feature Preprocessing
echo "📊 [Step 3/6] Preprocessing spectral features (parallelized)..."
# This will now use the *correct* node count (len(nodes))
# We REMOVE --adaptive-k to force a consistent k=16
"$VENV_PYTHON" batch_process_spectral.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$SPECTRAL_DIR" \
    --k $SPECTRAL_K \
    --num_workers $NUM_WORKERS_SPECTRAL
    # --adaptive-k has been REMOVED
echo "✅ Spectral features regenerated in $SPECTRAL_DIR with k=$SPECTRAL_K."
echo "-------------------------------------------------"

# 4. Model Training
echo "🧠 [Step 4/6] Starting model training..."
echo "   - Model: TacticGuidedGNN (via get_model)"
echo "   - Dataset: StepPredictionDataset"
echo "   - Loss: ProofSearchRankingLoss (Fixed) + Value + Tactic"
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
    --k-dim $SPECTRAL_K \
    --num-layers $NUM_LAYERS # <-- ADD THIS LINE with your desired value
echo "✅ Training complete. Results in $EXP_DIR."
echo "================================================="
# 5. Post-Training Validation
echo "🔬 [Step 5/6] Generating post-training validation reports..."
"$VENV_PYTHON" validate_spectral.py \
    --cache-dir "$SPECTRAL_DIR" \
    --output-dir "$VALIDATION_DIR/spectral_report"
"$VENV_PYTHON" validate_temporal.py \
    --output-dir "$VALIDATION_DIR/temporal_report"
echo "✅ Validation reports saved to $VALIDATION_DIR."
echo "-------------------------------------------------"

# 6. Inference (Placeholder)
echo "🔍 [Step 6/6] Inference (Beam Search)..."
echo "   - The next step is to use the trained model for proof search."
# "$VENV_PYTHON" run_search.py \
#     --model-path "$EXP_DIR/best.pt" \
#     --data-dir "$DATA_DIR" \ 
#     --output-file "$EXP_DIR/inference_results.json" \
#     --beam-width 5
echo "✅ Inference placeholder complete."
echo "-------------------------------------------------"

echo "🎉 Full Pipeline Finished Successfully!"
date
echo "================================================="

exit 0