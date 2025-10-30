#!/bin/bash

# ==============================================================================
# SOTA Horn Clause GNN - Full Research Pipeline (SOTA Fix v1.0)
#
# This script runs the complete pipeline using the SOTA-fixed components:
#
# 1. Model: FixedTemporalSpectralGNN (Spectral, Structural, Temporal pathways)
# 2. Loss:  Triplet Ranking + SOTA Value + Contrastive Loss
# 3. Data:  StepPredictionDataset (with contrastive sampling)
# 4. Cache: Clears stale spectral cache to prevent K-mismatch errors.
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
EPOCHS=30
BATCH_SIZE=16
HIDDEN_DIM=128
NUM_LAYERS=3
LEARNING_RATE=0.0005
VALUE_LOSS_WEIGHT=0.1
CONTRAST_LOSS_WEIGHT=0.3   # <-- NEW: Replaces tactic loss
LOSS_TYPE="triplet_hard"   # <-- NEW: Specify SOTA loss
MARGIN=1.0                 # <-- NEW: For triplet loss

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

# 2. Clear Stale Cache (Robustness Fix)
echo "🔥 [Step 2/6] Clearing stale spectral cache..."
rm -rf "$SPECTRAL_DIR"
mkdir -p "$SPECTRAL_DIR"
echo "✅ Stale cache cleared. Ready for regeneration."
echo "-------------------------------------------------"

# 3. Spectral Feature Preprocessing
echo "📊 [Step 3/6] Preprocessing spectral features (parallelized)..."
# We REMOVE --adaptive-k to force a consistent k=16
"$VENV_PYTHON" batch_process_spectral.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$SPECTRAL_DIR" \
    --k $SPECTRAL_K \
    --num_workers $NUM_WORKERS_SPECTRAL
    # --adaptive-k has been REMOVED
echo "✅ Spectral features regenerated in $SPECTRAL_DIR with k=$SPECTRAL_K."
echo "-------------------------------------------------"

# 4. Model Training (MODIFIED FOR SOTA)
echo "🧠 [Step 4/6] Starting model training..."
echo "   - Model: FixedTemporalSpectralGNN (SOTA 3-Pathway)"
echo "   - Dataset: StepPredictionDataset (w/ Contrastive Sampling)"
echo "   - Loss: $LOSS_TYPE + SOTA Value Loss + Contrastive Loss"
"$VENV_PYTHON" train.py \
    --data-dir "$DATA_DIR" \
    --spectral-dir "$SPECTRAL_DIR" \
    --exp-dir "$EXP_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --hidden-dim $HIDDEN_DIM \
    --num-layers $NUM_LAYERS \
    --lr $LEARNING_RATE \
    --value-loss-weight $VALUE_LOSS_WEIGHT \
    --contrast-loss-weight $CONTRAST_LOSS_WEIGHT \
    --loss-type $LOSS_TYPE \
    --margin $MARGIN \
    --k-dim $SPECTRAL_K
echo "✅ Training complete. Results in $EXP_DIR."
echo "================================================="

# 5. Post-Training Validation (Placeholder)
echo "🔬 [Step 5/6] Generating post-training validation reports..."
# "$VENV_PYTHON" validate_spectral.py \
#     --cache-dir "$SPECTRAL_DIR" \
#     --output-dir "$VALIDATION_DIR/spectral_report"
# "$VENV_PYTHON" validate_temporal.py \
#     --output-dir "$VALIDATION_DIR/temporal_report"
echo "✅ Validation reports placeholder complete."
echo "-------------------------------------------------"

# 6. Inference (Placeholder)
echo "🔍 [Step 6/6] Inference (Beam Search)..."
echo "   - The next step is to use search_agent.py with the trained model."
# "$VENV_PYTHON" run_search.py \
#     --model-path "$EXP_DIR/best.pt" \
#     --data-dir "$DATA_DIR" \ 
#     --output-file "$EXP_DIR/inference_results.json" \
#     --beam-width 5
echo "✅ Inference placeholder complete."
echo "-------------------------------------------------"

echo "🎉 Full SOTA Pipeline Finished Successfully!"
date
echo "================================================="

exit 0