#!/bin/bash

# Job script for t-SNE visualization only (requires pre-trained model)
# Usage: bash job_visualize_only.sh <checkpoint_path>

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Configuration
CONFIG_FILE="configs/unibifas.yaml"

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Error: Please provide checkpoint path as argument"
    echo "Usage: bash job_visualize_only.sh <checkpoint_path>"
    echo "Example: bash job_visualize_only.sh output/c_i_best.pt"
    exit 1
fi

CHECKPOINT_PATH="$1"

echo "=========================================="
echo "Creating t-SNE Visualization from Trained Model"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "=========================================="
echo ""

# Run visualization only
python visualize_only.py \
    --config $CONFIG_FILE \
    --ckpt $CHECKPOINT_PATH

echo ""
echo "=========================================="
echo "Visualization Complete!"
echo "=========================================="
